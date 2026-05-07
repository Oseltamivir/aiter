// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_blockscale_bpreshuffle_common.cuh"
#include "gemm_a8w8_blockscale_bpreshuffle_lookup.h"
#include "gemm_common.h"
#include "gemm_a8w8_blockscale_bpreshuffle_manifest.h"
#include "gemm_dispatch_utils.h"

#include <cmath>

using BlockwiseKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;

using BlockwiseKernelMap = GemmDispatchMap<BlockwiseKernel>;

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num)
{
    if(num <= 1)
        return 1;
    return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename DDataType, typename EDataType = DDataType>
BlockwiseKernel blockscale_bpreshuffle_dispatch(int M, int N, int K)
{
    // For a given shape, either find the best kernel via lookup or heuristic.
    // For many small M shapes, we bucket them to the next largest kernel.
    // This is fine since kernels are padded anyway.

    static const auto lookup = [] {
        if constexpr(std::is_same_v<EDataType, F16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, F16)};
        }
        else if constexpr(std::is_same_v<EDataType, B16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, B16)};
        }
        else
        {
            static_assert(false, "blockscale_bpreshuffle_dispatch used with unsupported dtype!");
        }
    }();

    const int cu_num         = get_device_cu_num();
    const std::string& gfx   = get_device_gfx();

    // DSv4-Pro TP8 FP8 blockscale projection family:
    //   wkv:            [M, 7168] x [512, 7168]
    //   shared gate_up: [M, 7168] x [768, 7168]
    //   wq_a:           [M, 7168] x [1536, 7168]
    //
    // Check this before exact/padded lookup. Some table hits are not
    // correctness-safe for identical rows in DSv4 batched prefill.
    if(K == 7168 && (N == 512 || N == 768 || N == 1536))
    {
        return a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<
            DDataType,
            EDataType>;
    }

    // DSv4-Pro shared expert w2 under TP8:
    //   [M, 384] x [7168, 384] -> [M, 7168]
    //
    // This shape is not in the tuned table. Without an explicit override it
    // falls through to the generic heuristic, which corrupts identical rows at
    // high concurrency. Keep this before exact/padded lookup so future table
    // changes cannot bypass the safe path.
    if(N == 7168 && K == 384)
    {
        return a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<
            DDataType,
            EDataType>;
    }

    // DSv4-Pro wo_b under TP8 has local GEMM shape [M, 2048] x [7168, 2048].
    // Keep this before lookup as well so direct/padded table hits cannot
    // bypass the DSv4-safe path.
    if(N == 7168 && K == 2048)
    {
        return a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<
            DDataType,
            EDataType>;
    }

    // First check if this shape(M,N,K) is available in the direct lookup.
    auto it = lookup.find({gfx, cu_num, M, N, K});
    // If we found an optimal kernel, use it.
    if(it != lookup.end())
    {
        return it->second;
    }

    int padded_m = M;

    // Fine-grained search
    padded_m = getPaddedM(M, N, K, 0);

    // Second check if this shape(padded_m,N,K) is available in the direct lookup.
    it = lookup.find({gfx, cu_num, padded_m, N, K});
    // If we found an optimal kernel, use it.
    if(it != lookup.end())
    {
        return it->second;
    }

    // Coarse-grained search
    padded_m = getPaddedM(M, N, K, 1);
    it = lookup.find({gfx, cu_num, padded_m, N, K});
    if(it != lookup.end())
    {
        return it->second;
    }

    // Otherwise, use heuristics.
    return a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<
        DDataType,
        EDataType>;
}

torch::Tensor gemm_a8w8_blockscale_bpreshuffle(torch::Tensor& XQ,
                                   torch::Tensor& WQ,
                                   torch::Tensor& x_scale,
                                   torch::Tensor& w_scale,
                                   torch::Tensor& Y)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
    {
        blockscale_bpreshuffle_dispatch<F32, F16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    }
    else if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
        blockscale_bpreshuffle_dispatch<F32, B16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}
