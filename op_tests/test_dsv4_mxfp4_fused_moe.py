# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import pytest
import torch

if os.environ.get("AITER_RUN_HEAVY_DSV4_TESTS") != "1":
    pytest.skip(
        "DSv4 routed MoE invariant test allocates multi-GB FP4 expert weights",
        allow_module_level=True,
    )
if not torch.cuda.is_available():
    pytest.skip("CUDA/HIP device is required", allow_module_level=True)

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4


def _rand_fp4(shape: tuple[int, ...], device: str) -> torch.Tensor:
    x = torch.empty(shape, device=device, dtype=dtypes.fp4x2)
    x.view(torch.uint8).random_(0, 256)
    return x


def test_dsv4_mxfp4_fused_moe_repeated_rows_invariant():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device is required")

    torch.manual_seed(0)
    device = "cuda"
    batch = int(os.environ.get("AITER_DSV4_MOE_TEST_BATCH", "63"))
    seqlen = int(os.environ.get("AITER_DSV4_MOE_TEST_SEQLEN", "88"))
    model_dim = 7168
    # DSv4-Pro TP8 local routed expert dim is 3072 / 8 = 384, padded to 512.
    intermediate = 384
    intermediate_padded = 512
    intermediate_pad = intermediate_padded - intermediate
    topk = 6
    num_experts = 384
    tokens = batch * seqlen

    pattern = torch.randn(seqlen, model_dim, device=device, dtype=torch.bfloat16)
    hidden = pattern.repeat(batch, 1).contiguous()

    ids_pattern = torch.randint(
        0, num_experts, (seqlen, topk), device=device, dtype=torch.int32
    )
    weights_pattern = torch.rand(seqlen, topk, device=device, dtype=torch.float32)
    weights_pattern = weights_pattern / weights_pattern.sum(dim=-1, keepdim=True)
    topk_ids = ids_pattern.repeat(batch, 1).contiguous()
    topk_weight = weights_pattern.repeat(batch, 1).contiguous()

    w1 = _rand_fp4((num_experts, 2 * intermediate_padded, model_dim // 2), device)
    w2 = _rand_fp4((num_experts, model_dim, intermediate_padded // 2), device)

    # Match ATOM's padded TP-local layout: random valid rows, zero padded rows,
    # then A16W4 preshuffle.
    w1_unpacked = w1.view(torch.uint8).view(
        num_experts, 2, intermediate_padded, model_dim // 2
    )
    w1_unpacked[:, :, intermediate:, :].zero_()
    w2.view(torch.uint8)[:, :, intermediate // 2 :].zero_()
    w1 = shuffle_weight_a16w4(w1, 16, True)
    w2 = shuffle_weight_a16w4(w2, 16, False)

    w1_scale = torch.empty(
        num_experts,
        2 * intermediate_padded,
        model_dim // 32,
        device=device,
        dtype=dtypes.fp8_e8m0,
    )
    w2_scale = torch.empty(
        num_experts,
        model_dim,
        intermediate_padded // 32,
        device=device,
        dtype=dtypes.fp8_e8m0,
    )
    w1_scale.view(torch.uint8).fill_(0x7F)
    w2_scale.view(torch.uint8).fill_(0x7F)
    w1_scale = shuffle_scale_a16w4(
        w1_scale.view(-1, w1_scale.shape[-1]), num_experts, True
    )
    w2_scale = shuffle_scale_a16w4(
        w2_scale.view(-1, w2_scale.shape[-1]), num_experts, False
    )

    out = fused_moe(
        hidden,
        w1,
        w2,
        topk_weight,
        topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        hidden_pad=0,
        intermediate_pad=intermediate_pad,
    )

    view = out.view(batch, seqlen, model_dim).float()
    diff = (view - view[:1]).abs()
    max_abs = diff.max().item()
    if max_abs > 1e-3:
        bad = (diff.amax(dim=2) > 1e-3).nonzero()
        raise AssertionError(
            "DSv4 MXFP4 fused_moe repeated-row invariant failed: "
            f"{max_abs=}, first_bad={bad[:8].detach().cpu().tolist()}, "
            f"{tokens=}"
        )


def test_dsv4_mxfp4_fused_moe_row0_repeated_replay_invariant():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device is required")

    torch.manual_seed(1)
    device = "cuda"
    M = int(os.environ.get("AITER_TEST_MOE_M", "16"))
    model_dim = 7168
    # DSv4-Pro TP8 local routed expert dim is 3072 / 8 = 384, padded to 512.
    intermediate = 384
    intermediate_padded = 512
    intermediate_pad = intermediate_padded - intermediate
    topk = 6
    num_experts = 384

    row = torch.randn(1, model_dim, device=device, dtype=torch.bfloat16)
    hidden = row.expand(M, model_dim).contiguous()
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device=device, dtype=torch.int32)
    topk_ids = ids.expand(M, topk).contiguous()
    weights = torch.full((1, topk), 1.0 / topk, device=device, dtype=torch.float32)
    topk_weight = weights.expand(M, topk).contiguous()

    w1 = _rand_fp4((num_experts, 2 * intermediate_padded, model_dim // 2), device)
    w2 = _rand_fp4((num_experts, model_dim, intermediate_padded // 2), device)

    # Match ATOM's padded TP-local layout: random valid rows, zero padded rows,
    # then A16W4 preshuffle.
    w1_unpacked = w1.view(torch.uint8).view(
        num_experts, 2, intermediate_padded, model_dim // 2
    )
    w1_unpacked[:, :, intermediate:, :].zero_()
    w2.view(torch.uint8)[:, :, intermediate // 2 :].zero_()
    w1 = shuffle_weight_a16w4(w1, 16, True)
    w2 = shuffle_weight_a16w4(w2, 16, False)

    w1_scale = torch.empty(
        num_experts,
        2 * intermediate_padded,
        model_dim // 32,
        device=device,
        dtype=dtypes.fp8_e8m0,
    )
    w2_scale = torch.empty(
        num_experts,
        model_dim,
        intermediate_padded // 32,
        device=device,
        dtype=dtypes.fp8_e8m0,
    )
    w1_scale.view(torch.uint8).fill_(0x7F)
    w2_scale.view(torch.uint8).fill_(0x7F)
    w1_scale = shuffle_scale_a16w4(
        w1_scale.view(-1, w1_scale.shape[-1]), num_experts, True
    )
    w2_scale = shuffle_scale_a16w4(
        w2_scale.view(-1, w2_scale.shape[-1]), num_experts, False
    )

    out = fused_moe(
        hidden,
        w1,
        w2,
        topk_weight,
        topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        hidden_pad=0,
        intermediate_pad=intermediate_pad,
    )

    diff = (out.float() - out[:1].float()).abs()
    max_abs = diff.max().item()
    assert max_abs <= 1e-3, f"{max_abs=}"
