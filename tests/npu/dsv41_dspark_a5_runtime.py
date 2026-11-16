# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5-only native bootstrap and independent packed-SWA component reference.

The scalar reference follows locked KvCompressEpilogV2 FP8-g32, round_scale=True:
FP32 amax/448, power-of-two ceiling, E4M3FN ties-to-even, then BF16 scales.
It never reads an operator output to construct its expected bytes. Runtime
imports occur only in explicit worker leaves; no native method is replaced.
"""

from __future__ import annotations

import hashlib
import math
import struct
from bisect import bisect_left
from pathlib import Path

HIDDEN_SIZE = 512
QUANT_GROUP_SIZE = 32
SCALE_COUNT = HIDDEN_SIZE // QUANT_GROUP_SIZE
PACKED_ROW_BYTES = HIDDEN_SIZE + SCALE_COUNT * 2
FP8_MAX = 448.0
AMAX_FLOOR = 1.0e-4
NATIVE_ATTENTION_HEADS = 64


def fp32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if not math.isfinite(value):
        raise ValueError("the bounded packed-SWA reference requires finite values")
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    if rounded & 0x7F80 == 0x7F80:
        raise ValueError("input overflows finite BF16 in the bounded reference")
    return struct.pack("<H", rounded)


def bf16_value(raw: bytes) -> float:
    return struct.unpack("<f", b"\0\0" + raw)[0]


def fp8_value(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    exponent, mantissa = (code >> 3) & 0xF, code & 7
    if exponent == 0:
        return sign * math.ldexp(mantissa / 8, -6)
    if exponent == 15 and mantissa == 7:
        raise ValueError("NaN E4M3FN byte is outside the controlled cache reference")
    return sign * math.ldexp(1 + mantissa / 8, exponent - 7)


FP8_POSITIVE_VALUES = tuple(fp8_value(code) for code in range(127))


def fp8_encode(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("the packed reference does not accept NaN or infinity")
    sign = 0x80 if math.copysign(1, value) < 0 else 0
    magnitude = min(abs(value), FP8_MAX)
    # E4M3FN positive values are ordered by their seven-bit representation.
    # The low bit is the destination significand parity for ties-to-even.
    upper = bisect_left(FP8_POSITIVE_VALUES, magnitude)
    lower = max(upper - 1, 0)
    code = min(
        (lower, upper),
        key=lambda code: (abs(FP8_POSITIVE_VALUES[code] - magnitude), code & 1),
    )
    return sign | code


def encode_swa_row(values: list[float]) -> bytes:
    if len(values) != HIDDEN_SIZE:
        raise ValueError("packed SWA requires 512 BF16 input values")
    data, scales = bytearray(), bytearray()
    coefficient = fp32(1.0 / FP8_MAX)
    for offset in range(0, HIDDEN_SIZE, QUANT_GROUP_SIZE):
        group = [
            bf16_value(bf16(value))
            for value in values[offset : offset + QUANT_GROUP_SIZE]
        ]
        maximum = max(max(abs(value) for value in group), fp32(AMAX_FLOOR))
        scale = fp32(maximum * coefficient)
        bits = struct.unpack("<I", struct.pack("<f", scale))[0]
        exponent = (bits >> 23) & 0xFF
        exponent += bool(bits & 0x7FFFFF)
        scale = struct.unpack("<f", struct.pack("<I", exponent << 23))[0]
        scales.extend(bf16(scale))
        data.extend(fp8_encode(fp32(value / scale)) for value in group)
    assert len(data) + len(scales) == PACKED_ROW_BYTES
    return bytes(data + scales)


def decode_swa_row(raw: bytes) -> list[float]:
    if len(raw) != PACKED_ROW_BYTES:
        raise ValueError("packed SWA row must contain exactly 544 bytes")
    values = []
    for index, code in enumerate(raw[:HIDDEN_SIZE]):
        start = HIDDEN_SIZE + 2 * (index // QUANT_GROUP_SIZE)
        scale = bf16_value(raw[start : start + 2])
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("invalid packed-SWA BF16 scale")
        values.append(bf16_value(bf16(fp32(fp8_value(code) * scale))))
    return values


def read_reference_row(cache, raw: bytes):
    import torch

    if cache.dtype == torch.uint8:
        assert cache.shape[-1] == PACKED_ROW_BYTES
        return torch.tensor(decode_swa_row(raw), dtype=torch.bfloat16, device="cpu")
    assert cache.dtype == torch.bfloat16 and cache.shape[-1] == HIDDEN_SIZE
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()


def encode_reference_row(cache, row) -> bytes:
    import torch

    if cache.dtype == torch.uint8:
        assert cache.shape[-1] == PACKED_ROW_BYTES
        return encode_swa_row(row.float().tolist())
    assert cache.dtype == torch.bfloat16 and cache.shape[-1] == HIDDEN_SIZE
    return row.contiguous().view(torch.uint8).numpy().tobytes()


def controlled_prefix_attention(cache, slots, positions, hidden_states):
    """Keep controlled arithmetic, with the native A5 packed writer unchanged."""
    import torch
    from vllm_ascend.device.device_op import DeviceOperator

    assert cache.ndim == 4 and slots.shape == (hidden_states.shape[0], 2)
    assert cache.dtype == torch.uint8 and cache.shape[-1] == PACKED_ROW_BYTES
    assert cache.stride(-1) == 1 and cache.stride(1) == PACKED_ROW_BYTES
    assert cache.stride(0) >= cache.shape[1] * PACKED_ROW_BYTES
    block = slots[:, 0].clamp_min(0).long()
    position = slots[:, 1].clamp_min(0).long()
    rows = cache[block, (position - 1).clamp_min(0), 0]
    values = rows[:, :HIDDEN_SIZE].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows[:, HIDDEN_SIZE:].contiguous().view(torch.bfloat16).float()
    previous = values.reshape(-1, SCALE_COUNT, QUANT_GROUP_SIZE) * scales[:, :, None]
    previous = previous.reshape(-1, HIDDEN_SIZE).bfloat16()
    previous = previous.masked_fill((slots[:, 0] <= 0).unsqueeze(1), 0)
    flat_slots = slots[:, 0] * cache.shape[1] + slots[:, 1]
    flat_slots = flat_slots.masked_fill(slots[:, 0] <= 0, -1)
    DeviceOperator.get_dsv41_packed_cache_ops().write_attention_cache(
        cache, flat_slots, hidden_states, kind="win"
    )
    return (hidden_states.float() + previous.float() + positions[:, None]).bfloat16()


def native_a5_bootstrap() -> dict:
    """Load the installed native A5 leaves without enabling optional custom ops.

    The direct extension import follows native platform physical-device lookup,
    and is needed by the unchanged V4.1 partial rotary call. A5 wheel HC, packed
    cache and mixed-quant Attention remain the original native implementation.
    No hardware capability, dispatcher function or kernel is replaced.
    """
    import cann_ops_transformer
    import custom_ops
    import torch
    from vllm.platforms import current_platform
    from vllm_ascend.device.hardware_profile import (
        HardwareCapability,
        get_current_hardware_profile,
    )
    from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

    profile = get_current_hardware_profile()
    assert not profile.supports(HardwareCapability.RUNTIME_CUSTOM_OPS)
    assert profile.supports(HardwareCapability.DSV41_PACKED_CACHE)
    result = enable_custom_op()
    assert result is False
    current_platform.import_kernels()
    bootstrap_custom_op_env()
    from vllm_ascend import vllm_ascend_C

    schema = torch.ops._C_ascend.inplace_partial_rotary_mul.default._schema
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "_C_ascend::inplace_partial_rotary_mul", "PrivateUse1"
    ):
        raise RuntimeError("the installed A5 extension lacks native partial rotary")
    schemas = {
        "partial_rotary": str(schema),
        "hc_pre": str(torch.ops.custom.npu_hc_pre_v2.default._schema),
        "hc_post": str(torch.ops.cann_ops_transformer.mhc_post.default._schema),
        "packed_swa_writer": str(
            torch.ops.custom.kv_compress_epilog_v2.default._schema
        ),
    }
    packed_dispatch = {
        key: torch._C._dispatch_has_kernel_for_dispatch_key(
            "custom::kv_compress_epilog_v2", key
        )
        for key in (
            "PrivateUse1",
            "CompositeExplicitAutograd",
            "CompositeImplicitAutograd",
        )
    }
    if not any(packed_dispatch.values()):
        raise RuntimeError("the installed A5 wheel lacks a native packed-SWA writer")
    modules = {}
    for name, module in (
        ("vllm_ascend_C", vllm_ascend_C),
        ("custom_ops", custom_ops),
        ("cann_ops_transformer", cann_ops_transformer),
    ):
        path = Path(module.__file__).resolve(strict=True)
        modules[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "native_enable_custom_op_result": result,
        "runtime_custom_ops_capability": False,
        "packed_cache_capability": True,
        "native_partial_rotary_kernel_registered": True,
        "direct_installed_extension_import": True,
        "replaced_native_method_or_kernel": False,
        "modules": modules,
        "schemas": schemas,
        "packed_writer_dispatch": packed_dispatch,
    }
