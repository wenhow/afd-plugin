---
title: DSV4.1 packed attention head adapter
kind: module
status: draft
owners:
  - "@hsliuustc0106"
  - "@jiangkuaixue123"
primary_code_paths:
  - "afd_plugin/model_executor/models/npu/dsv41_packed_cache.py"
related_code_paths:
  - "afd_plugin/model_executor/models/npu/deepseek_v41.py"
depends_on:
  - "model_integration.md"
validation_paths:
  - "tests/unit/model_executor/models/npu/test_dsv41_packed_cache.py"
  - "tests/npu/dsv41_packed_attention_probe.py"
upstream_refs:
  - "vllm-ascend 7e37e29a7803d6897a7113088e2b771f83205bc0"
verified_platform_refs:
  - "CPU headwise reference only; packed A5 kernel acceptance pending"
last_reviewed: 2026-11-12
---

# DSV4.1 packed attention head adapter

The deployed A5 ds41 wheel rejects the first AFD TP4 request because it accepts
64 query heads while each Attention rank supplies 16. The native packed-cache
metadata builder already uses 64 heads. `AFDDeepseekV41Attention` preserves the
native constructor and replaces only its exact `MixedQuantPackedCacheOps`
backend class with `AFDMixedQuantPackedCacheOps`. A3's `None` and other backend
classes retain their original selection. No upstream class or operator is
modified globally; the native DSpark draft retains its native construction.

The adapter inherits the native cache, indexer and metadata ABI and overrides
only `qsmla`, with the same parameters, defaults and lack of annotations as
`mixed_quant_attention.py::qsmla` at the locked Ascend revision. For 16 heads,
it pads TND queries and sinks with 48 zero heads, delegates to native `qsmla`,
and returns the first 16 output heads as a contiguous tensor. For other head
counts, it delegates unchanged, including native 64-head output identity and
the native rejection of unsupported counts. This is a TP4 candidate, not a
general qualification of other tensor parallel sizes.

Each head attends independently to the shared K=V cache. For a head `h`, its
output is `sum(exp(q_h K_j * scale) * V_j) /
(exp(sink_h) + sum(exp(q_h K_j * scale)))`. Adding heads cannot change the
original heads' numerator or denominator. Native `qsmla` continues to own
window/compressed sparse indices and lengths, block tables, metadata waits,
FP32 detached sink conversion, and packed operator dispatch. Original queries,
sinks, shared cache tensors and metadata are passed or read without mutation;
only the padded Q/sinks and cropped output are allocated here.

Delegation is intentional: copying native `qsmla` or its large Attention
constructor would duplicate device metadata and shared-cache ownership for a
shape-only adaptation. Native lifetime, checkpoint parameter names, shared-KV
ownership, HC/pre_mix and Engram behavior remain with the native components.
Remove these two subclasses when the pinned A5 wheel accepts native 16 heads.

At TP4 the packed attention computes four times as many query heads. With
BF16 and head dimension 512, padding adds 48 KiB of query storage per token;
the 64-head output uses 64 KiB per token, followed by a 16 KiB contiguous crop.
At 1,024 tokens, padded Q and output each occupy 64 MiB, plus a 16 MiB crop;
actual peak depends on native kernel workspace and tensor lifetimes. KV storage,
token sparse indices, metadata and A/F communication are not replicated. This
workaround requires a device latency/memory measurement before performance
qualification.

The CPU tests use explicit headwise softmax with a sink, shared original and
compressed cache rows, token-dependent sparse indices/lengths, multiple tokens,
and FP32/BF16 queries. They verify retained-head results, contiguous BF16
outputs, unchanged input/cache bytes, native 64-head output identity, the
32-head native error, and local backend selection. They do not execute packed
quantization, NPU kernels, HCCL, or full-model inference. A5 operator comparison
and a fresh full-model short request remain separate acceptance gates.


The A5 launcher runs a small real-kernel probe before model startup, in a
separate supervised process. It creates native 544-byte SWA and 320-byte
compressed cache rows, uses native cache writes, window indices and 64-head
scheduling metadata, and compares three-query prefill and one-query decode
with and without compressed KV. Native 64-head zero-tail and nonzero-tail
baselines run before the adapter, each with a synchronization and fresh task
metadata. The shared scheduling-length carrier is conservatively 128 (the
sparse index width); actual per-query original/compressed lengths are supplied
separately to compute. A failed native baseline is recorded separately from
the adapter stage. This probe checks component shape and head independence;
it does not establish quantization accuracy against unquantized attention,
full-model correctness, HCCL support, or performance.
