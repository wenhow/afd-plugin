from __future__ import annotations

import importlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _install_fake_modules(monkeypatch: pytest.MonkeyPatch):
    class TokenDispatcherWithAll2AllV:
        def _dispatch_postprocess(
            self,
            global_input_tokens,
            dynamic_scale_after_all2all,
            global_input_tokens_local_experts_indices,
            with_quant,
            dst_type,
            scale_type,
        ):
            return (
                global_input_tokens,
                dynamic_scale_after_all2all,
                global_input_tokens_local_experts_indices,
                with_quant,
                dst_type,
                scale_type,
            )

    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu_moe_token_permute = None

    def obsolete_init_routing(*args, **kwargs):
        del args, kwargs
        raise AssertionError("MoeInitRoutingV3 must not handle the MXFP payload")

    torch_npu.npu_moe_init_routing_v2 = obsolete_init_routing
    root = types.ModuleType("vllm_ascend")
    root.__path__ = []
    ops = types.ModuleType("vllm_ascend.ops")
    ops.__path__ = []
    fused_moe = types.ModuleType("vllm_ascend.ops.fused_moe")
    fused_moe.__path__ = []
    token_dispatcher = types.ModuleType(
        "vllm_ascend.ops.fused_moe.token_dispatcher",
    )
    token_dispatcher.TokenDispatcherWithAll2AllV = TokenDispatcherWithAll2AllV

    def vulnerable_byte_helper(payload, expert_indices, *, output_dtype):
        payload_bytes = payload.reshape(payload.shape[0], -1).view(torch.int8)
        permuted, reverse_mapping = torch_npu.npu_moe_token_permute(
            payload_bytes,
            expert_indices,
        )
        return permuted.view(output_dtype), reverse_mapping

    token_dispatcher._permute_mxfp_byte_payload = vulnerable_byte_helper

    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "vllm_ascend", root)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", ops)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe", fused_moe)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.token_dispatcher",
        token_dispatcher,
    )
    return TokenDispatcherWithAll2AllV, torch_npu


@pytest.fixture
def mxfp_patch(monkeypatch: pytest.MonkeyPatch):
    dispatcher_cls, torch_npu = _install_fake_modules(monkeypatch)
    module_name = "afd_plugin.compat.patches.npu.mxfp_alltoallv"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    return module, dispatcher_cls, torch_npu


def test_mxfp_alltoallv_patch_preserves_scale_layout(mxfp_patch):
    module, dispatcher_cls, torch_npu = mxfp_patch
    order = torch.tensor([1, 3, 0, 2], dtype=torch.int64)
    permute_calls = []

    def permute(payload, expert_indices):
        permute_calls.append((payload.clone(), expert_indices))
        assert payload.dtype == torch.int8
        assert payload.ndim == 2
        return payload.index_select(0, order), order.to(torch.int32)

    torch_npu.npu_moe_token_permute = permute
    assert module.apply_afd_mxfp_alltoallv_patch()

    dispatcher = object.__new__(dispatcher_cls)
    dispatcher.num_local_experts = 2
    tokens = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
    scales = torch.arange(24, dtype=torch.uint8).reshape(4, 3, 2)
    expert_indices = torch.tensor([1, 0, 1, 0], dtype=torch.int32)

    routed_tokens, routed_scales, reverse_mapping = dispatcher._dispatch_postprocess(
        tokens,
        scales,
        expert_indices,
        True,
        torch.float8_e4m3fn,
        torch.float8_e8m0fnu,
    )

    assert len(permute_calls) == 2
    assert all(call[1] is expert_indices for call in permute_calls)
    assert torch.equal(routed_tokens, tokens.index_select(0, order))
    assert routed_scales.dtype == torch.uint8
    assert routed_scales.shape == scales.shape
    assert torch.equal(routed_scales, scales.index_select(0, order))
    assert torch.equal(reverse_mapping, order.to(torch.int32))


def test_mxfp_alltoallv_patch_is_idempotent(mxfp_patch):
    module, dispatcher_cls, _ = mxfp_patch

    assert module.apply_afd_mxfp_alltoallv_patch()
    patched_postprocess = dispatcher_cls._dispatch_postprocess
    assert module.apply_afd_mxfp_alltoallv_patch()

    assert dispatcher_cls._dispatch_postprocess is patched_postprocess


def test_mxfp_byte_helper_accepts_empty_payload(mxfp_patch):
    module, _, torch_npu = mxfp_patch
    seen_shapes = []

    def permute(payload, expert_indices):
        seen_shapes.append(tuple(payload.shape))
        assert payload.dtype == torch.int8
        assert expert_indices.numel() == 0
        return payload, torch.empty((0,), dtype=torch.int32)

    torch_npu.npu_moe_token_permute = permute
    expert_indices = torch.empty((0,), dtype=torch.int32)
    routed, reverse_mapping = module._permute_mxfp_byte_payload(
        torch.empty((0, 3, 2), dtype=torch.uint8),
        expert_indices,
        output_dtype=torch.uint8,
    )

    assert seen_shapes == [(0, 6)]
    assert routed.shape == (0, 3, 2)
    assert reverse_mapping.shape == (0,)


def test_mxfp_patch_replaces_f87_empty_payload_helper(mxfp_patch):
    module, dispatcher_cls, torch_npu = mxfp_patch
    token_dispatcher = sys.modules["vllm_ascend.ops.fused_moe.token_dispatcher"]
    original_helper = token_dispatcher._permute_mxfp_byte_payload

    def fixed_layout_postprocess(
        self,
        global_input_tokens,
        dynamic_scale_after_all2all,
        global_input_tokens_local_experts_indices,
        with_quant,
        scale_type,
    ):
        del (
            self,
            global_input_tokens,
            dynamic_scale_after_all2all,
            global_input_tokens_local_experts_indices,
            with_quant,
            scale_type,
        )

    dispatcher_cls._dispatch_postprocess = fixed_layout_postprocess
    assert module.apply_afd_mxfp_alltoallv_patch()
    assert (
        token_dispatcher._permute_mxfp_byte_payload is module._permute_mxfp_byte_payload
    )
    assert token_dispatcher._permute_mxfp_byte_payload is not original_helper

    torch_npu.npu_moe_token_permute = lambda payload, indices: (
        payload,
        indices,
    )
    routed, _ = token_dispatcher._permute_mxfp_byte_payload(
        torch.empty((0, 4), dtype=torch.uint8),
        torch.empty((0,), dtype=torch.int32),
        output_dtype=torch.uint8,
    )
    assert routed.shape == (0, 4)


def test_mxfp_alltoallv_patch_leaves_other_signatures_unchanged(mxfp_patch):
    module, dispatcher_cls, _ = mxfp_patch

    def old_postprocess(
        self,
        global_input_tokens,
        dynamic_scale_after_all2all,
        global_input_tokens_local_experts_indices,
        scale_type,
    ):
        del (
            self,
            global_input_tokens,
            dynamic_scale_after_all2all,
            global_input_tokens_local_experts_indices,
            scale_type,
        )

    dispatcher_cls._dispatch_postprocess = old_postprocess

    assert not module.apply_afd_mxfp_alltoallv_patch()
    assert dispatcher_cls._dispatch_postprocess is old_postprocess


def test_mxfp_alltoallv_patch_leaves_fixed_helper_unchanged(mxfp_patch):
    module, dispatcher_cls, _ = mxfp_patch
    token_dispatcher = sys.modules["vllm_ascend.ops.fused_moe.token_dispatcher"]

    def fixed_layout_postprocess(
        self,
        global_input_tokens,
        dynamic_scale_after_all2all,
        global_input_tokens_local_experts_indices,
        with_quant,
        scale_type,
    ):
        del (
            self,
            global_input_tokens,
            dynamic_scale_after_all2all,
            global_input_tokens_local_experts_indices,
            with_quant,
            scale_type,
        )

    def fixed_helper(payload, expert_indices, *, output_dtype):
        del expert_indices, output_dtype
        return payload

    dispatcher_cls._dispatch_postprocess = fixed_layout_postprocess
    token_dispatcher._permute_mxfp_byte_payload = fixed_helper

    assert not module.apply_afd_mxfp_alltoallv_patch()
    assert token_dispatcher._permute_mxfp_byte_payload is fixed_helper


def test_mxfp_alltoallv_patch_preserves_non_quantized_path(mxfp_patch):
    module, dispatcher_cls, torch_npu = mxfp_patch
    reverse_mapping = torch.tensor([1, 0], dtype=torch.int32)
    calls = []

    def permute(payload, expert_indices):
        calls.append((payload, expert_indices))
        return payload.flip(0), reverse_mapping

    torch_npu.npu_moe_token_permute = permute
    assert module.apply_afd_mxfp_alltoallv_patch()

    dispatcher = object.__new__(dispatcher_cls)
    dispatcher.num_local_experts = 2
    tokens = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    expert_indices = torch.tensor([1, 0], dtype=torch.int32)

    routed_tokens, routed_scale, actual_reverse_mapping = (
        dispatcher._dispatch_postprocess(
            tokens,
            None,
            expert_indices,
            False,
            torch.float8_e4m3fn,
            torch.float32,
        )
    )

    assert calls == [(tokens, expert_indices)]
    assert torch.equal(routed_tokens, tokens.flip(0))
    assert routed_scale is None
    assert actual_reverse_mapping is reverse_mapping


def test_worker_resolution_patch_covers_prefill_and_afd_workers(monkeypatch):
    worker_base = types.ModuleType("vllm.v1.worker.worker_base")
    resolved_worker = object()
    resolved_qualnames = []

    def original_resolver(qualname):
        resolved_qualnames.append(qualname)
        return resolved_worker

    worker_base.resolve_obj_by_qualname = original_resolver
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.worker_base", worker_base)

    apply_calls = []
    dispatcher_patch = types.ModuleType(
        "afd_plugin.compat.patches.npu.mxfp_alltoallv",
    )
    dispatcher_patch.apply_afd_mxfp_alltoallv_patch = lambda: apply_calls.append(True)
    monkeypatch.setitem(
        sys.modules,
        "afd_plugin.compat.patches.npu.mxfp_alltoallv",
        dispatcher_patch,
    )

    module_name = "afd_plugin.compat.patches.npu.mxfp_worker_resolution"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    assert module.apply_afd_mxfp_worker_resolution_patch()

    target_workers = (
        "vllm_ascend.worker.worker.NPUWorker",
        "afd_plugin.v1.worker.npu.AFDNPUAttentionWorker",
        "afd_plugin.v1.worker.npu.AFDNPUFFNWorker",
    )
    for qualname in target_workers:
        assert worker_base.resolve_obj_by_qualname(qualname) is resolved_worker

    assert resolved_qualnames == list(target_workers)
    assert apply_calls == [True, True, True]


def test_worker_resolution_patch_leaves_non_npu_worker_lazy(monkeypatch):
    worker_base = types.ModuleType("vllm.v1.worker.worker_base")
    resolved_worker = object()
    worker_base.resolve_obj_by_qualname = lambda qualname: resolved_worker
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.worker_base", worker_base)

    module_name = "afd_plugin.compat.patches.npu.mxfp_worker_resolution"
    dispatcher_module_name = "afd_plugin.compat.patches.npu.mxfp_alltoallv"
    sys.modules.pop(module_name, None)
    sys.modules.pop(dispatcher_module_name, None)
    module = importlib.import_module(module_name)
    assert module.apply_afd_mxfp_worker_resolution_patch()

    assert (
        worker_base.resolve_obj_by_qualname("vllm.v1.worker.gpu_worker.Worker")
        is resolved_worker
    )
    assert dispatcher_module_name not in sys.modules
