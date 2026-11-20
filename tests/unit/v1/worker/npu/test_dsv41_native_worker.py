# SPDX-License-Identifier: Apache-2.0
"""Native-control backend selection contracts; no NPU execution."""

import json
import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm_ascend.models.deepseek_v41.vl_model import (  # noqa: E402
    AscendDeepseekV41ForCausalLM,
)

from afd_plugin.model_executor.models.npu.dsv41_packed_cache import (  # noqa: E402
    AFDMixedQuantPackedCacheOps,
    native,
)
from afd_plugin.v1.worker.npu import dsv41_native_worker  # noqa: E402
from afd_plugin.v1.worker.npu.dsv41_native_worker import (  # noqa: E402
    DSV41NativeTP4ControlWorker,
    DSV41NativeTP4PrefillTraceWorker,
    NPUWorker,
)
from tests.unit.fakes.dsv41_packed_cache import make_native_control_case  # noqa: E402


@pytest.fixture
def native_control_case(monkeypatch):
    case = make_native_control_case(
        DSV41NativeTP4ControlWorker, native.MixedQuantPackedCacheOps
    )

    def load(owner):
        owner.model_runner = SimpleNamespace(model=case.model)

    monkeypatch.setattr(NPUWorker, "load_model", load)
    return case


def test_native_control_changes_only_its_attention_backend(native_control_case):
    case = native_control_case
    native_qsmla = native.MixedQuantPackedCacheOps.qsmla
    case.worker.load_model()

    assert case.model["attention"].packed_cache_ops is AFDMixedQuantPackedCacheOps
    assert case.model["unrelated"].packed_cache_ops is native.MixedQuantPackedCacheOps
    assert native.MixedQuantPackedCacheOps.qsmla is native_qsmla


@pytest.mark.parametrize("backend", [None, type])
def test_native_control_does_not_replace_other_backends(native_control_case, backend):
    case = native_control_case
    case.model["attention"].packed_cache_ops = backend

    with pytest.raises(ValueError, match="no A5 packed Attention"):
        case.worker.load_model()
    assert case.model["attention"].packed_cache_ops is backend


def test_native_load_failure_keeps_original_error(native_control_case, monkeypatch):
    def failed_load(owner):
        raise RuntimeError("original weight load failure")

    monkeypatch.setattr(NPUWorker, "load_model", failed_load)
    with pytest.raises(RuntimeError, match="original weight load failure"):
        native_control_case.worker.load_model()
    assert (
        native_control_case.model["attention"].packed_cache_ops
        is native.MixedQuantPackedCacheOps
    )


def test_native_control_rejects_other_tp_sizes_before_loading(native_control_case):
    native_control_case.worker.vllm_config.parallel_config.tensor_parallel_size = 1
    with pytest.raises(ValueError, match="requires TP4"):
        native_control_case.worker.load_model()


@pytest.fixture
def prefill_trace_worker(monkeypatch):
    class LogitsProcessor(torch.nn.Module):
        def forward(self, head, hidden_states):
            logits = torch.zeros((1, 2048), dtype=hidden_states.dtype)
            logits[0, 223] = 4
            return logits

    wrapper_class = AscendDeepseekV41ForCausalLM
    model = wrapper_class.__new__(wrapper_class)
    torch.nn.Module.__init__(model)
    language_class = wrapper_class.language_model_cls
    language_model = language_class.__new__(language_class)
    torch.nn.Module.__init__(language_model)
    language_model.logits_processor = LogitsProcessor()
    language_model.lm_head = torch.nn.Identity()
    model.language_model = language_model
    batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=0,
        req_ids=["cmpl-trace-0"],
        prefill_len_np=[2],
        num_computed_tokens_np=[0],
        num_tokens=2,
        input_ids=torch.tensor([43, 16]),
        positions=torch.tensor([0, 1]),
        logits_indices=torch.tensor([1]),
    )
    worker = object.__new__(DSV41NativeTP4PrefillTraceWorker)
    worker.rank = 2
    worker.model_runner = SimpleNamespace(
        execute_model_state=SimpleNamespace(input_batch=batch),
        model=model,
    )

    def sample(owner, grammar_output):
        logits = owner.model_runner.model.compute_logits(
            torch.ones((1, 4), dtype=torch.bfloat16)
        )
        logits[0, 223] = 2
        return logits

    monkeypatch.setattr(NPUWorker, "sample_tokens", sample)
    monkeypatch.setattr(dsv41_native_worker, "init_logger", logging.getLogger)
    return worker, batch


@pytest.mark.parametrize("step", [1, 2, 3])
def test_native_prefill_trace_reports_unmodified_scores_for_first_two_steps(
    prefill_trace_worker, caplog, step
):
    worker, batch = prefill_trace_worker
    if step > 1:
        batch.num_tokens = 1
        batch.num_computed_tokens_np = [step]
        batch.input_ids = torch.tensor([223])
        batch.positions = torch.tensor([step])
        batch.logits_indices = torch.tensor([0])

    with caplog.at_level(logging.INFO):
        sampled = worker.sample_tokens(None)

    assert sampled[0, 223].item() == 2
    marker = "AFD native prefill trace "
    messages = [m.removeprefix(marker) for m in caplog.messages if m.startswith(marker)]
    if step == 3:
        assert messages == []
        return
    (message,) = messages
    record = json.loads(message)
    assert (record["request_id"], record["rank"], record["step"]) == (
        "cmpl-trace-0",
        2,
        step,
    )
    assert record["input_ids"] == batch.input_ids.tolist()
    assert record["positions"] == batch.positions.tolist()
    assert record["hidden_state"]["dtype"] == "torch.bfloat16"
    assert record["raw_logits"]["top5"][0] == [223, 4]
    assert record["raw_logits"]["candidates"]["223"] == 4
