# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit graph compute forks; every HCCL scope remains on its parent.

This component owns stream/event ordering and allocator consumers, not tensor
transport, graph launch, phase negotiation or ModelState retirement. A graph
must retain it alongside the bank and body until reset. Manual CPU construction
is a host ordering seam; only ``allocate`` creates device streams and events.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import torch

from afd_plugin.compat.npu.stream import use_npu_current_stream
from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSelection


@dataclass(frozen=True, slots=True)
class HcclGraphPipelinePolicy:
    """One session's capture-constant compute DAG; None selects no pipeline."""

    hybrid: bool = True
    attention_three_stream: bool = True
    ffn_recv_stream: bool = True
    ffn_cross_layer: bool = True

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.hybrid,
                self.attention_three_stream,
                self.ffn_recv_stream,
                self.ffn_cross_layer,
            )
        ):
            raise ValueError("graph pipeline policy values must be bool")
        if self.ffn_cross_layer and not self.ffn_recv_stream:
            raise ValueError("graph FFN cross-layer policy requires receive stream")

    @classmethod
    def from_selection(
        cls, selection: HcclOverlapSelection
    ) -> HcclGraphPipelinePolicy | None:
        """Keep fixed settings while deriving each actual descriptor's eligibility.

        Unused role-specific fields remain configured values, so a retained
        policy does not lose the session's other axes. The selection exposes
        their effective role/mode gates separately.
        """
        if selection.phase == "eager":
            raise ValueError("graph pipeline selection must describe a Graph phase")
        if not selection.graph_u2_compute_overlap:
            return None
        settings = selection.settings
        return cls(
            hybrid=settings.graph_u2_hybrid_dag,
            attention_three_stream=settings.graph_u2_attention_three_stream,
            ffn_recv_stream=settings.graph_u2_ffn_recv_stream,
            ffn_cross_layer=settings.graph_u2_ffn_cross_layer,
        )


@dataclass(frozen=True, slots=True)
class HcclGraphPipelineEvents:
    """Distinct events for one static layer/stage, never shared with a peer."""

    ready: torch.npu.Event
    compute_done: torch.npu.Event
    receive_done: torch.npu.Event
    send_done: torch.npu.Event


@dataclass(frozen=True, slots=True)
class HcclGraphPipelineTailEvents:
    """Separate final parent fence and completion; no event is re-recorded."""

    ready: torch.npu.Event
    done: torch.npu.Event


class HcclGraphComputePipeline:
    """One retained side compute stream and a bounded per-stage dependency DAG.

    Source: donor 4ddd69a3 p2p_hccl.py:726-807,985-1099,1241-1317 and
    deepseek_v4.py:1067-1244. This is a new composition API, not an upstream
    function replacement. Unlike logical send/receive stream selection, these
    communication contexts always require the actual parent stream. Native
    V4.1 compute, continuations, Engram and TP remain with the caller.

    One factory phase records this object once; replay never re-enters Python.
    Owners must fix this policy for a session/cache/manager lifetime. Changing
    it needs fresh owners: the graph key does not encode stream policy, and a
    generation miss does not authorize recapturing an existing native graph.

    Removal: use native stage-aware stream/event hooks when they preserve these
    same per-stage receive dependencies and graph-visible final joins.
    """

    def __init__(
        self,
        bank: HcclGraphStageBank,
        compute_stream: torch.npu.Stream,
        step_events: Mapping[tuple[int, int], HcclGraphPipelineEvents],
        tail_events: Mapping[int, HcclGraphPipelineTailEvents],
        *,
        hybrid: bool | None = None,
        policy: HcclGraphPipelinePolicy | None = None,
        selection: HcclOverlapSelection | None = None,
    ) -> None:
        selected = self._select_policy(bank, hybrid, policy, selection)
        if compute_stream.device != bank.device:
            raise ValueError(
                "pipeline requires a same-device compute stream and bool mode"
            )
        steps = tuple(
            (layer, stage)
            for layer in bank.layer_indices
            for stage, rows in enumerate(bank.stage_rows)
            if rows
        )
        stages = tuple(stage for stage, rows in enumerate(bank.stage_rows) if rows)
        if set(step_events) != set(steps) or set(tail_events) != set(stages):
            raise ValueError(
                "pipeline events must cover exactly the positive bank layout"
            )
        events = tuple(
            event
            for step in steps
            for event in (
                step_events[step].ready,
                step_events[step].compute_done,
                step_events[step].receive_done,
                step_events[step].send_done,
            )
        ) + tuple(
            event for tail in tail_events.values() for event in (tail.ready, tail.done)
        )
        if len({id(event) for event in events}) != len(events):
            raise ValueError("pipeline dependency events must not alias")
        self.bank, self._compute_stream, self._policy = bank, compute_stream, selected
        self._selection = selection
        self.step_events = MappingProxyType(dict(step_events))
        self.tail_events = MappingProxyType(dict(tail_events))
        self._steps, self._stages = steps, stages
        self._layout = (
            bank.role,
            bank.layer_indices,
            bank.stage_rows,
            bank.device,
            bank.key,
        )
        self._parent: torch.npu.Stream | None = None
        self._active = self._failed = self._joined = self._recorded = False
        self._wire_index = self._compute_index = 0
        self._received: set[tuple[int, int]] = set()
        self._computed: set[tuple[int, int]] = set()
        self._sent: set[tuple[int, int]] = set()
        self._finalized: set[int] = set()
        self._joined_layers: set[int] = set()
        self._held: dict[int | None, tuple[torch.Tensor, ...]] = {}

    @property
    def compute_stream(self) -> torch.npu.Stream:
        return self._compute_stream

    @property
    def hybrid(self) -> bool:
        return self.policy.hybrid

    @property
    def policy(self) -> HcclGraphPipelinePolicy:
        return self._policy

    @property
    def selection(self) -> HcclOverlapSelection | None:
        return self._selection

    @property
    def logical_send_stream(self) -> str:
        if self.selection is not None:
            return self.selection.logical_send_stream
        return (
            "send"
            if self.bank.role == "ffn" or self.policy.attention_three_stream
            else "parent"
        )

    @property
    def logical_recv_stream(self) -> str:
        if self.selection is not None:
            return self.selection.logical_recv_stream
        selected = (
            self.policy.attention_three_stream
            if self.bank.role == "attention"
            else self.policy.ffn_recv_stream
        )
        return "recv" if selected else "parent"

    def communication_plan(self) -> dict[str, str | bool]:
        """Report donor logical choices and the actual Graph parent override.

        No dedicated communication streams are allocated to represent logical
        choices that warmup/capture override. This record is host diagnostics,
        not evidence of three physical streams or actual communication overlap.
        """
        return {
            "logical.send_stream": self.logical_send_stream,
            "logical.recv_stream": self.logical_recv_stream,
            "physical.send_stream": "parent",
            "physical.recv_stream": "parent",
            "parent_transport_override": True,
        }

    @staticmethod
    def _select_policy(
        bank: HcclGraphStageBank,
        hybrid: bool | None,
        policy: HcclGraphPipelinePolicy | None,
        selection: HcclOverlapSelection | None,
    ) -> HcclGraphPipelinePolicy:
        if sum(value is not None for value in (hybrid, policy, selection)) > 1:
            raise ValueError(
                "pass one graph pipeline policy, selection or legacy hybrid"
            )
        if selection is not None:
            if (
                selection.role != bank.role
                or selection.num_stages != len(bank.stage_rows)
                or selection.physical_send_stream != "parent"
                or selection.physical_recv_stream != "parent"
            ):
                raise ValueError("graph pipeline selection must match this bank/parent")
            selected = HcclGraphPipelinePolicy.from_selection(selection)
            if selected is None:
                raise ValueError("graph pipeline selection requires active U2 compute")
            return selected
        if policy is not None:
            return policy
        return HcclGraphPipelinePolicy(hybrid=True if hybrid is None else hybrid)

    @classmethod
    def allocate(
        cls,
        bank: HcclGraphStageBank,
        *,
        hybrid: bool | None = None,
        policy: HcclGraphPipelinePolicy | None = None,
        selection: HcclOverlapSelection | None = None,
    ) -> HcclGraphComputePipeline:
        """Create real NPU owners outside capture; no groups or buffers change."""
        cls._select_policy(bank, hybrid, policy, selection)
        if bank.device.type != "npu" or bank.device.index is None:
            raise ValueError("allocate pipeline on an indexed NPU bank")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("allocate pipeline streams and events outside capture")
        events = {
            (layer, stage): HcclGraphPipelineEvents(
                torch.npu.Event(),
                torch.npu.Event(),
                torch.npu.Event(),
                torch.npu.Event(),
            )
            for layer in bank.layer_indices
            for stage, rows in enumerate(bank.stage_rows)
            if rows
        }
        tails = {
            stage: HcclGraphPipelineTailEvents(torch.npu.Event(), torch.npu.Event())
            for stage, rows in enumerate(bank.stage_rows)
            if rows
        }
        return cls(
            bank,
            torch.npu.Stream(device=bank.device),
            events,
            tails,
            hybrid=hybrid,
            policy=policy,
            selection=selection,
        )

    @contextmanager
    def run(self) -> Iterator[None]:
        """Bind one parent; return only after every required final join."""
        if self._active or self._failed:
            raise RuntimeError(
                "pipeline is active or failed; retain failed device owners"
            )
        if self._recorded:
            raise RuntimeError(
                "pipeline events already recorded; use a fresh phase owner"
            )
        bank = self.bank
        if self._layout != (
            bank.role,
            bank.layer_indices,
            bank.stage_rows,
            bank.device,
            bank.key,
        ):
            raise RuntimeError("pipeline bank layout changed")
        parent = torch.npu.current_stream(bank.device)
        if parent.device != bank.device or parent == self.compute_stream:
            raise RuntimeError("pipeline requires a distinct same-device parent stream")
        self._parent, self._active, self._joined = parent, True, False
        self._recorded = True
        self._wire_index = self._compute_index = 0
        self._received.clear()
        self._computed.clear()
        self._sent.clear()
        self._finalized.clear()
        self._joined_layers.clear()
        try:
            yield
            self._require_parent(allow_joined=True)
            if not self._joined:
                raise RuntimeError("pipeline return requires all final joins")
        except BaseException:
            # A Python failure does not cancel queued device tasks. Keep stream,
            # event, bank and temporary references for external supervision.
            self._failed = True
            raise
        finally:
            self._active = False
            if not self._failed:
                self._held.clear()

    def _require_parent(self, *, allow_joined: bool = False) -> torch.npu.Stream:
        if not self._active or self._failed or (self._joined and not allow_joined):
            raise RuntimeError("pipeline requires an active unjoined run")
        assert self._parent is not None
        if torch.npu.current_stream(self.bank.device) != self._parent:
            raise RuntimeError("pipeline communication must stay on the parent stream")
        return self._parent

    def _communication_stream(self, *, send: bool) -> torch.npu.Stream:
        """Resolve logical choice under the donor's Graph transport override."""
        parent = self._require_parent()
        logical = self.logical_send_stream if send else self.logical_recv_stream
        if logical == "parent":
            return parent
        # Warmup and capture override dedicated logical send/receive choices.
        # Preserve their diagnostics without creating unused physical streams.
        return parent

    def _require_step(self, layer_idx: int, stage_idx: int) -> tuple[int, int]:
        step = (layer_idx, stage_idx)
        if step not in self.step_events:
            raise ValueError("pipeline step must name a positive static layer/stage")
        return step

    def _require_wire_step(self, step: tuple[int, int]) -> None:
        if (
            self._wire_index >= len(self._steps)
            or self._steps[self._wire_index] != step
        ):
            raise RuntimeError(
                "pipeline wire order must remain layer-major and stage-minor"
            )

    def _retain_on(
        self,
        tensors: tuple[torch.Tensor, ...],
        stream: torch.npu.Stream,
        stage_idx: int | None = None,
    ) -> None:
        if any(tensor.device != self.bank.device for tensor in tensors):
            raise ValueError("pipeline tensors must share the bank device")
        # Earlier consumers were already recorded on their actual streams.
        # Bound Python retention to one frontier per stage instead of extending
        # every layer's temporary lifetime through the entire graph capture.
        self._held[stage_idx] = tensors
        if self.bank.device.type == "npu":
            for tensor in tensors:
                tensor.record_stream(stream)

    @contextmanager
    def compute(
        self, layer_idx: int, stage_idx: int, *, inputs: tuple[torch.Tensor, ...] = ()
    ) -> Iterator[None]:
        """A waits its prior receive; F waits its current receive before compute."""
        parent = self._require_parent()
        step = self._require_step(layer_idx, stage_idx)
        if (
            self._compute_index >= len(self._steps)
            or self._steps[self._compute_index] != step
        ):
            raise RuntimeError(
                "pipeline compute order must remain layer-major and stage-minor"
            )
        events = self.step_events[step]
        if self.bank.role == "ffn":
            dependency = step
        else:
            index = self.bank.layer_indices.index(layer_idx)
            dependency = (
                (self.bank.layer_indices[index - 1], stage_idx) if index else None
            )
        if dependency is not None and dependency not in self._received:
            raise RuntimeError("pipeline compute requires its matching stage receive")
        if (
            self.bank.role == "attention"
            and dependency is not None
            and any(
                (dependency[0], stage) not in self._received for stage in self._stages
            )
        ):
            raise RuntimeError("enqueue every preceding layer exchange before compute")
        if dependency is not None and (self.bank.role == "ffn" or self.hybrid):
            ready = self.step_events[dependency].receive_done
        else:
            events.ready.record(parent)
            ready = events.ready
        try:
            with use_npu_current_stream(self.compute_stream):
                ready.wait(self.compute_stream)
                self._retain_on(inputs, self.compute_stream, stage_idx)
                yield
                if torch.npu.current_stream(self.bank.device) != self.compute_stream:
                    raise RuntimeError(
                        "pipeline compute must stay on its compute stream"
                    )
                events.compute_done.record(self.compute_stream)
            self._computed.add(step)
            self._compute_index += 1
        except BaseException:
            self._failed = True
            raise

    @contextmanager
    def attention_exchange(
        self, layer_idx: int, stage_idx: int, *, tensors: tuple[torch.Tensor, ...] = ()
    ) -> Iterator[None]:
        """Join the actual producer before BF16/FP32/IDs copies and parent HCCL."""
        send_stream = self._communication_stream(send=True)
        step = self._require_step(layer_idx, stage_idx)
        self._require_wire_step(step)
        if self.bank.role != "attention" or step not in self._computed:
            raise RuntimeError(
                "Attention exchange requires its completed compute frontier"
            )
        if any((layer_idx, stage) not in self._computed for stage in self._stages):
            raise RuntimeError("enqueue every stage compute before this layer exchange")
        events = self.step_events[step]
        events.compute_done.wait(send_stream)
        self._retain_on(tensors, send_stream, stage_idx)
        try:
            yield
            events.receive_done.record(self._communication_stream(send=False))
            self._received.add(step)
            self._wire_index += 1
        except BaseException:
            self._failed = True
            raise

    @contextmanager
    def ffn_receive(self, layer_idx: int, stage_idx: int) -> Iterator[None]:
        """Fence the previous use of this stage's bank before receiving again."""
        parent = self._communication_stream(send=False)
        step = self._require_step(layer_idx, stage_idx)
        self._require_wire_step(step)
        if self.bank.role != "ffn" or step in self._received:
            raise RuntimeError("FFN receive requires a fresh ordered step")
        index = self.bank.layer_indices.index(layer_idx)
        if index:
            previous = (self.bank.layer_indices[index - 1], stage_idx)
            if previous not in self._sent:
                raise RuntimeError("FFN stage reuse requires its preceding send")
            if (
                not self.policy.ffn_cross_layer
                and previous[0] not in self._joined_layers
            ):
                raise RuntimeError(
                    "FFN receive requires its selected preceding layer join"
                )
            if self.policy.ffn_recv_stream:
                self.step_events[previous].send_done.wait(parent)
        try:
            yield
            self._require_parent()
            self.step_events[step].receive_done.record(parent)
            self._received.add(step)
        except BaseException:
            self._failed = True
            raise

    @contextmanager
    def ffn_send(
        self, layer_idx: int, stage_idx: int, *, tensors: tuple[torch.Tensor, ...] = ()
    ) -> Iterator[None]:
        """Join native MoE before its output copy/mask/send on the parent."""
        parent = self._communication_stream(send=True)
        step = self._require_step(layer_idx, stage_idx)
        self._require_wire_step(step)
        if self.bank.role != "ffn" or step not in self._computed:
            raise RuntimeError("FFN send requires its completed compute frontier")
        events = self.step_events[step]
        events.compute_done.wait(parent)
        self._retain_on(tensors, parent, stage_idx)
        try:
            yield
            self._require_parent()
            events.send_done.record(parent)
            self._sent.add(step)
            self._wire_index += 1
        except BaseException:
            self._failed = True
            raise

    @contextmanager
    def final_compute(
        self, stage_idx: int, *, inputs: tuple[torch.Tensor, ...] = ()
    ) -> Iterator[None]:
        """Complete Attention's final native continuation on the side stream."""
        parent = self._require_parent()
        if (
            self.bank.role != "attention"
            or stage_idx not in self.tail_events
            or stage_idx in self._finalized
        ):
            raise RuntimeError("final compute requires one positive Attention stage")
        step = (self.bank.layer_indices[-1], stage_idx)
        if self._wire_index != len(self._steps) or step not in self._received:
            raise RuntimeError("final continuation requires its matching final receive")
        tail = self.tail_events[stage_idx]
        if not self.hybrid:
            tail.ready.record(parent)
        try:
            with use_npu_current_stream(self.compute_stream):
                if self.hybrid:
                    self.step_events[step].receive_done.wait(self.compute_stream)
                else:
                    tail.ready.wait(self.compute_stream)
                self._retain_on(inputs, self.compute_stream, stage_idx)
                yield
                if torch.npu.current_stream(self.bank.device) != self.compute_stream:
                    raise RuntimeError(
                        "pipeline compute must stay on its compute stream"
                    )
                tail.done.record(self.compute_stream)
            self._finalized.add(stage_idx)
        except BaseException:
            self._failed = True
            raise

    def join_layer(self, layer_idx: int) -> None:
        """Apply the donor's non-cross-layer FFN send frontier on the parent."""
        parent = self._require_parent()
        if (
            self.bank.role != "ffn"
            or self.policy.ffn_cross_layer
            or layer_idx not in self.bank.layer_indices
            or layer_idx in self._joined_layers
            or any((layer_idx, stage) not in self._sent for stage in self._stages)
            or self._wire_index
            != (self.bank.layer_indices.index(layer_idx) + 1) * len(self._stages)
            or self._joined_layers
            != set(self.bank.layer_indices[: self.bank.layer_indices.index(layer_idx)])
        ):
            raise RuntimeError("FFN layer join requires its complete selected frontier")
        for stage in self._stages:
            self.step_events[(layer_idx, stage)].send_done.wait(parent)
        self._joined_layers.add(layer_idx)

    def join(self, *, outputs: tuple[torch.Tensor, ...] = ()) -> None:
        """Make every final side task an ancestor of parent output consumption."""
        parent = self._require_parent()
        if self._wire_index != len(self._steps) or self._compute_index != len(
            self._steps
        ):
            raise RuntimeError("pipeline join requires every static layer/stage")
        if self.bank.role == "attention":
            if self._finalized != set(self._stages):
                raise RuntimeError("pipeline join requires every final continuation")
            events = tuple(self.tail_events[stage].done for stage in self._stages)
        else:
            if not self.policy.ffn_cross_layer and self._joined_layers != set(
                self.bank.layer_indices
            ):
                raise RuntimeError("FFN final join requires every selected layer join")
            events = tuple(
                self.step_events[(self.bank.layer_indices[-1], stage)].send_done
                for stage in self._stages
            )
        for event in events:
            event.wait(parent)
        self._retain_on(outputs, parent)
        self._joined = True
