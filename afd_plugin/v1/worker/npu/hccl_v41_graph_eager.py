# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Graph-protocol eager execution before the native graph manager exists.

This explicit bridge agrees an eager reservation before native input/Engram
preparation. It preserves native execute, request gathering and retirement;
the later KV-initialized coordinator owns capture/replay. No feature admission
or graph/model/device emulation lives here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.model_runner import BatchReqState

from afd_plugin.compat.patches.npu.hccl_graph_dispatch import (
    HcclGraphDispatchArguments,
    use_hccl_graph_dispatch,
)
from afd_plugin.distributed.hccl_plan import HcclLocalForward
from afd_plugin.v1.worker.npu.hccl_v41_execution import (
    HcclV41Execution,
    validate_hccl_v41_binding,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph import eager_hccl_v41_geometry
from afd_plugin.v1.worker.npu.hccl_v41_graph_control import (
    HcclV41GraphControl,
    HcclV41GraphReservation,
)

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )
    from afd_plugin.v1.worker.npu.hccl_v41_graph_coordinator import (
        HcclV41GraphResources,
    )


class HcclV41GraphEager:
    """One pre-KV model owner and one fixed, still-empty graph cache/session."""

    def __init__(
        self,
        owner: AFDNPUAttentionModelRunnerV2,
        control: HcclV41GraphControl[HcclV41GraphResources],
        *,
        next_transaction_id: Callable[[], str],
    ) -> None:
        self.owner, self.control = owner, control
        self.transport, self.cache = control.transport, control.cache
        self.model, self.model_state = owner.model, owner.model_state
        self.vllm_config, self.parallel_config = (
            owner.vllm_config,
            owner.parallel_config,
        )
        self.ubatch_runner = owner.ubatch_runner
        self.next_transaction_id = next_transaction_id
        self._active = False
        self._require_storage()
        self._require_idle()

    def _require_storage(self) -> None:
        owner = self.owner
        if (
            owner.cudagraph_manager is not None
            or self.control.manager is not None
            or owner.hccl_v41_transport is not self.transport
            or self.control.transport is not self.transport
            or self.control.cache is not self.cache
            or owner.model is not self.model
            or owner.model_state is not self.model_state
            or self.model_state.model is not self.model
            or owner.vllm_config is not self.vllm_config
            or owner.parallel_config is not self.parallel_config
            or owner.parallel_config is not self.vllm_config.parallel_config
            or owner.ubatch_runner is not self.ubatch_runner
            or owner.pcp_manager is not None
        ):
            raise ValueError("pre-KV eager bridge requires its manager-free owners")

    def _require_idle(self) -> None:
        capabilities = self.transport.negotiated_capabilities
        if (
            self._active
            or capabilities.graph_execution_version != 1
            or capabilities.local_counts_version != 1
            or self.transport.graph_reservation is not None
            or self.cache.terminal
            or self.cache.closed
            or not self.cache.is_empty
        ):
            raise ValueError("pre-KV eager bridge requires an idle fresh graph session")
        validate_hccl_v41_binding(self.vllm_config, self.transport, self.owner.device)

    def _transaction_id(self) -> str | None:
        return self.next_transaction_id() if self.transport.is_coordinator else None

    def fail(self) -> None:
        # This preserves ownership for supervisor cleanup; no peer/device abort
        # is claimed by a local exception or terminal flag.
        self.transport.terminal = True
        self.cache.fail()

    @contextmanager
    def execution_scope(
        self, *, mode: str, valid_tokens: int, dummy_run: bool
    ) -> Iterator[None]:
        """Surround native execute; reserve before any eager preparation.

        A genuine no-work early return uses an explicit zero descriptor. It
        never fabricates BatchReqState or consumes/finishes Engram itself.
        """
        if (
            mode not in ("target", "dummy", "profile")
            or type(valid_tokens) is not int
            or valid_tokens < 0
            or type(dummy_run) is not bool
            or dummy_run
            and valid_tokens != 0
        ):
            raise ValueError("pre-KV execution requires native mode and valid counts")
        self._require_storage()
        self._require_idle()
        self._active = True
        reservation: HcclV41GraphReservation | None = None

        def select(
            batch_state: BatchReqState | None,
            arguments: HcclGraphDispatchArguments,
            native_descriptor: BatchExecutionDescriptor,
        ) -> BatchExecutionDescriptor:
            nonlocal reservation
            self._require_storage()
            if batch_state is None:
                reservation = self.control.reserve_eager(
                    eager_hccl_v41_geometry(native_descriptor),
                    mode=mode,
                    transaction_id=self._transaction_id(),
                )
            else:
                reservation = self.control.reserve_batch(
                    batch_state,
                    parallel_config=self.parallel_config,
                    allow_ubatching=arguments.allow_ubatching,
                    need_eager=True,
                    max_query_len=arguments.max_query_len,
                    num_active_loras=arguments.num_active_loras,
                    transaction_id=self._transaction_id(),
                    mode=mode,
                )
            if reservation is None:
                raise RuntimeError("pre-KV graph session stopped before preparation")
            if reservation.decision.phase != "eager":
                raise ValueError("manager-free native execution requires eager phase")
            eager = HcclV41Execution(
                self.owner,
                self.transport,
                mode=mode,
                valid_tokens=valid_tokens,
                dummy_run=dummy_run,
            )
            preparation.enter_context(eager.scope(manage_dispatch=False))
            return reservation.geometry.descriptor

        try:
            with (
                ExitStack() as preparation,
                use_hccl_graph_dispatch(self.owner, select),
            ):
                yield
                self._require_storage()
                if reservation is None:
                    if valid_tokens:
                        raise RuntimeError(
                            "native execution returned before pre-KV dispatch"
                        )
                    reservation = self.control.reserve_eager(
                        eager_hccl_v41_geometry(
                            BatchExecutionDescriptor(CUDAGraphMode.NONE, 0, 0)
                        ),
                        mode=mode,
                        transaction_id=self._transaction_id(),
                    )
                    if reservation is None:
                        raise RuntimeError(
                            "pre-KV graph session stopped before empty parent"
                        )
                if reservation.geometry.descriptor.num_tokens == 0:
                    plan = self.transport.submit_forward(
                        HcclLocalForward(mode, (0,), (0,)),
                        transaction_id=reservation.decision.transaction_id
                        if self.transport.is_coordinator
                        else None,
                    )
                    if plan is None:
                        raise RuntimeError(
                            "pre-KV graph session stopped before acceptance"
                        )
                    self.transport.finish_forward()
                if (
                    self.transport.plan is not None
                    or self.transport.graph_reservation is not None
                ):
                    raise RuntimeError(
                        "native execution did not complete its eager parent"
                    )
        except BaseException:
            self.fail()
            raise
        finally:
            self._active = False
