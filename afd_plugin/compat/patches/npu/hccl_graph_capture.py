# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Reserve each native model factory phase before dummy input preparation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import get_ident
from types import FunctionType

from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CreateForwardFn,
    CudaGraphManager,
)


class _HcclGraphCaptureScope:
    def __init__(
        self,
        manager: CudaGraphManager,
        before_factory: Callable[[BatchExecutionDescriptor, bool], None],
        after_capture: Callable[[], None],
        fail: Callable[[], None],
    ) -> None:
        self.manager = manager
        self.before_factory, self.after_capture, self.fail = (
            before_factory,
            after_capture,
            fail,
        )
        self.original_capture = CudaGraphManager.capture
        self.thread_id = get_ident()
        self.started = False

    def run(
        self,
        manager: CudaGraphManager,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str,
    ) -> None:
        if manager is not self.manager:
            return self.original_capture(manager, create_forward_fn, progress_bar_desc)
        try:
            if self.started or get_ident() != self.thread_id:
                raise RuntimeError(
                    "native graph capture scope permits one owning-thread capture"
                )
            self.started = True

            def reserved_factory(desc: BatchExecutionDescriptor, warmup: bool):
                if get_ident() != self.thread_id:
                    raise RuntimeError(
                        "native graph factory belongs to its capture thread"
                    )
                self.before_factory(desc, warmup)
                return create_forward_fn(desc, warmup)

            self.original_capture(manager, reserved_factory, progress_bar_desc)
            self.after_capture()
        except BaseException:
            self.fail()
            raise


@contextmanager
def use_hccl_graph_capture(
    manager: CudaGraphManager,
    *,
    before_factory: Callable[[BatchExecutionDescriptor, bool], None],
    after_capture: Callable[[], None],
    fail: Callable[[], None],
) -> Iterator[None]:
    """Scope one native capture; retain its factory and output-store ownership.

    ``before_factory`` retires the preceding phase, then reserves and arms the
    new phase before native prepare_dummy_inputs. ``after_capture`` retires the
    final phase after native capture has stored its graph. Neither callback is
    inserted into a captured tensor body. Concurrent capture is unsupported.
    """
    original = CudaGraphManager.capture
    if isinstance(original, FunctionType) and any(
        isinstance(cell.cell_contents, _HcclGraphCaptureScope)
        for cell in original.__closure__ or ()
    ):
        raise RuntimeError("native graph capture scopes cannot be nested")
    scope = _HcclGraphCaptureScope(manager, before_factory, after_capture, fail)

    # Upstream: gpu/cudagraph_utils.py:CudaGraphManager.capture @ ced6857a:398.
    # Reason: ModelCudaGraphManager hides its factory in a local closure and
    # calls prepare_dummy_inputs before any UBatchRunner hook. Functionality:
    # reserve before that native factory and flush after final graph storage.
    # Signature matches upstream exactly; no added parameters. Delegation
    # exception: its native graph pool/capture/output lifecycle is large and
    # remains authoritative; copying it would duplicate that ownership. Other
    # manager instances retain the selected native implementation unchanged.
    # Remove when upstream exposes before-factory and after-capture callbacks.
    def capture(
        self,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        # ### PATCH START: reserve outside the native model factory
        scope.run(self, create_forward_fn, progress_bar_desc)
        # ### PATCH END: reserve outside the native model factory

    try:
        CudaGraphManager.capture = capture
        yield
    finally:
        CudaGraphManager.capture = original
