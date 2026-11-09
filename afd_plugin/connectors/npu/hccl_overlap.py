# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Immutable donor-compatible overlap settings, without runtime selection.

Callers supply an environment snapshot once. Parsing a setting does not enable
its schedule or prove that the selected backend implements that capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

EAGER_U2_STREAM_OVERLAP_ENV = "AFD_HCCL_EAGER_U2_STREAM_OVERLAP"
GRAPH_U2_COMPUTE_OVERLAP_ENV = "AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"
GRAPH_U2_HYBRID_DAG_ENV = "AFD_HCCL_GRAPH_U2_HYBRID_DAG"
GRAPH_U2_ATTENTION_THREE_STREAM_ENV = "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM"
GRAPH_U2_FFN_RECV_STREAM_ENV = "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM"
GRAPH_U2_FFN_CROSS_LAYER_ENV = "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER"
STAGE_DIAGNOSTICS_ENV = "AFD_HCCL_STAGE_DIAGNOSTICS"
FFN_COMPUTE_SYNC_DIAGNOSTICS_ENV = "AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS"


# Source: donor 4ddd69a391dfb5b3e362c7601399e538a31de702,
# afd_plugin/connectors/npu/p2p_hccl.py:249-253.
# Reason/functionality: preserve the donor's binary syntax with an explicit
# environment dependency rather than reading process-global state.
# Added parameter: environment supplies that snapshot; this is composition,
# not an installed replacement of the donor helper.
def _strict_binary_enabled(
    environment: Mapping[str, str], name: str, *, default: str = "1"
) -> bool:
    # ### PATCH START: read the caller-owned environment snapshot
    raw = environment.get(name, default).strip()
    # ### PATCH END: read the caller-owned environment snapshot
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{name} must be 0 or 1, got {raw!r}")
    return raw == "1"


@dataclass(frozen=True, slots=True)
class HcclOverlapSettings:
    """Parsed connector values; consumers still own mode/capability gating."""

    eager_u2_stream_overlap: bool = True
    graph_u2_compute_overlap: bool = True
    graph_u2_hybrid_dag: bool = True
    graph_u2_attention_three_stream: bool = True
    graph_u2_ffn_recv_stream: bool = True
    graph_u2_ffn_cross_layer: bool = True
    stage_diagnostics: bool = False
    ffn_compute_sync_diagnostics: bool = False

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in self.effective_settings().values()):
            raise ValueError("HCCL overlap settings must be bool")
        if self.graph_u2_ffn_cross_layer and not self.graph_u2_ffn_recv_stream:
            raise RuntimeError(
                f"{GRAPH_U2_FFN_CROSS_LAYER_ENV}=1 requires "
                f"{GRAPH_U2_FFN_RECV_STREAM_ENV}=1"
            )

    # Source: donor p2p_hccl.py:256-295,370-398 at the revision above.
    # Reason/functionality: snapshot the same defaults and U1 eager short-circuit
    # before callers create a session. No stream policy or process env changes.
    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str], *, num_stages: int
    ) -> HcclOverlapSettings:
        if type(num_stages) is not int or num_stages < 1:
            raise ValueError("HCCL num_stages must be a positive integer")
        # ### PATCH START: immutable settings from explicit dependencies
        return cls(
            eager_u2_stream_overlap=(
                num_stages > 1
                and _strict_binary_enabled(environment, EAGER_U2_STREAM_OVERLAP_ENV)
            ),
            graph_u2_compute_overlap=_strict_binary_enabled(
                environment, GRAPH_U2_COMPUTE_OVERLAP_ENV
            ),
            graph_u2_hybrid_dag=_strict_binary_enabled(
                environment, GRAPH_U2_HYBRID_DAG_ENV
            ),
            graph_u2_attention_three_stream=_strict_binary_enabled(
                environment, GRAPH_U2_ATTENTION_THREE_STREAM_ENV
            ),
            graph_u2_ffn_recv_stream=_strict_binary_enabled(
                environment, GRAPH_U2_FFN_RECV_STREAM_ENV
            ),
            graph_u2_ffn_cross_layer=_strict_binary_enabled(
                environment, GRAPH_U2_FFN_CROSS_LAYER_ENV
            ),
            stage_diagnostics=_strict_binary_enabled(
                environment, STAGE_DIAGNOSTICS_ENV, default="0"
            ),
            ffn_compute_sync_diagnostics=_strict_binary_enabled(
                environment, FFN_COMPUTE_SYNC_DIAGNOSTICS_ENV, default="0"
            ),
        )
        # ### PATCH END: immutable settings from explicit dependencies

    def effective_settings(self) -> dict[str, bool]:
        """Return a fresh rank-diagnostic record, without claiming active modes.

        U1's eager value is already false when constructed from an environment.
        Graph values retain their parsed values even in an eager session. This
        record does not replace each consumer's Graph/eager/role eligibility.
        """
        return {
            EAGER_U2_STREAM_OVERLAP_ENV: self.eager_u2_stream_overlap,
            GRAPH_U2_COMPUTE_OVERLAP_ENV: self.graph_u2_compute_overlap,
            GRAPH_U2_HYBRID_DAG_ENV: self.graph_u2_hybrid_dag,
            GRAPH_U2_ATTENTION_THREE_STREAM_ENV: self.graph_u2_attention_three_stream,
            GRAPH_U2_FFN_RECV_STREAM_ENV: self.graph_u2_ffn_recv_stream,
            GRAPH_U2_FFN_CROSS_LAYER_ENV: self.graph_u2_ffn_cross_layer,
            STAGE_DIAGNOSTICS_ENV: self.stage_diagnostics,
            FFN_COMPUTE_SYNC_DIAGNOSTICS_ENV: self.ffn_compute_sync_diagnostics,
        }
