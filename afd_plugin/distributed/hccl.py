# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe HCCL rank and token layouts, independent of runtime admission.

The locked donor places FFN ranks before Attention ranks and balances fanout
over the padded Attention input. These contracts describe eager U1/U2 layouts;
they allocate no groups or tensors and do not enable unsupported runtime modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from afd_plugin.config import HCCL_P2P_CONNECTOR, AFDConfig
from afd_plugin.distributed.topology import AFDRankMapping, build_rank_mapping

MAX_HCCL_STAGES = 2
MAX_HCCL_TOKEN_COUNT = (1 << 31) - 1


def build_hccl_rank_mapping(
    config: AFDConfig, role_rank: int, *, native_tp_size: int = 1
) -> AFDRankMapping:
    """Map integral HCCL groups without broadening other P2P connectors.

    Reuse the common mapping for A>=F. Fanout follows topology.py at donor
    4ddd69a3, but remains local to HCCL. Unequal role counts are prepared only
    for TP1; equal counts preserve each native TP coordinate independently.
    This validates a layout, not the full AFD runtime configuration.
    """
    if config.connector != HCCL_P2P_CONNECTOR:
        raise ValueError("HCCL layout requires P2pHcclAFDConnector")
    attention_size, ffn_size = config.num_attention_ranks, config.num_ffn_ranks
    for name, size in (
        ("attention_size", attention_size),
        ("ffn_size", ffn_size),
        ("native_tp_size", native_tp_size),
    ):
        if type(size) is not int or size <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max(attention_size, ffn_size) % min(attention_size, ffn_size):
        raise ValueError("HCCL layout requires an integral A/F ratio")
    if attention_size % native_tp_size or ffn_size % native_tp_size:
        raise ValueError("HCCL role sizes must contain complete native TP groups")
    if attention_size != ffn_size and native_tp_size != 1:
        raise ValueError("HCCL unequal layouts currently require native TP1")
    if config.role not in ("attention", "ffn"):
        raise ValueError("HCCL layout requires attention or ffn role")
    role_size = attention_size if config.role == "attention" else ffn_size
    if type(role_rank) is not int or not 0 <= role_rank < role_size:
        raise ValueError("HCCL role_rank is outside its role size")
    if attention_size >= ffn_size:
        return build_rank_mapping(config, role_rank)

    ratio = ffn_size // attention_size
    subgroup_index = role_rank if config.role == "attention" else role_rank // ratio
    ffn_ranks = tuple(range(subgroup_index * ratio, (subgroup_index + 1) * ratio))
    attention_rank = ffn_size + subgroup_index
    world_rank = ffn_size + role_rank if config.role == "attention" else role_rank
    subgroup_ranks = (*ffn_ranks, attention_rank)
    return AFDRankMapping(
        role=config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        p2p_rank=world_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
        min_size=attention_size,
        subgroup_index=subgroup_index,
        rank_in_subgroup=subgroup_ranks.index(world_rank),
        subgroup_ranks=subgroup_ranks,
        dp_metadata_destinations=ffn_ranks if config.role == "attention" else (),
    )


@dataclass(frozen=True, slots=True)
class HcclPeerSlice:
    """One peer's row range in Attention transport and FFN aggregate buffers."""

    attention_world_rank: int
    ffn_world_rank: int
    attention_offset: int
    ffn_offset: int
    valid_tokens: int
    padded_tokens: int

    @property
    def attention_output_slice(self) -> slice:
        return slice(self.attention_offset, self.attention_offset + self.valid_tokens)

    @property
    def ffn_output_slice(self) -> slice:
        return slice(self.ffn_offset, self.ffn_offset + self.valid_tokens)


@dataclass(frozen=True, slots=True)
class HcclStageLayout:
    """Derive both ends' row contracts from native valid and padded counts.

    Valid rows are a prefix on every Attention rank. Active stages retain at
    least one transport row per peer, matching the donor's empty-peer policy.
    A wholly unpadded empty stage has zero tensor work on every peer. Native
    padding is preserved before fanout; balanced chunks must not be computed
    from valid counts alone. All generated padding requires zero raw IDs.
    """

    stage_idx: int
    attention_valid_tokens: tuple[int, ...]
    attention_padded_tokens: tuple[int, ...]
    ffn_size: int
    native_tp_size: int = 1
    peer_slices: tuple[HcclPeerSlice, ...] = field(init=False)

    def __post_init__(self) -> None:
        if type(self.stage_idx) is not int or not 0 <= self.stage_idx < MAX_HCCL_STAGES:
            raise ValueError("HCCL stage_idx must be 0 or 1")
        valid, padded = self.attention_valid_tokens, self.attention_padded_tokens
        if (
            not isinstance(valid, tuple)
            or not isinstance(padded, tuple)
            or not valid
            or len(valid) != len(padded)
        ):
            raise ValueError("Attention counts must be matching nonempty tuples")
        config = AFDConfig(
            connector=HCCL_P2P_CONNECTOR,
            role="ffn",
            num_attention_ranks=len(valid),
            num_ffn_ranks=self.ffn_size,
        )
        build_hccl_rank_mapping(config, 0, native_tp_size=self.native_tp_size)
        for count, capacity in zip(valid, padded, strict=True):
            if (
                type(count) is not int
                or type(capacity) is not int
                or not 0 <= count <= capacity <= MAX_HCCL_TOKEN_COUNT
            ):
                raise ValueError(
                    "Attention counts must satisfy 0 <= valid <= padded <= INT32_MAX"
                )
        for start in range(0, len(valid), self.native_tp_size):
            stop = start + self.native_tp_size
            if len(set(valid[start:stop])) != 1 or len(set(padded[start:stop])) != 1:
                raise ValueError("native TP coordinates must agree on token counts")
        active = any(padded)
        peers = []
        ffn_offsets = [0] * self.ffn_size
        group_config = (
            config if len(valid) >= self.ffn_size else replace(config, role="attention")
        )
        for group_idx in range(min(len(valid), self.ffn_size)):
            mapping = build_hccl_rank_mapping(
                group_config,
                group_idx,
                native_tp_size=self.native_tp_size,
            )
            ffn_ranks = tuple(
                rank for rank in mapping.subgroup_ranks if rank < self.ffn_size
            )
            for world_rank in mapping.subgroup_ranks:
                if world_rank < self.ffn_size:
                    continue
                attention_rank = world_rank - self.ffn_size
                count, capacity = valid[attention_rank], padded[attention_rank]
                transport_tokens = max(capacity, len(ffn_ranks)) if active else 0
                attention_offset = 0
                for peer_index, ffn_rank in enumerate(ffn_ranks):
                    width = (transport_tokens + len(ffn_ranks) - 1 - peer_index) // len(
                        ffn_ranks
                    )
                    peers.append(
                        HcclPeerSlice(
                            attention_world_rank=world_rank,
                            ffn_world_rank=ffn_rank,
                            attention_offset=attention_offset,
                            ffn_offset=ffn_offsets[ffn_rank],
                            valid_tokens=min(max(count - attention_offset, 0), width),
                            padded_tokens=width,
                        )
                    )
                    attention_offset += width
                    ffn_offsets[ffn_rank] += width
                    if ffn_offsets[ffn_rank] > MAX_HCCL_TOKEN_COUNT:
                        raise ValueError(
                            "FFN padded count exceeds INT32 transport capacity"
                        )
        object.__setattr__(self, "peer_slices", tuple(peers))

    def for_attention(self, role_rank: int) -> tuple[HcclPeerSlice, ...]:
        if type(role_rank) is not int or not 0 <= role_rank < len(
            self.attention_valid_tokens
        ):
            raise ValueError("Attention role_rank is outside its role size")
        world_rank = self.ffn_size + role_rank
        return tuple(
            peer for peer in self.peer_slices if peer.attention_world_rank == world_rank
        )

    def for_ffn(self, role_rank: int) -> tuple[HcclPeerSlice, ...]:
        if type(role_rank) is not int or not 0 <= role_rank < self.ffn_size:
            raise ValueError("FFN role_rank is outside its role size")
        return tuple(
            peer for peer in self.peer_slices if peer.ffn_world_rank == role_rank
        )


__all__ = ["HcclPeerSlice", "HcclStageLayout", "build_hccl_rank_mapping"]
