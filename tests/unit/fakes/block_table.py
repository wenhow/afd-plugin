# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU kernel leaves for actual native block-table gather and Ascend slots.

Persistent native BlockTables, full absolute IDs including null coordinates,
gather/slot wrappers and output views remain intact. Pointer-valued tables are
resolved only within the installer's known CPU tensors, never dereferenced.
"""

from collections.abc import Callable
from functools import partial

import pytest
import torch
from vllm.v1.worker.gpu import block_table as gpu_block_table
from vllm_ascend.worker.v2 import block_table as ascend_block_table
from vllm_ascend.worker.v2.block_table import AscendBlockTables


class CPUGatherBlockTablesKernel:
    def __init__(self, block_tables: AscendBlockTables):
        tensors = [
            *(table.gpu for table in block_tables.block_tables),
            *block_tables.input_block_tables,
        ]
        self.tables = {tensor.data_ptr(): tensor for tensor in tensors}

    def __getitem__(self, grid: tuple[int, int]) -> Callable[..., None]:
        return partial(self.gather, num_groups=grid[0], num_rows=grid[1])

    def gather(
        self,
        idx_mapping: torch.Tensor,
        source_ptrs: torch.Tensor,
        destination_ptrs: torch.Tensor,
        table_strides: torch.Tensor,
        num_blocks: torch.Tensor,
        num_blocks_stride: int,
        num_reqs: int,
        *,
        BLOCK_SIZE: int,  # noqa: N803 - native launch keyword
        num_groups: int,
        num_rows: int,
    ) -> None:
        if num_blocks_stride != num_blocks.stride(0):
            raise ValueError("CPU gather fixture requires the native count stride")
        for group in range(num_groups):
            source = self.tables[int(source_ptrs[group])]
            destination = self.tables[int(destination_ptrs[group])]
            if source.stride(0) != int(table_strides[group]):
                raise ValueError("CPU gather fixture requires native table strides")
            for row in range(num_rows):
                if row >= num_reqs:
                    destination[row].zero_()
                    continue
                request_index = int(idx_mapping[row])
                count = int(num_blocks[group, request_index])
                destination[row, :count].copy_(source[request_index, :count])


class CPUSlotMappingsKernel:
    def __init__(self, block_tables: AscendBlockTables):
        self.tables = {
            table.gpu.data_ptr(): table.gpu for table in block_tables.block_tables
        }

    def __getitem__(self, grid: tuple[int, int]) -> Callable[..., None]:
        return partial(self.compute, num_groups=grid[0])

    def compute(
        self,
        max_num_tokens: int,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        table_ptrs: torch.Tensor,
        table_strides: torch.Tensor,
        block_sizes: torch.Tensor,
        kernel_block_sizes: torch.Tensor,
        slot_mappings: torch.Tensor,
        slot_mapping_stride: int,
        cp_rank: int,
        *,
        CP_SIZE: int,  # noqa: N803 - native launch keyword
        CP_INTERLEAVE: int,  # noqa: N803 - native launch keyword
        PAD_ID: int,  # noqa: N803 - native launch keyword
        TRITON_BLOCK_SIZE: int,  # noqa: N803 - native launch keyword
        BLOCK_TABLE_WINDOW_SIZE: int,  # noqa: N803 - native launch keyword
        slot_mapping_enabled: torch.Tensor | None = None,
        HAS_SLOT_MAPPING_ENABLED: bool = False,  # noqa: N803 - native launch keyword
        num_groups: int,
    ) -> None:
        if CP_SIZE != 1 or cp_rank != 0 or CP_INTERLEAVE != 1:
            raise ValueError("CPU slot fixture requires the actual CP1 role")
        if slot_mapping_stride != slot_mappings.stride(0):
            raise ValueError("CPU slot fixture requires the native output stride")
        actual_tokens = int(query_start_loc[len(idx_mapping)])
        for group in range(num_groups):
            slot_mappings[group, actual_tokens:max_num_tokens].fill_(PAD_ID)
            if HAS_SLOT_MAPPING_ENABLED and not bool(slot_mapping_enabled[group]):
                slot_mappings[group, :actual_tokens].fill_(PAD_ID)
                continue
            table = self.tables[int(table_ptrs[group])]
            if table.stride(0) != int(table_strides[group]):
                raise ValueError("CPU slot fixture requires native table strides")
            block_size = int(kernel_block_sizes[group])
            for row, request_index in enumerate(idx_mapping.tolist()):
                start, stop = query_start_loc[row : row + 2].tolist()
                for token_index in range(start, stop):
                    position = int(positions[token_index])
                    block_index, offset = divmod(position, block_size)
                    block_id = int(table[request_index, block_index])
                    slot_mappings[group, token_index] = block_id * block_size + offset


def install_native_block_table_kernels_cpu(
    monkeypatch: pytest.MonkeyPatch, block_tables: AscendBlockTables
) -> None:
    """Keep native gather/slot methods, replacing only their device launches."""
    if block_tables.device.type != "cpu":
        raise ValueError("native block-table fixture requires CPU tables")
    monkeypatch.setattr(
        gpu_block_table,
        "_gather_block_tables_kernel",
        CPUGatherBlockTablesKernel(block_tables),
    )
    monkeypatch.setattr(
        ascend_block_table,
        "_compute_slot_mappings_kernel",
        CPUSlotMappingsKernel(block_tables),
    )
