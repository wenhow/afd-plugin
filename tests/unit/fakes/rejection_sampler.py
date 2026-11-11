# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit greedy CPU boundaries for native speculative-decoding operators.

These implementations replace device launches, not RejectionSampler or the
runner's sampling/history control flow. Probabilistic/synthetic verification,
penalties and device execution are deliberately outside this CPU seam.
"""

from types import SimpleNamespace

import torch
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS


class GreedySamplingState:
    """No sampling transforms; actual native RejectionSampler owns verification."""

    def __init__(self, req_states, num_reqs):
        self.req_states = req_states
        self.compute_nans = self.use_fp64_gumbel = False
        self.logprobs_mode = "raw_logprobs"
        self.penalties_state = SimpleNamespace(output_bin_counts=None)
        self.sampling_states = SimpleNamespace(
            temperature=SimpleNamespace(gpu=torch.zeros(num_reqs)),
            seeds=SimpleNamespace(gpu=torch.arange(num_reqs, dtype=torch.int64)),
            max_num_logprobs=lambda indices: NO_LOGPROBS,
        )

    def apply_staged_writes(self):
        pass

    def apply_sampling_params(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        positions,
        draft_sampled,
        expanded_local_pos,
    ):
        if logits.device.type != "cpu" or torch.any(
            self.sampling_states.temperature.gpu[idx_mapping] != 0
        ):
            raise AssertionError("CPU sampling boundary supports greedy decoding only")
        return logits


def greedy_rejection_cpu(
    target_logits,
    draft_logits,
    draft_sampled,
    cu_num_logits,
    positions,
    idx_mapping,
    expanded_idx_mapping,
    expanded_local_pos,
    temperature,
    seeds,
    num_speculative_steps,
    synthetic_conditional_rates=None,
    *,
    use_fp64=False,
    use_block_verification=False,
):
    """Logits-driven greedy prefix verification at the Triton launch boundary.

    Each target row predicts the next token: compare row i's argmax with draft
    row i+1 until the first mismatch, then emit that argmax as recovery/bonus.
    No acceptance lengths are injected by a test or synthetic-acceptance mode.
    """
    if (
        target_logits.device.type != "cpu"
        or draft_logits is not None
        or synthetic_conditional_rates is not None
        or use_block_verification
        or torch.any(temperature != 0)
    ):
        raise AssertionError("CPU verification is greedy, non-synthetic and one-hot")
    num_reqs = idx_mapping.numel()
    sampled = torch.full((num_reqs, num_speculative_steps + 1), -1, dtype=torch.int64)
    counts = torch.empty(num_reqs, dtype=torch.int32)
    predictions = target_logits.argmax(dim=-1)
    for row in range(num_reqs):
        start, stop = (int(value) for value in cu_num_logits[row : row + 2])
        width = stop - start
        if not 1 <= width <= num_speculative_steps + 1:
            raise AssertionError(
                "verification query exceeds the configured draft width"
            )
        accepted = 0
        while accepted < width - 1:
            if predictions[start + accepted] != draft_sampled[start + accepted + 1]:
                break
            sampled[row, accepted] = draft_sampled[start + accepted + 1]
            accepted += 1
        sampled[row, accepted] = predictions[start + accepted]
        counts[row] = accepted + 1
    return sampled, counts


def greedy_sample_cpu(
    logits,
    expanded_idx_mapping,
    temperature,
    seed,
    pos,
    apply_temperature,
    is_drafting,
    logits_cache=None,
    logits_cache_col=None,
    use_fp64=False,
):
    """Only the target Gumbel operator leaf for actual greedy Sampler calls."""
    if (
        logits.device.type != "cpu"
        or torch.any(temperature[expanded_idx_mapping] != 0)
        or is_drafting
        or logits_cache is not None
        or logits_cache_col is not None
        or use_fp64
    ):
        raise AssertionError("CPU target sampling is greedy and has no logits cache")
    return logits.argmax(dim=-1)


def sampled_and_rejected_cpu(
    num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
):
    """The native counter kernel's decode and chunked-prefill formulas."""
    chunked_prefill = seq_lens < prefill_len[idx_mapping]
    num_sampled.masked_fill_(chunked_prefill, 0)
    rejected = torch.diff(cu_num_logits).to(num_sampled.dtype) - num_sampled
    rejected.masked_fill_(chunked_prefill, 0)
    return num_sampled, rejected


def update_speculative_history_cpu(
    idx_mapping,
    num_computed_tokens,
    last_sampled_tokens,
    output_bin_counts,
    sampled_tokens,
    num_sampled,
    num_rejected,
    query_start_loc,
    all_token_ids,
    total_len,
):
    """Native post_update's token/history formulas over explicit CPU storage."""
    if sampled_tokens.device.type != "cpu":
        raise AssertionError("history boundary requires CPU token storage")
    for row, request_index in enumerate(idx_mapping.tolist()):
        if request_index < 0:
            continue
        count = int(num_sampled[row])
        position = int(total_len[request_index])
        if count:
            tokens = sampled_tokens[row, :count]
            all_token_ids[request_index, position : position + count].copy_(tokens)
            last_sampled_tokens[request_index] = tokens[-1]
            total_len[request_index] += count
            if output_bin_counts is not None:
                for token in tokens.tolist():
                    output_bin_counts[request_index, token] += 1
        query_len = (
            0
            if query_start_loc is None
            else int(query_start_loc[row + 1] - query_start_loc[row])
        )
        num_computed_tokens[request_index] += query_len - num_rejected[row]


class CPULookbackGather:
    """Only Engram's gather launch; native ModelState chooses history and mode."""

    def __getitem__(self, grid):
        def gather(
            window,
            idx_mapping,
            num_computed_tokens,
            all_token_ids,
            history_stride,
            num_reqs,
            *,
            DEPTH,  # noqa: N803 - exact native Triton launch keyword
            BLOCK_DEPTH,  # noqa: N803 - exact native Triton launch keyword
        ):
            if window.device.type != "cpu" or window.shape[1] != DEPTH:
                raise AssertionError("lookback boundary requires explicit CPU storage")
            window.fill_(-1)
            for row in range(num_reqs):
                request_index = int(idx_mapping[row])
                computed = int(num_computed_tokens[request_index])
                for offset in range(DEPTH):
                    position = computed - 1 - offset
                    if position >= 0:
                        window[row, offset] = all_token_ids[request_index, position]

        return gather
