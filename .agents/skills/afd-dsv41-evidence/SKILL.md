---
name: afd-dsv41-evidence
description: Assess or record validation for the DSV4.1 HCCL adaptation, separating planning, CPU protocol checks, NPU acceptance and performance. Use when reporting readiness, regressions or benchmark results.
---
# Record DSV4.1 validation evidence

Read the active change's tasks and [source lock](../../dsv41/sources.lock.json).
Record commands, exit codes, raw log locations, actual source/package versions, checkpoint/config identity, dtype/quantization, topology and device IDs.
Keep evidence outside the support claim until it demonstrates that exact claim.

Separate:
- planning/static checks: structure, syntax, source availability and unchanged upstream workspaces;
- CPU contracts: observable protocol/output/config/weight ownership;
- NPU components: HCCL ordering, payload/IDs/precision, actual operators and process groups;
- end-to-end NPU acceptance: native control versus AFD, same checkpoint and decoding parameters;
- performance: total throughput, TTFT, TPOT, actual cards, per-card efficiency and deployment scale.

Mock/AST checks do not establish NPU correctness. Missing dependencies that reproduce on baseline do not establish an adaptation regression.
The local harness is not official OpenSpec validation, a runtime package check, or hardware acceptance.
For DSpark record actual proposed/accepted tokens and rejection/fallback behavior; do not force acceptance or reuse old V4 numbers as V4.1 evidence.

Retain unsupported combinations as explicitly pending. Diagnose supplied errors using the matching request/log chain rather than healthcheck success.
Follow the [2026-11-19 priority clarification](../../../openspec/ROADMAP_ZH.md#engram-priority): real-checkpoint eager/U1 validation reuses native Engram now; new Engram adaptation remains medium priority.
Distinguish the explicit native Engram+DBO/microbatch rejection from unvalidated Graph/U1, DSpark/U1 or PD/U1 combinations. The restriction is not evidence that all Engram modes fail or that all U1 combinations work.
Preserve prior Engram and independent non-Engram component evidence. Record real-checkpoint U1 stages separately from new U2 adaptation and final HCCL acceptance; dummy/BF16/no-Engram results cannot close full-model or performance tasks.
State remaining blockers with their exact scope; do not imply FastAPI solver conflicts are resolved by --no-deps.
The user accepted the deployed container FastAPI on 2026-11-06 (verified 0.123.10). Continue development and runtime validation in that environment; the declared requirement mismatch remains a recorded fact, not a blocker. Keep resolver results separate from actual runtime evidence.
Do not launch services or consume hardware merely because a report is being written.

For full HCCL migration, also read [the capability matrix](../../../openspec/HCCL_CAPABILITY_MIGRATION_ZH.md).
Require actual U2 stage execution, graph capture/replay/miss transitions and A/F timelines for multi-stream claims. Preserve existing C0/C1/C2 evidence scope; full migration requires C4–C7 combination acceptance. Record all effective overlap/diagnostic values and card-normalized performance.
