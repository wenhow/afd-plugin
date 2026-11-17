---
name: afd-dsv41-patch-review
description: Review copied or patched upstream functions for the DSV4.1 v0.30 HCCL adaptation, including signatures, state ownership, cost and removal. Use before accepting an AFD compatibility patch; it does not modify upstream sources.
---
# Review a DSV4.1 compatibility patch

Read root AGENTS.md and [source lock](../../dsv41/sources.lock.json). Inspect the actual diff and exact upstream function at the locked SHA.
A path existing in the source index proves availability, not API or runtime compatibility.

For each patched function, record upstream path/symbol/SHA and exact signature including return type.
Require the immediately preceding reason/functionality comments, documented added parameters if any, and short PATCH START/END markers around every AFD difference.
Use the target function as the skeleton. An _original_* delegation exception needs a concrete size/suitability reason in the function comments.

Assess the correct owner and smallest seam: model construction versus MoE proxy, worker group initialization versus ModelState, control plane versus tensor transport.
Check native delayed HC, Engram history, shared KV/index state, token/FP32 precision and SP/TP reduction.
Verify non-AFD and existing V4/V2 connector behavior. Prefer direct upstream fields and concrete types; avoid reflection used to hide a changed contract.

Record bandwidth, synchronization, HBM or startup cost, plus upstream contribution or removal condition.
Report findings with file/line evidence, severity, required fix and validation.
Do not mark architectural review passed with unresolved contract or ownership findings.
This review is local; publishing or remote delivery requires the user's task to include it.

For full HCCL migration, also read [the capability matrix](../../../openspec/HCCL_CAPABILITY_MIGRATION_ZH.md).
Review stage/peer ownership, V4.1 continuation, capture parent-stream rules, graph keys/fallback, FP32/IDs lifetime and PD history handoff. A Graph flag or existing CAMP2P V2 path is not an equivalent implementation by itself.
