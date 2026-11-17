---
name: afd-dsv41-v030-workflow
description: Plan or implement DSV4.1-Flash AFD support in the v0.30 checkout, preserving the pinned RFC sources and HCCL-first scope. Use to resume this adaptation; this skill does not authorize implementation during a plan-only request.
---
# DSV4.1 v0.30 workflow

Read root AGENTS.md, [session](../../dsv41/session.json), [source lock](../../dsv41/sources.lock.json), and [roadmap](../../../openspec/ROADMAP_ZH.md).
Open only the active change's proposal/design/spec/tasks for detailed work. tasks.md is its sole progress checklist.

The first milestone is V2, text, eager, U1, P2pHcclAFDConnector, FFN-side gate, DSpark off.
Keep v0.30 V2 ownership and interfaces. Migrate the locked donor's HCCL execution capabilities through V2 equivalents; old V1 classes/APIs need not be copied.
Final scope includes U2/DBO, eager streams, FULL_DECODE_ONLY Graph/replay/fallback, Graph multi-stream/cross-layer, peer topologies, DSpark/PD and diagnostics.
Use the [capability matrix](../../../openspec/HCCL_CAPABILITY_MIGRATION_ZH.md) to avoid treating eager/U1 as final delivery.
Scheduling update 2026-11-12: follow the [priority order](../../../openspec/ROADMAP_ZH.md#engram-priority).
Finish other DSV4.1 non-Engram implementation, review and independent validation first; retain DSpark priority.
Engram adaptation is medium priority and starts only after those functions are ready, using C7 tasks section 2.
Preserve existing native Engram and A/F isolation. Do not add Engram-specific fixes, admission bypasses or device probes during the high-priority phase.
Mixed tasks may advance their non-Engram parts, but full-model acceptance stays pending until all dependencies pass; do not require that acceptance before starting Engram work.
Ordinary old-V4 MTP follow-up remains paused and is not an Engram start prerequisite.
U2 requires stage-local V4.1 continuation/pre_mix, Engram producer/events and shared attention state; a MoE-only substitution is not sufficient for every asynchronous mode.
V4.1 uses delayed pre_mix, native Engram history and shared cache state. Preserve the native forward, split at MoE construction, and prove router-input precision.

For a plan-only request, edit planning/workflow artifacts only. When the user requests implementation, continue that authorized scope without an additional approval loop.
Before a runtime change, resolve the active task's actual contracts and architectural blockers; do not interpret document generation as a passed architecture review.
Use the locked git objects through git show. Do not switch, edit or install into either upstream checkout.
The branch name rfc/vllm_cann does not mean automatically adopting its latest commit.

Use a bounded loop: choose the next observable behavior; add a meaningful failing boundary test when needed; implement the minimal change; review its upstream contract; run relevant checks; record evidence and mark the task only when complete.
Follow root test conventions, including validation_paths ownership and CPU/device separation.
Do not require tests for simple document edits or create tests that mirror helper calls.

Run python -B .agents/dsv41/harness.py status to locate the active task list and guard to check upstream preservation.
The first unchecked task is a navigation hint; read its priority/dependencies before executing it. Do not select deferred Engram or paused MTP work merely because it is unchecked.
check validates planning scope while session.phase is planning. Switch the session phase only after the user asks for implementation.
Keep implementation, hardware acceptance and remote delivery distinct; the user request determines which are authorized.
Do not silently commit, push or install a tool suite. Graph/PD are explicitly in scope; vision and unrelated connector features need their own task scope.
Keep ordinary MTP behavior for applicable models distinct from V4.1 native DSpark; document each donor branch's mapping rather than dropping it.

Use [patch review](../afd-dsv41-patch-review/SKILL.md) for copied/monkey-patched upstream functions and [evidence](../afd-dsv41-evidence/SKILL.md) for support claims.
