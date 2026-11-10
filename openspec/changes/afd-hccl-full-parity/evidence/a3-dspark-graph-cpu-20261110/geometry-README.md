# DSpark target Graph consumer CPU evidence

Two independent immutable snapshots used the existing A3 environment. No
package was installed and no NPU probe was started. Both runs matched all
21 installed upstream source pins and the complete uploaded plugin/unit
source manifest. Raw artifacts remain unchanged.

| Attempt | Scope | Passed | Failed | Errors | Skipped | Exit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `geometry-001` | Nine complete consumer mirrors | 551 | 4 | 0 | 0 | 1 |
| `geometry-002` | Complete coordinator mirror after fixture repair | 100 | 0 | 0 | 0 | 0 |

The first attempt passed graph key 65, stage bank 45, geometry 71, graph
control 43, F graph service 76, F graph body 89, F model runner 19 and F
worker 47. Coordinator passed 96 of 100, including all four independently
added target/draft capture isolation cases.

The four failures were the new q2/q3 × U1/U2 runtime replay cases. The CPU
helper accidentally supplied the target query width as native `dp_size`,
the fifth positional parameter to `dispatch_cg_and_sync_dp`. The production
DP1 guard correctly rejected it. Only the shared fake changed: explicit
`dp_size` and `dp_rank` keywords now read the real parallel config fields.
This was not a missing-rank or production-admission defect.

The complete coordinator module then passed all 100 cases. The final fake
SHA256 is `e0c5d683638567895e926f730926b89b05395d660a5e2c9e5e30f623841ce048`;
its mirror stayed
`7e26e6dc06fe3226780dd06c63251fafb6728ff4e7cbe91a55677fc0b4a260fc`.
All eight production consumer files remained identical across the two
snapshots and to the final source at this evidence freeze. Do not describe
the two attempts as one 555-pass run.

Both snapshot HEADs were `e18be5ca611bb9597b05f34b51d4e0883e4f1e0b`.
The immutable source manifest SHAs, source archive SHAs, exact commands,
per-module JUnit counts, versions, selected source hashes and every raw
artifact hash are recorded in
[geometry-validation-summary.json](geometry-validation-summary.json).

Raw results:

- [First nine-module JUnit](geometry-001/results/cpu.xml),
  [summary](geometry-001/results/summary.json),
  [source identity](geometry-001/identity.json).
- [Coordinator closure JUnit](geometry-002/results/cpu.xml),
  [summary](geometry-002/results/summary.json),
  [source identity](geometry-002/identity.json).

These tests exercise fixed target-q key/codec bounds, real native CPU
request/token slicing, native manager dispatch selection, graph control
and cache ownership, isolated F target shape, warmup/capture/replay host
protocol boundaries and unchanged F runner/worker construction contracts.
The capture isolation tests execute the real outer GPU runner capture
method and real DFlash NONE-manager path with explicit CPU memory,
workspace and graph boundaries. The outer capture scope still exists during
the draft path: exact-manager dispatch preserves the foreign draft manager,
and the retired target phase/provider adds no target context. Both success
and draft-error cases verify this isolation.

No CPU boundary is evidence for NPU graph capture, HCCL tensors, native
HC/Engram/MoE, checkpoint inference, DSpark proposed/accepted tokens,
sampling equivalence, timeline or performance. Production admission stays
closed. Subsequent q3 NPU component evidence is a separate increment.
