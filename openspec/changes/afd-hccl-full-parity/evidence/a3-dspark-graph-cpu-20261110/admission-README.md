# DSpark Graph admission CPU attempts

`admission-001` stopped at installed-source preflight before any test started.
Twenty-one vLLM/Ascend source pins matched; the extra donor AFD connector had
been incorrectly classified as an installed dependency. Its identity and
source-check failure remain intact. The source identity now records the
donor separately and checks only real vLLM/Ascend upstream dependencies.

`admission-002` passed **323 cases**, zero failures/errors/skips, using the
existing A3 venv: runtime config 106, HCCL config 139 and UBatch runner 78.
All 21 installed source pins matched. The exact source snapshot, selected
command, JUnit, logs, versions and environment are retained in that directory.
The [index](admission-summary.json) records every raw hash independently.

This validates fixed DSpark/eager-draft configuration, A/F target-shape
capabilities and explicit U2 constructor seams. CPU fakes replace device
allocation/model/kernel boundaries; no NPU probe or model started, no package
was installed and no upstream source changed. Local Windows separately
passed 126 with 120 dependency skips. Neither result qualifies full-model
serving, NPU capture/replay, real proposals/acceptance or performance.
