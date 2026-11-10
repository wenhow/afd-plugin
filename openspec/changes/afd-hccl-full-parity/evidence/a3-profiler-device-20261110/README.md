# A3 role-profiler component evidence

The separately authorized fresh
[role-owner-002](role-owner-002/results/summary.json) **passed** on device 0 in
66.397 seconds. Attention-main and FFN-background ran in separate sequential
processes. Each completed seven exact CPU-oracle NPU matmul comparisons, six
profiler steps, start/stop/restart and owner-thread finalization. FFN management
stop/restart requests remained pending while the owner awaited its next Queue
command, then applied at that next execution boundary. These are explicit
handshake and status checks, without latency assertions.

All four exported initial/restart traces contain two `AI_CORE` device MatMul
events and matching kernel CSV task/name/stream multisets. Attention's owner is
its main management thread; FFN's owner is its background execution thread.
The [independent verification](role-owner-002/independent-verification.json)
reads both [Attention](role-owner-002/results/probe/attention/role.json) and
[FFN](role-owner-002/results/probe/ffn/role.json) results and actual trace files.
Both snapshot and installed-source gates matched, all eight boards were idle
before and after, and there was no timeout or process kill. The
[retrieval map](role-owner-002/retrieval.json) verifies 35 mapped key files and
all 339 regular members of the complete 156,312-byte archive under checkout
`.tmp/profiler-role-owner-002-results.tar.gz`, SHA256
`f09c7cb7ad44546f02b78b5ea4af7398177020224ff865f3e0f206fbc5e559fe`.
Its [source identity](role-owner-002/identity.json) retains actual HEAD
`5aebfc5e954d20da18bff754a2008c44bd7e31bc` and manifest
`d971881105abb8a8d5e950da362a4f3820e82058499ded3c9a282c2db81b933c`.
This new result does not rewrite 001.

The original [role-owner-001](role-owner-001/results/summary.json) remains
**unsuccessful**. Attention completed seven exact CPU-oracle comparisons of
real NPU matmul, initial start/stop, restart, six profiler steps and owner-thread
finish. Its role result is still `pending-or-unproven`, return code 2. The
supervisor stopped at that first failure, so **FFN was not executed**.

The first trace detector required `cat=kernel`. Both real exported trace files
instead identify device events through `process_name=Ascend Hardware`, the
`NPU 0` process label, `ph=X` and `Task Type=AI_CORE`; the device MatMul events
have no category field. Independent offline parsing finds two MatMul events
in each 115-event trace, with matching Name/Task ID/Stream ID rows in each
`kernel_details.csv`. The
[offline record](role-owner-001/offline-detector-recovery-002.json) applies the
revised parser to those unchanged files without importing torch or executing a
device. It does not turn the original failed run into a pass or accept FFN.

Two independent harness defects are preserved. Its outer final `npu-smi` call
omitted the retained CANN environment and returned loader error 127 for missing
`libc_sec.so`, producing an invalid zero-idle sample. A separate
[read-only check with the retained CANN environment](role-owner-001/post-retrieval-with-cann-readonly.log)
returned zero and observed all eight boards idle; no processes were killed.
That later observation is separate from the unchanged original summary.
Windows then rejected a long trace pathname during ordinary tar extraction.
The complete original results archive is retained under the checkout's `.tmp`
at `.tmp/profiler-role-owner-001-results.tar.gz`, SHA256
`fee716ee670a5108b53b15388f766a9823e46957f0ab167aea9923c000bd5a8a`
(79,564 bytes, 176 regular files). The
[recovery map](role-owner-001/retrieval-recovery.json) records all member hashes
and short names for the unchanged trace JSON, kernel CSV, profiler metadata
and core results. The first partial extraction was preserved under `.tmp`.

The original probe/harness are pinned by
[001 identity](role-owner-001/identity.json): probe `2bed9af7…`, harness
`ad105055…`, plugin profiler `4dcd6647…`, and installed torch-npu profiler
`1a049025…`. Revised probe `d85e5866…` and harness `a86df96d…` use hardware
metadata plus the CSV task multiset, retain the CANN environment for final
sampling, and recover long names through a complete archive and short-name
mapping. Their [prepare-only snapshot](archive-precheck-002/identity.json)
contains 140 unique members and matching hashes. Preparation and offline
parsing are not another device attempt.

This evidence is limited to real plugin profiler sessions and controlled NPU
matmul. It does not instantiate native A/F model runners, an FFN service loop,
HTTP or executor RPC, HCCL, model weights, HC/Engram/MoE, Graph execution or a
performance benchmark. The installed source gate is specific to the existing
A3 environment; A5 use first requires collecting and reviewing its own
installed profiler sources. No packages or upstream sources were changed.
