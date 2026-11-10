# Independent Graph fallback diagnostic review

The probe/test owner reviewed the three parent-owned production deltas before
the immutable [021 actual-dependency CPU attempt](a3-pipeline-cpu-20261110/graph-fallback-diagnostics-021/identity.json).
No blocking source finding was identified in the diagnostic fallback scope.
The [machine-readable review](stage-diagnostics-graph-fallback-review-20261110.json)
records exact snapshot hashes and the raw outcome; this is a source review and
CPU seam result, not NPU execution or full-model acceptance.

`HcclV41GraphEager` constructs an optional rank binding from explicit settings
and sink. Accepted-parent submission precedes configuration emission. Both
normal preparation and native zero-parent early return bind the actual serial
execution without manufacturing ModelState history or calling the producer
again. Target U2 can retain its logical side-compute selection while actual
compute/send/receive remain on the parent. Profile/dummy native preparation
keeps U1 geometry, and FFN compute synchronization remains inactive.

`HcclV41Execution.scope` checks existing UBatch settings, installs the same
binding for the scoped execution, and restores the prior public binding and
preparation method in `finally`. The native forward context and dispatcher
scopes also unwind on sink failure. U1 configuration binds before the native
producer; U2 uses the injected binding before its parent producer. The default
None path preserves the original optional branch.

`HcclV41GraphCoordinator` passes the same binding into selected eager fallback
and binds accepted empty/preflight parents. If a pre-KV bridge requested
settings, a changed or omitted coordinator setting is rejected. An earlier
bridge with no requested settings allows later explicit enablement. FULL
selection retains its existing captured path: this delta adds no Python log,
wait, event, query or synchronization during captured replay.

The complete pre-KV module passed 49 cases, including 22 new diagnostic cases.
They compare output and auxiliary tensors with diagnostics off/on, preserve
one native producer consumption, verify exact accepted transaction/rank and
actual-versus-logical stream fields, verify empty-parent configuration, reject
sink failure before payload/producer work and restore scopes on success,
configuration failure and compute-returned failure. The coordinator's 81 cases
and UBatch's 74 cases also passed.

The four-module 021 attempt remains **212 passed, 2 failed, 26 setup errors**,
zero skips. Execution's eight successful cases and 28 unsuccessful cases are
reported separately. All 26 setup errors patch the retired
`step_afd_npu_profiler` symbol at Execution mirror line 127; both failures patch
retired `create_afd_npu_profiler` at line 530. They expose a concurrent H14
fixture/API mismatch, rather than a diagnostic output failure. Its owner will
repair those fixtures and rerun the complete Execution module independently;
the failed XML, logs and summary remain unchanged.

All ten installed upstream pins match and the uploaded source snapshot matches
the local manifest. No package installation, upstream edit, checkpoint load
or NPU probe was performed. These checks do not establish native HC/Engram/MoE
kernel output, device stream overlap, profiler trace collection, throughput or
full-model correctness. Historical 018b/020 and owner TP1/TP2 diagnostic
sessions retain their original source identities.
