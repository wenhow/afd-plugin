# F eager U2 consumer: static intake and actual CPU handoff

The standalone consumer and its shared fake/mirror are owned by this change.
`ffn-local-static-001.json` retains the initial local candidate. Review found
that stale/finished owner rejection must occur before terminal error handling;
the fixed candidate is recorded in `ffn-local-static-002.json`. It scopes actual
parent NPU selection and vLLM TLS through final completion and preserves the
prior cached stream. The TP2 rank-one fake also supplies the real ACCEPT
coordinator world rank.

Both local attempts passed Ruff/format and Python 3.11 grammar, but local
pytest collected no cases because torch was absent: one dependency skip,
exit 5. This is not a passing runtime regression or device acceptance.

The fixed exact source bytes were uploaded by the root task as
[eager-ffn-013](../a3-pipeline-cpu-20261110/eager-ffn-013/identity.json), where all
**43 F consumer cases passed** with actual A3 dependencies and CPU tensors.
The complete run was 231 passed / two unrelated Graph worker setup errors;
[worker-followup-014](../a3-pipeline-cpu-20261110/worker-followup-014/identity.json)
then passed 114 cases after the shared Graph fixture repair and worker restart
checks. Both raw attempts remain unchanged and are explained in the
[CPU index](../a3-pipeline-cpu-20261110/README.md). Controlled model arithmetic
and host stream/Work observations do not validate native device MoE, actual
multi-inflight HCCL progress or performance.
