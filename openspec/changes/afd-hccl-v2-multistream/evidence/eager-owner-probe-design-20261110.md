# Actual eager ticket owner device probe design

`tests/npu/hccl_eager_pipeline_probe.py` is a new component probe, independent of
the earlier endpoint primitive gate. At this writing it has only local Ruff,
Python 3.11 grammar, and CPU-only `--help` checks. No device session was launched.
The parent must review the artifact and allocate idle devices before execution.

The default is paired TP1 on two devices, FFN then Attention. TP2 uses four
explicit devices and a separate native TP world for each role. The existing
staged-probe config builder initializes the pinned runtime without weights;
the endpoint probe contributes only its already reviewed PID identity/kill
supervision functions. None of its raw isend/irecv data operations are reused.

Each rank constructs a new actual `HcclStagedTransport`, initializes it, and
explicitly runs matched warmup on both stage groups before plans. All Attention
ranks submit their real per-stage local counts with `local_counts_version=1`;
FFN receives the accepted plans. No private cursor, state override, backend
mock, fake Work, or direct payload operation is used. Each parent opens the
actual eager owner with four distinct parent/send/receive/compute streams.

Six consecutive actual target U2 parents use padded/valid stage tuples:

| Parent | Padded rows | Valid rows |
| --- | --- | --- |
| 0 | `(3, 5)` | `(2, 5)` |
| 1 | `(1, 4)` | `(1, 2)` |
| 2 | `(0, 3)` | `(0, 1)` |
| 3 | `(2, 0)` | `(0, 0)` |
| 4 | `(0, 0)` | `(0, 0)` |
| 5 | `(5, 2)` | `(4, 0)` |

Every parent executes two target layers. Attention posts both positive stages
before its first output loan, records public occupied-stage coordinates, and
then consumes loans in stage order. FFN preserves receive → native consumer
loan → output post per stage. The next layer uses a real clone of the preceding
borrowed output as BF16 input and separately forms original FP32 with a retained
small delta. This exercises the same stage slot again after its consumer
frontier. The CPU recurrence independently computes expected BF16, original
FP32, raw int32 IDs, and the full returned output at every step. FFN validates
all three received payloads, including zero padded IDs. Padding-only outputs
are compared in full and are never forcibly zeroed. CPU readbacks occur in the
actual compute stream so the probe does not depend on cross-stream host-copy
accidents. These readbacks are deliberately correctness checks, not a timing
benchmark or overlap demonstration.

After every parent, public `finish()` performs the existing COMPLETE handshake.
The probe checks finished state, no retained active owner, and a cleared plan.
After the final parent every Attention rank calls normal acknowledged STOP and
FFN receives it. Plugin groups close before the final native TP all-reduce.
The native collective before and after plugin activity must return TP size on
every rank. No native MoE or model is constructed, so those checks prove native
group survival rather than FFN/model collective equivalence.

The supervisor starts a separate session per rank, observes owned PIDs together
with Linux start ticks, enforces a finite deadline, and kills/reaps only its own
still-live processes on failure. Workers retain a failed owner rather than
destroying groups or claiming abort. Rank logs and JSON report full step checks,
parent outcomes, native TP checks, error tracebacks, exit codes, and supervision
actions. The outer validation harness should preserve source/package manifests
and before/after NPU inventory exactly as in the previous immutable device runs.

Review first, then TP1. TP2 follows only after the first run has drained normally
and idle devices have been rechecked. Passing this probe would accept only the
actual ticket/loan component and bounded recurrence covered above. It would not
accept native V4.1 Attention/HC/Engram, real FFN/MoE, multi-stream performance,
failure recovery, or a production factory/guard change.
