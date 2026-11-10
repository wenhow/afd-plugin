# Smallest eager owner justified by the endpoint gate

The actual A3 endpoint gate passed isend and irecv for BF16, independent FP32
and int32 IDs on the existing warmed `HcclStagedTransport.data_groups[0]`.
Both host calls and Work.wait returned before matching permission; consumer
events stayed pending across the held observation window. Matching then
delivered original payloads exactly despite send-storage reuse ordered after
Work.wait. See `a3-eager-backend-20261110/validation-summary.json` for identities.
This supports an opt-in **software implementation** of the proposal's Work and
event ownership. It does not accept a multi-stage runtime or model schedule.

The first production increment should stay inside the existing transport and
accepted plan: a public `open_eager_pipeline(...)` returns the sole parent owner,
exclusive with serial tensor methods and Graph execution. No caller modifies
the transport's private cursor to pretend raw work completed. Default C4 entry
points retain their behavior. Initially constrain eligibility to paired DP1,
PP1/CP1/non-SP eager U2 with matching native role TP/config coordinates.

An opaque owner-issued ticket retains the exact plan/transaction, layer/stage,
BF16/original FP32/raw IDs, any output buffer, all native Work handles, producer
events and consumer events. Permit at most one occupied ticket per stage and
preserve the existing layer-major/stage-minor posting order. Posting and reuse
frontiers are separate; no stage's buffer can be repurposed until its prior
send and native consumer are ordered complete. Existing shape, dtype, ID and
router precision validation remains in force before affected operations.

Use individual `dist.isend/irecv(group=existing_group, group_dst/group_src=...)`
and `use_npu_current_stream` for actual producer/posting/consumer streams.
`Work.wait()` with its default timeout establishes the tested consumer-stream
dependency; an event recorded after that consumer provides its later lifetime
frontier. Do not treat host return or an event on an unrelated stream as data
readiness. The eventual API should expose `post_attention`, `post_ffn_receive`,
`ffn_inputs`, `post_ffn_output`, and an Attention output loan with explicit
consumer stream, rather than returning a supposedly ready raw output tensor.

`finish()` must join real remaining work and native consumers before the
existing collective COMPLETE transition. STOP remains an idle acknowledged
boundary. Failure is terminal; no CANCEL/abort semantics or optimistic timeout
recovery follows from this experiment. Tickets and storage must remain retained
through any failed drain until the owning process exits safely.

There are two necessary next gates before claiming real eager overlap:

1. Communicator first-use: this experiment intentionally used matched warmup.
   A normal session therefore needs a paired, explicit backend warmup barrier
   on each used stage group, or a separate bounded first-use progress test.
   Do not assume mere ProcessGroup construction removes HCCL lazy rendezvous.
2. Actual owner: validate two stage tickets in flight, per-stage reuse and
   delayed Attention continuation, output transport in both directions, TP2,
   full shapes and pressure, then native F/MoE and V4.1 consumer ownership.
   Only an A/F timeline can establish overlap and its benefit.

This document changes no production source and does not mark eager tasks done.
