# DSpark target U2: actual-dependency CPU seams

The HCCL eager target selector now admits a validated native V4.1 DSpark
configuration with U2/DBO. FFN receives a private target configuration with
speculation and Engram removed before role construction. Same-checkpoint,
draft/auxiliary metadata, equal TP and native rejection constraints remain.
Other connector families retain their prior U1 restriction. Graph speculation
and adaptive U2 verification remain rejected.

| Immutable attempt | Result | Selected modules | JUnit seconds |
| --- | --- | --- | --- |
| `dspark-target-u2-001` | 214 passed, 4 failed, no errors/skips | Runtime config, HCCL selection, staged selector, actual target execution | 2.189 |
| `dspark-selector-002` | 26 passed, no failures/errors/skips | Complete affected staged-selector module | 0.187 |

All four initial failures came from new TP2 fixtures passing caller rank zero
when the native rank and role rank were one. The production binding correctly
rejected that mismatch. Only the fixture caller rank changed; the full affected
module then passed. Both attempts and their exact snapshots are retained.
Counts above describe independent pytest runs and must not be added into a
single all-green run. Authoritative timings and hashes are in the
[validation index](validation-summary.json).

The target execution cases use real installed native UBatch preparation and
the DFlash speculator's local `_run_model` context boundary. They compare
ordered auxiliary outputs for query lengths 2/3, draft lengths 1/2 and valid
5/padded 8 token coordinates, preserving auxiliary collection before Engram.
The controlled arithmetic model is a CPU collaborator. The draft call sees
neither a target sidecar nor AFD metadata and issues no extra F-side FORWARD.
These checks do not run real draft generation or accept/reject sampling.

Each attempt used the existing A3 Python 3.12.9 environment with deployed
FastAPI 0.123.10, matched all ten installed upstream source pins and started
no NPU probe. No dependencies or upstream sources were changed. The normal
V4.1 model admission guard remains closed pending checkpoint acceptance.

Each raw directory contains `identity.json`, `delivery.json`, the invocation
transcript and `results/` with summary, JUnit, stdout/stderr, source hashes and
package/environment identity. Use a fresh label with the existing
`a3-control-cpu-20261109/run_a3_graph_cpu.py` uploader to repeat a selection;
do not overwrite these attempts. The pinned native DSpark/PD call-chain review
is [recorded separately](../dspark-pd-source-contract-20261110.md).
