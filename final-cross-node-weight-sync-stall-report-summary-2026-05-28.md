# Weekly Status Summary: Cross-Node vLLM Weight Sync Stall

Date: 2026-05-28  
Workstream: Kubeflow Trainer + vLLM native weight sync (split cross-node topology)

## Current Decision
**No-Go for current sprint** on cross-node native weight sync integration.

Scope of No-Go:
- Applies to split topology (trainer pod on node A, vLLM pod on node B) using `init/start/update/finish` API flow.
- Does not apply to same-pod localhost sync patterns or checkpoint refresh alternatives.

## Why this is No-Go (plain language)
The handshake step required before weight transfer is not reliably completing across nodes. The same init-stage failure repeats across multiple tests, so weight transfer never safely starts. Proceeding now would likely produce more unstable runs instead of deliverable progress.

## Evidence Snapshot
1. **Repeated init failure signatures**: timeout to trainer rendezvous endpoint, `NCCL error: remote process exited or there was a network error`, and `1/2 clients joined` timeout in trainer-side store [S1][S3].
2. **Contract drift**: repository `api_router.py` implements queued async init with status endpoints, while deployed runtime returned `404` for `/weight_transfer_status` in test environment [S2][S9].
3. **Topology/test mismatch in harness**: `weight_sync_test.py` assumes localhost and two GPUs in one pod; split topology with 1 GPU in vLLM pod triggered deterministic device ordinal failure in one run [S4][S8].
4. **Policy tuning alone did not unblock**: temporary policy widening to include a port range still did not produce successful init convergence [S1].
5. **Fallback blocked in-sprint**: Ray fallback path could not be validated end-to-end due dependency split and memory limits [S1][S2].

## Root-Cause Framing
- **Immediate cause:** distributed init rendezvous non-convergence in split cross-node topology.
- **Contributing factors:** API/runtime contract misalignment, runtime dependency asymmetry, transport/addressing fragility, and harness assumptions not matching target topology.
- **Ruled out:** wrong node placement and basic vLLM health outage.

## Sprint Recommendation
Keep this workstream in **No-Go** for sprint delivery.  
Shift sprint execution to bounded unblockers only:
1. Lock one API/runtime contract and align orchestration.
2. Establish runtime/dependency parity for selected topology.
3. Pass an init-only convergence gate before any full update attempt.

## Source Labels
- [S1] `runs_log/nccl-cross-node-autonomous-loop-2026-05-27.md`
- [S2] `runs_log/init-weight-transfer-compatibility-and-ray-test-2026-05-26.md`
- [S3] `runs_log/multinode-cross-node-run-2026-05-26-corrected-1gpu-each.md`
- [S4] `runs_log/multinode-cross-node-run-2026-05-26-kap-test-notebook.md`
- [S8] `scripts/weight_sync_test.py`
- [S9] `learning/vllm/vllm/vllm/entrypoints/serve/rlhf/api_router.py`

