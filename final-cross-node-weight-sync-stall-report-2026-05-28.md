# Cross-Node vLLM Weight Sync Investigation Report

Date: 2026-05-28  
Author: Cursor investigation agent  
Decision audience: Engineering leads, program leadership, and platform owners

---

## 1) Executive Summary (for non-technical readers)

This investigation confirms that cross-node weight sync between Kubeflow Trainer and vLLM is currently failing at initialization, before any model weights are transferred. The failure is repeatable across multiple attempts and appears at the same step: distributed rendezvous for `init_weight_transfer_engine` does not converge in the split topology (trainer notebook pod on one node, vLLM rollout pod on another) [S1][S3].

The workstream is stalled because the current runtime shape is internally inconsistent for this feature path: trainer-side code expects capabilities that are either missing or differently implemented in deployed components (for example, missing trainer-side `vllm` package in one path, API contract drift between repository code and deployed runtime, and unresolved cross-node rendezvous instability even after policy adjustments) [S1][S2][S9].

Decision for this sprint is **No-Go** for cross-node native vLLM weight sync integration. This is a scope-limited No-Go: it does **not** invalidate single-pod localhost sync patterns already known to work, and it does **not** invalidate checkpoint-based model refresh flows [S6][S7][S8].

---

## 2) Decision Statement (Go/No-Go) and Exact Scope

### Decision
**No-Go for current sprint**.

### Exact scope of this No-Go
No-Go applies to:  
- Kubeflow Trainer integration with vLLM native weight-sync API (`/init_weight_transfer_engine` -> `/start_weight_update` -> `/update_weights` -> `/finish_weight_update`) in **split cross-node topology**.

No-Go does **not** apply to:  
- Same-pod localhost weight-sync workflow.  
- Non-live checkpoint reload/restart workflows for model refresh.

### Why this decision is justified now
- Repeated init-stage failure signatures across runs (timeouts, partial client join, remote process/network error) [S1][S3].  
- Code/runtime contract mismatch between repository API behavior and deployed runtime behavior [S2][S9].  
- Environment constraints blocked fallback paths within sprint boundary (dependency split across pods, Ray memory cap) [S1][S2].

---

## 3) Objective and Non-objectives

### Objective
Determine, using direct evidence, whether cross-node Kubeflow Trainer -> vLLM weight sync is ready to move forward in this sprint.

### Non-objectives
- Redesigning vLLM architecture.  
- Implementing upstream vLLM fixes.  
- Benchmarking final model quality/latency under production workloads.  
- Approving cluster-wide infra changes outside the tested namespace.

---

## 4) Test Environment and Topologies

### Environment under test

| Item | Observed state |
|---|---|
| Namespace | `kapil-test` [S1][S3][S4] |
| Trainer-side pod | `kap-test-notebook-0` [S1][S3] |
| vLLM pod | `grpo-vllm-rollout-77d7f8c6cd-99nz7` [S1][S3] |
| Placement | Different nodes (split topology confirmed) [S1][S3] |
| vLLM health | `/health` and `/get_world_size` reachable in several runs [S1][S3][S4] |
| vLLM world size | `1` in tested rollout [S3][S4] |

### Topologies exercised

1. **Split cross-node (target topology)**  
   Trainer notebook pod on node A, vLLM rollout pod on node B [S1][S3][S4].

2. **Same-pod localhost reference topology (known working baseline in prior notes/scripts)**  
   Trainer and vLLM colocated; script explicitly instructs execution inside vLLM pod [S8].

---

## 5) Methodology and Validation Criteria

### Method used
- Chronological replay and analysis of mandatory run logs and code artifacts.  
- Correlation of runtime failure signatures with API/control-flow code.  
- Negative testing to rule out likely non-causes (policy range widening, DNS/service path checks, process-vs-thread init variation) [S1][S3][S2].

### Validation criteria (decision gates)

| Gate | Definition | Result |
|---|---|---|
| C1 | Cross-node split topology confirmed | Pass [S1] |
| C2 | `init_weight_transfer_engine` converges with trainer counterpart | Fail [S1][S3] |
| C3 | `start/update/finish` reached and completed safely | Fail (not reached after reliable init) [S1] |
| C4 | Reproducible evidence captured (logs, commands, code references) | Pass [S1][S2][S3][S9] |

### Reporting structure basis (external)
This report format follows established incident/post-incident engineering guidance emphasizing: summary, impact/scope, timeline, root cause vs contributing factors, and corrective actions backed by data [X1][X2].

---

## 6) Experiment Timeline (Chronological)

| Date | Activity | Outcome |
|---|---|---|
| 2026-05-19 | Distributed weight-sync failure analysis produced with synchronous-init concern and cross-pod instability notes [S7] | Cross-pod path flagged as not production-ready |
| 2026-05-26 (runbook execution) | Notebook runbook failed early on service name resolution from host context [S5] | Execution context/path issue, no distributed init reached |
| 2026-05-26 (multinode run) | Cross-node placement validated; harness failed with `invalid device ordinal` due 1-GPU pod vs script expecting GPU1 trainer [S4][S8] | Env/config mismatch prevented meaningful init validation |
| 2026-05-26 (corrected 1-GPU each) | Corrected world-size/rank-offset attempts run; init rendezvous still stalls/timeouts [S3] | First stable hard-fail signature at init |
| 2026-05-26 (compat + Ray) | API drift identified (`/weight_transfer_status` 404 in deployed runtime), Ray fallback blocked by 4Gi memory cap [S2] | Contract and resource blockers persisted |
| 2026-05-27 (autonomous loop T1-T6) | Iterative tests (fixed ports, temp policy range, DNS/IP path, process-sequencing, Ray sanity) [S1] | C2/C3 remain failed; stall confirmed |

---

## 7) Findings with Evidence

### Finding 1: The exact failure point is init convergence, before weight update.
**Evidence**
- Repeated exceptions during init path in vLLM logs:  
  - `Exception: Call to collective_rpc method failed: The client socket has timed out after 300000ms while trying to connect to (10.129.30.85, 29501)` [S1].  
  - `RuntimeError: NCCL error: remote process exited or there was a network error` [S1].  
- Trainer-side attempt reports:  
  - `DistStoreError: Timed out after 301 seconds waiting for clients. 1/2 clients joined.` [S3].

**Interpretation (non-speculative)**  
The cluster repeatedly fails to bring both sides into the same distributed init group. Because init does not converge, update steps are never reliably entered [S1][S3].

---

### Finding 2: Code-level API contract and deployed runtime behavior are misaligned.
**Code evidence**
- Repository `api_router.py` shows async init semantics:  
  - `POST /init_weight_transfer_engine` queues an operation and returns `202` with `operation_id`.  
  - Status is tracked via `/weight_transfer_status` endpoints [S9].

**Runtime evidence**
- Deployed runtime queried in testing returns `404` for `/weight_transfer_status`, indicating older/different API surface than repository head expectations [S2].

**Interpretation (non-speculative)**  
Even if transport were stable, orchestration logic must match the deployed API contract. Current environment shows contract drift between code reference and active runtime [S2][S9].

---

### Finding 3: The main test harness is topology-specific and can misfit split topology.
**Code evidence**
- `weight_sync_test.py` explicitly states: run inside vLLM pod, `vLLM uses GPU 0, trainer uses GPU 1`, and binds to `BASE_URL = "http://localhost:8000"` [S8].

**Runtime evidence**
- In split topology with 1 GPU in vLLM pod, attempt failed with `CUDA error: invalid device ordinal` in earlier run [S4].

**Interpretation (non-speculative)**  
The script is designed for colocated execution, not as-is for split cross-node topology. This caused at least one deterministic failure mode unrelated to final cross-node rendezvous behavior [S4][S8].

---

### Finding 4: Network policy was a real constraint, but not the sole blocker.
**Evidence**
- Baseline policy allowed only `TCP 29501` ingress from vLLM to notebook [S1][S3].  
- A temporary widened range (`30000-30100`) was applied and then reverted; init still failed after widening [S1].

**Interpretation (non-speculative)**  
Port restrictions explain some failures (ephemeral ports), but widening did not unblock convergence. Therefore policy narrowness is a contributing factor, not a complete explanation [S1].

---

### Finding 5: Addressing path is fragile in current service shape.
**Evidence**
- Direct pod IP path to notebook rendezvous port succeeded in connectivity checks.  
- Notebook service/FQDN path for port `29501` was not viable in tested shape [S1].

**Interpretation (non-speculative)**  
The path currently depends on direct pod IP addressing for rendezvous. This increases operational fragility and complicates stable production orchestration [S1].

---

### Finding 6: Dependency/runtime split blocked practical fallback.
**Evidence**
- Notebook pod lacked `vllm` in key attempt; vLLM pod had `vllm` but environment split prevented straightforward trainer-side protocol execution in notebook [S1].  
- Ray fallback check: one pod had Ray without vLLM, the other had vLLM without Ray; additional run showed Ray actor startup blocked by 4Gi memory ceiling [S1][S2].

**Interpretation (non-speculative)**  
Fallback routes were not executable within current pod/runtime constraints in this sprint window [S1][S2].

---

## 8) What Was Ruled Out (Negative Findings)

| Hypothesis | Ruling | Evidence |
|---|---|---|
| "Pods are accidentally on same node" | Ruled out | Distinct node names/IPs repeatedly verified [S1][S3][S4] |
| "vLLM service is simply down" | Ruled out | `/health` and `/get_world_size` respond 200 in multiple runs [S1][S3][S4] |
| "Only ephemeral port policy caused the issue" | Ruled out as sole cause | Failure persisted after temporary range allow [S1] |
| "Threading deadlock alone explains failures" | Ruled out as sole cause | Process-based init attempt still timed out with 1/2 clients joined [S3] |
| "This is only a naming/DNS typo issue" | Partially ruled out | DNS/service path had limits, but direct pod IP path was used and init still failed [S1] |

---

## 9) Root-Cause Analysis

### Immediate cause (direct blocker)
Distributed initialization for weight transfer does not converge in split cross-node topology. The system fails at the init rendezvous stage with consistent timeout and partial-join signatures [S1][S3].

### Contributing factors
1. **Control-plane contract drift** between repository async-init API and deployed runtime endpoints [S2][S9].  
2. **Runtime/dependency asymmetry** across pods (trainer-side missing `vllm` in key path; split capabilities) [S1].  
3. **Transport fragility** (policy constraints plus reliance on direct pod IP rendezvous path) [S1][S3].  
4. **Test harness topology mismatch** (localhost/two-GPU assumptions in script) that introduced early failures in some runs [S4][S8].  
5. **Resource ceiling on fallback path** (Ray blocked by memory limit) [S2].

### Non-causes (explicit)
- Not caused by lack of cross-node placement.  
- Not caused by vLLM basic service health.  
- Not explained solely by one policy port value.

### Uncertainty statement
It is **not yet proven** which single contributing factor is the dominant one under fully aligned runtime versions and dependencies, because those preconditions were not jointly satisfied in current sprint runs [S1][S2][S3]. The immediate blocker itself is proven (init non-convergence).

---

## 10) Why We Are Stalling This Now

### Plain-language reason
The team cannot reliably start the weight-sync handshake across nodes. Until that handshake is reliable, there is no safe path to transfer model weights, so further feature work would be building on a failing foundation [S1][S3].

### Technical reason
Progress now requires coordinated changes across API contract alignment, runtime packaging/dependency parity, and distributed rendezvous reliability. These are foundational integration tasks, not incremental tuning, and they exceeded current sprint execution bandwidth [S1][S2][S9].

---

## 11) Risk If We Proceed Anyway

| Risk | Expected impact |
|---|---|
| Repeated init hangs/timeouts | Unpredictable pipeline runtime and missed sprint commitments [S1][S3] |
| Partial/failed sync attempts | Potential model state inconsistency and non-deterministic serving behavior |
| Operational complexity growth without stability | Increased debugging cost and leadership visibility risk |
| Technical debt lock-in | Team may overfit around unstable assumptions (direct pod IP, incompatible scripts, drifted API contracts) |

---

## 12) Recommendation (Current Sprint) + Bounded Next Steps

### Recommendation for this sprint
Keep decision at **No-Go** for cross-node native weight sync integration.

### Bounded next steps (next decision cycle)
1. **Contract lock (must-pass):** pin and document one API/runtime contract (deployed behavior vs repo-head behavior), then align trainer orchestration accordingly [S2][S9].  
2. **Runtime parity (must-pass):** ensure required trainer-side and vLLM-side dependencies are intentionally present for the selected topology [S1][S8].  
3. **Init-only gate (must-pass):** run dedicated init convergence test with deterministic addressing and policy set; success criterion is repeatable no-timeout group formation.  
4. **End-to-end gate (must-pass):** only after init pass, run `start -> update -> finish` with explicit success markers and rollback-safe logging.  
5. **Fallback scope protection:** if gates 1-4 are not green quickly, continue with known-safe alternatives (same-pod sync or checkpoint refresh path) [S6][S7].

---

## 13) Appendix

### A) Command snippets (representative)

```bash
kubectl get pod -n kapil-test kap-test-notebook-0 grpo-vllm-rollout-77d7f8c6cd-99nz7 -o wide
kubectl get svc -n kapil-test grpo-vllm-rollout -o wide
kubectl describe networkpolicy -n kapil-test kap-test-notebook-vllm-weight-sync
kubectl logs -n kapil-test grpo-vllm-rollout-77d7f8c6cd-99nz7 -c vllm --tail=300
```

```bash
# API shape check from notebook pod
python -c "import requests, json; base='http://grpo-vllm-rollout:8000'; obj=requests.get(base+'/openapi.json',timeout=20).json(); print([p for p in obj.get('paths',{}) if 'weight' in p or 'world' in p])"
```

```bash
# Representative init payload used
{
  "init_info": {
    "master_address": "10.129.30.85",
    "master_port": 29501,
    "rank_offset": 1,
    "world_size": 2
  }
}
```

### B) Evidence source index

- **[S1]** `research/grpo-trainer/runs_log/nccl-cross-node-autonomous-loop-2026-05-27.md`  
  (T1-T6 loop, failure signatures, policy experiments, addressing checks, dependency split)
- **[S2]** `research/grpo-trainer/runs_log/init-weight-transfer-compatibility-and-ray-test-2026-05-26.md`  
  (API/runtime compatibility assessment, `/weight_transfer_status` 404, Ray memory blocker)
- **[S3]** `research/grpo-trainer/runs_log/multinode-cross-node-run-2026-05-26-corrected-1gpu-each.md`  
  (corrected split-topology attempts, init timeout, `1/2 clients joined`)
- **[S4]** `research/grpo-trainer/runs_log/multinode-cross-node-run-2026-05-26-kap-test-notebook.md`  
  (cross-node proof, initial env/config failure due device ordinal)
- **[S5]** `research/grpo-trainer/runs_log/grpo-vllm-cross-node-runbook-execution-2026-05-26.md`  
  (runbook early failure in host-context name resolution)
- **[S6]** `research/grpo-trainer/vllm-weight-sync-limitations.md`  
  (documented practical limitations and known-working/known-failing modes)
- **[S7]** `research/grpo-trainer/distributed-weight-sync-vllm-api-report-2026-05-19.md`  
  (earlier cross-pod failure framing and API-level fix direction)
- **[S8]** `research/grpo-trainer/scripts/weight_sync_test.py`  
  (code-level assumptions: localhost, GPU0+GPU1 colocated flow)
- **[S9]** `learning/vllm/vllm/vllm/entrypoints/serve/rlhf/api_router.py`  
  (code-level async queued init with status endpoints in repo code)
- **[X1]** Atlassian incident postmortem handbook: <https://www.atlassian.com/incident-management/handbook/postmortems>  
  (recommended report components: summary, impact, timeline, five whys/root cause, corrective actions)
- **[X2]** AWS Well-Architected REL12-BP02: <https://docs.aws.amazon.com/wellarchitected/latest/framework/rel_testing_resiliency_rca_resiliency.html>  
  (standardized, evidence-backed, blameless post-incident analysis with corrective actions)

### C) Claim -> evidence mapping (major claims)

| Claim ID | Claim | Evidence |
|---|---|---|
| C-01 | Failure occurs at init convergence before update stage | [S1], [S3] |
| C-02 | Repeated timeout/network signatures are present | [S1], [S3] |
| C-03 | Repo API contract differs from deployed runtime behavior | [S2], [S9] |
| C-04 | Main script assumes same-pod localhost + 2 GPU layout | [S8], [S4] |
| C-05 | Network policy constraints existed but were not sole cause | [S1], [S3] |
| C-06 | Cross-node topology was correctly established | [S1], [S3], [S4] |
| C-07 | vLLM basic service health was available during tests | [S1], [S3], [S4] |
| C-08 | Fallback path was blocked by env/resource constraints | [S1], [S2] |
| C-09 | Sprint decision No-Go is limited to split cross-node native sync | [S1]-[S9] combined |

