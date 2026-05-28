# GRPO Research Repro Package

This package adds reproducible, namespace-scoped research artifacts for:

1. Notebook `pods/exec` prerequisites.
2. Canonical weight-sync `NetworkPolicy`.
3. Deterministic cold-start validation (unsynced -> synced).
4. vLLM baseline capture (latency/TTFT/throughput/GPU/error-rate).

The flow is designed for research namespaces and minimizes mutation by default; only the explicit validation command restarts the rollout to force deterministic cold-start behavior.

## 0) Assumptions and risks

- You are targeting a research namespace and can use `oc`.
- Notebook image already includes Python deps used by existing notebook (`openai`, `requests`).
- `scripts/weight_sync_test.py` is valid for your deployed vLLM model/runtime.
- Deterministic proof depends on cold start: skipping rollout restart can produce false positives (already synced output before test).

## 1) Set environment once

Run from repo root (`/Users/knema/Project/agentic-ai-skills/trainer-GRPO/research/grpo-trainer`):

```bash
export NAMESPACE="kapil-test"
export NOTEBOOK_NAME="kap-test-notebook"
export NOTEBOOK_SERVICE_ACCOUNT="kap-test-notebook"
export NOTEBOOK_POD="kap-test-notebook-0"
export NOTEBOOK_CONTAINER="kap-test-notebook"
export VLLM_APP_LABEL="grpo-vllm-rollout"
export WEIGHT_SYNC_PORT="29501"
```

## 2) Apply reproducibility prerequisites (RBAC + canonical policy)

### 2.1 Notebook SA `pods/exec` RBAC (namespace-scoped)

```bash
envsubst < k8s/notebook-pod-exec-rbac.yaml | oc apply -f -
oc auth can-i create pods/exec -n "${NAMESPACE}" --as "system:serviceaccount:${NAMESPACE}:${NOTEBOOK_SERVICE_ACCOUNT}"
```

Expected: final command returns `yes`.

### 2.2 Canonical weight-sync NetworkPolicy (single source of truth)

```bash
envsubst < k8s/vllm-weight-sync-networkpolicy.yaml | oc apply -f -
oc get netpol -n "${NAMESPACE}" | grep -E "vllm-weight-sync|vllm-weight-sync-ingress"
```

If historical duplicate policies exist, remove them and keep only `${NOTEBOOK_NAME}-vllm-weight-sync`:

```bash
oc delete netpol -n "${NAMESPACE}" vllm-weight-sync-ingress --ignore-not-found
oc delete netpol -n "${NAMESPACE}" "${NOTEBOOK_NAME}-vllm-weight-sync" --ignore-not-found
envsubst < k8s/vllm-weight-sync-networkpolicy.yaml | oc apply -f -
```

Notes:
- The delete/apply sequence intentionally resets to exactly one canonical policy.
- The canonical manifest is `k8s/vllm-weight-sync-networkpolicy.yaml`; avoid creating notebook-generated duplicates.

## 3) Rollback (research environment)

```bash
envsubst < k8s/notebook-pod-exec-rbac.yaml | oc delete -f - --ignore-not-found
envsubst < k8s/vllm-weight-sync-networkpolicy.yaml | oc delete -f - --ignore-not-found
```

Optional cleanup for historical names:

```bash
oc delete netpol -n "${NAMESPACE}" vllm-weight-sync-ingress --ignore-not-found
```

## 4) Deterministic cold-start validation (required proof)

Run:

```bash
chmod +x scripts/validate_cold_start_weight_sync.sh
scripts/validate_cold_start_weight_sync.sh \
  --namespace "${NAMESPACE}" \
  --notebook-pod "${NOTEBOOK_POD}" \
  --notebook-container "${NOTEBOOK_CONTAINER}" \
  --vllm-service grpo-vllm-rollout \
  --vllm-selector "app=grpo-vllm-rollout" \
  --vllm-deployment grpo-vllm-rollout
```

### Expected signals

- SA check passes: `pods/exec -> yes`.
- Pre-sync probe reports healthy service + world size >= 1.
- Pre-sync answers are **not all `"4"`** (unsynced proof).
- In-pod sync run prints:
  - `BEFORE weight sync:`
  - `AFTER weight sync:`
  - `WEIGHT SYNC COMPLETE`
- Post-sync answers are all `"4"` (synced proof).

### Failure criteria

- Any probe API failure (`/health`, `/get_world_size`, chat request).
- Pre-sync already all `"4"` after restart (cold-start assumption violated).
- Missing sync output markers.
- Post-sync not converged to `"4"` on all attempts.

## 5) vLLM baseline capture

Run:

```bash
chmod +x scripts/capture_vllm_performance_baseline.sh
scripts/capture_vllm_performance_baseline.sh \
  --namespace "${NAMESPACE}" \
  --notebook-pod "${NOTEBOOK_POD}" \
  --notebook-container "${NOTEBOOK_CONTAINER}" \
  --vllm-service grpo-vllm-rollout \
  --vllm-selector "app=grpo-vllm-rollout" \
  --requests 30 \
  --max-tokens 64 \
  --output-dir runs_log
```

Artifacts written under `runs_log/`:
- `vllm-baseline-raw-<timestamp>.json`: per-request raw measurements.
- `vllm-baseline-summary-<timestamp>.md`: summary with key metrics.
- `vllm-baseline-nvidia-smi-<timestamp>.txt`: GPU util/mem snapshot.
- `vllm-baseline-pod-state-<timestamp>.txt`: pod restarts/state snapshot.

Primary metrics to report:
- Latency p50/p95 (ms)
- TTFT p50/p95 (ms)
- Throughput (req/s)
- Token throughput (approx token/s)
- GPU utilization/memory (`nvidia-smi`)
- Error rate (failed/total requests)
