#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Deterministic cold-start validator for vLLM NCCL weight sync.

Usage:
  scripts/validate_cold_start_weight_sync.sh \
    --namespace <ns> \
    --notebook-pod <pod-name> \
    [--notebook-container <container-name>] \
    [--vllm-deployment grpo-vllm-rollout] \
    [--vllm-service grpo-vllm-rollout] \
    [--vllm-selector app=grpo-vllm-rollout] \
    [--model-name Qwen/Qwen2.5-0.5B-Instruct] \
    [--attempts 3] \
    [--skip-restart]

What this proves:
  1) Cold start response is unsynced (not consistently "4").
  2) Weight sync script completes successfully.
  3) Post-sync response converges to "4" for all attempts.

Failure criteria:
  - Notebook SA cannot create pods/exec.
  - /health != 200 or /get_world_size < 1.
  - Pre-sync responses are already all "4" (cold-start invalid).
  - weight_sync_test.py does not print "WEIGHT SYNC COMPLETE".
  - Post-sync responses are not all "4".
EOF
}

NAMESPACE=""
NOTEBOOK_POD=""
NOTEBOOK_CONTAINER=""
VLLM_DEPLOYMENT="grpo-vllm-rollout"
VLLM_SERVICE="grpo-vllm-rollout"
VLLM_SELECTOR="app=grpo-vllm-rollout"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
PROMPT="What is 2+2? Answer with just the number."
ATTEMPTS="3"
SKIP_RESTART="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --notebook-pod)
      NOTEBOOK_POD="$2"
      shift 2
      ;;
    --notebook-container)
      NOTEBOOK_CONTAINER="$2"
      shift 2
      ;;
    --vllm-deployment)
      VLLM_DEPLOYMENT="$2"
      shift 2
      ;;
    --vllm-service)
      VLLM_SERVICE="$2"
      shift 2
      ;;
    --vllm-selector)
      VLLM_SELECTOR="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --attempts)
      ATTEMPTS="$2"
      shift 2
      ;;
    --skip-restart)
      SKIP_RESTART="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$NAMESPACE" || -z "$NOTEBOOK_POD" ]]; then
  echo "Both --namespace and --notebook-pod are required." >&2
  usage
  exit 1
fi

if ! command -v oc >/dev/null 2>&1; then
  echo "oc CLI is required but not found in PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT_LOCAL="${SCRIPT_DIR}/weight_sync_test.py"
SYNC_SCRIPT_REMOTE="/tmp/weight_sync_test.py"
BASE_URL="http://${VLLM_SERVICE}.${NAMESPACE}.svc.cluster.local:8000"

if [[ ! -f "$SYNC_SCRIPT_LOCAL" ]]; then
  echo "Missing required script: $SYNC_SCRIPT_LOCAL" >&2
  exit 1
fi

if [[ -z "$NOTEBOOK_CONTAINER" ]]; then
  NOTEBOOK_CONTAINER="$(oc get pod "$NOTEBOOK_POD" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].name}')"
fi

if [[ -z "$NOTEBOOK_CONTAINER" ]]; then
  echo "Could not resolve notebook container name." >&2
  exit 1
fi

NOTEBOOK_SA="$(oc get pod "$NOTEBOOK_POD" -n "$NAMESPACE" -o jsonpath='{.spec.serviceAccountName}')"
if [[ -z "$NOTEBOOK_SA" ]]; then
  echo "Could not resolve notebook service account from pod ${NOTEBOOK_POD}." >&2
  exit 1
fi

CAN_EXEC_DIRECT="$(oc auth can-i create pods/exec -n "$NAMESPACE" --as "system:serviceaccount:${NAMESPACE}:${NOTEBOOK_SA}" 2>/dev/null || true)"
CAN_EXEC_SUBRESOURCE="$(oc auth can-i create pods --subresource=exec -n "$NAMESPACE" --as "system:serviceaccount:${NAMESPACE}:${NOTEBOOK_SA}" 2>/dev/null || true)"
if [[ "$CAN_EXEC_DIRECT" != "yes" && "$CAN_EXEC_SUBRESOURCE" != "yes" ]]; then
  echo "FAIL: serviceaccount ${NOTEBOOK_SA} cannot create pods/exec in ${NAMESPACE}." >&2
  echo "can-i(create pods/exec)=${CAN_EXEC_DIRECT:-<empty>}; can-i(create pods --subresource=exec)=${CAN_EXEC_SUBRESOURCE:-<empty>}" >&2
  echo "Apply k8s/notebook-pod-exec-rbac.yaml for this namespace/serviceaccount first." >&2
  exit 1
fi
echo "PASS: Notebook SA pods/exec check -> yes (${NOTEBOOK_SA}); direct=${CAN_EXEC_DIRECT:-<empty>}, subresource=${CAN_EXEC_SUBRESOURCE:-<empty>}"

if [[ "$SKIP_RESTART" != "true" ]]; then
  echo "Restarting deployment ${VLLM_DEPLOYMENT} to force cold-start unsynced state..."
  oc rollout restart "deployment/${VLLM_DEPLOYMENT}" -n "$NAMESPACE" >/dev/null
  oc rollout status "deployment/${VLLM_DEPLOYMENT}" -n "$NAMESPACE" --timeout=300s
else
  echo "Skipping restart; pre-sync behavior may be non-deterministic."
fi

resolve_vllm_pod() {
  local resolved
  resolved="$(
    oc get pod -n "$NAMESPACE" -l "$VLLM_SELECTOR" \
      -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.creationTimestamp}{"|"}{.metadata.name}{"\n"}{end}' | \
      sort -r | \
      awk -F'|' 'NR==1 {print $2}'
  )"

  if [[ -z "$resolved" ]]; then
    echo "Could not resolve running vLLM pod for selector: ${VLLM_SELECTOR}" >&2
    return 1
  fi
  VLLM_POD="$resolved"
}

oc wait --for=condition=Ready pod -n "$NAMESPACE" -l "$VLLM_SELECTOR" --timeout=300s >/dev/null
resolve_vllm_pod
echo "Using vLLM pod: ${VLLM_POD}"
echo "Using base URL: ${BASE_URL}"

wait_for_service_readiness() {
  local timeout_seconds="$1"
  local poll_interval_seconds=5
  local start_epoch
  start_epoch="$(date +%s)"

  while true; do
    local readiness_output=""
    set +e
    readiness_output="$(oc exec -i -n "$NAMESPACE" "$NOTEBOOK_POD" -c "$NOTEBOOK_CONTAINER" -- \
      env BASE_URL="$BASE_URL" python3 - <<'PY'
import os
import sys

import requests

base_url = os.environ["BASE_URL"]

try:
    health_status = requests.get(f"{base_url}/health", timeout=10).status_code
    world_size = requests.get(f"{base_url}/get_world_size", timeout=10).json()["world_size"]
    if health_status == 200 and int(world_size) >= 1:
        print(f"READY health={health_status} world_size={world_size}")
        sys.exit(0)
    print(f"NOT_READY health={health_status} world_size={world_size}")
    sys.exit(1)
except Exception as exc:  # noqa: BLE001
    print(f"NOT_READY error={exc}")
    sys.exit(1)
PY
)"
    local readiness_rc=$?
    set -e

    if [[ $readiness_rc -eq 0 ]]; then
      echo "PASS: vLLM service readiness gate -> ${readiness_output}"
      break
    fi

    local now_epoch elapsed_seconds
    now_epoch="$(date +%s)"
    elapsed_seconds="$((now_epoch - start_epoch))"
    if (( elapsed_seconds >= timeout_seconds )); then
      echo "FAIL: vLLM service did not become ready within ${timeout_seconds}s." >&2
      echo "Last readiness output: ${readiness_output}" >&2
      return 1
    fi

    echo "Waiting for vLLM service readiness (${elapsed_seconds}s/${timeout_seconds}s): ${readiness_output}" >&2
    sleep "$poll_interval_seconds"
  done
}

echo "Waiting for post-restart vLLM service readiness..."
wait_for_service_readiness 300

run_probe() {
  oc exec -i -n "$NAMESPACE" "$NOTEBOOK_POD" -c "$NOTEBOOK_CONTAINER" -- \
    env BASE_URL="$BASE_URL" MODEL_NAME="$MODEL_NAME" ATTEMPTS="$ATTEMPTS" PROMPT="$PROMPT" python3 - <<'PY'
import json
import os
import requests
from openai import OpenAI

base_url = os.environ["BASE_URL"]
model_name = os.environ["MODEL_NAME"]
attempts = int(os.environ["ATTEMPTS"])
prompt = os.environ["PROMPT"]

health_status = requests.get(f"{base_url}/health", timeout=20).status_code
world_size = requests.get(f"{base_url}/get_world_size", timeout=20).json()["world_size"]

client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy")
answers = []
for _ in range(attempts):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.0,
    )
    answers.append((completion.choices[0].message.content or "").strip())

print(json.dumps({"health_status": health_status, "world_size": world_size, "answers": answers}))
PY
}

assert_probe() {
  local stage="$1"
  local expected="$2"
  local payload="$3"

  PAYLOAD="$payload" STAGE="$stage" EXPECTED="$expected" python3 - <<'PY'
import json
import os
import re
import sys

data = json.loads(os.environ["PAYLOAD"])
stage = os.environ["STAGE"]
expected = os.environ["EXPECTED"]

health = data["health_status"]
world_size = data["world_size"]
answers = data["answers"]

if health != 200:
    print(f"FAIL[{stage}]: /health={health}, expected 200", file=sys.stderr)
    sys.exit(1)
if int(world_size) < 1:
    print(f"FAIL[{stage}]: /get_world_size={world_size}, expected >=1", file=sys.stderr)
    sys.exit(1)

normalized = [re.sub(r"[^0-9-]", "", a or "") for a in answers]
all_four = all(a == "4" for a in normalized)

print(f"{stage} answers: {answers}")
if expected == "unsynced" and all_four:
    print(
        f"FAIL[{stage}]: all answers are already '4'; cold-start unsynced proof missing.",
        file=sys.stderr,
    )
    sys.exit(1)
if expected == "synced" and not all_four:
    print(
        f"FAIL[{stage}]: expected all answers == '4' after sync, got {answers}",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

echo "Running pre-sync probe..."
PRE_PROBE="$(run_probe)"
assert_probe "pre-sync" "unsynced" "$PRE_PROBE"

SYNC_LOG="$(mktemp)"
trap 'rm -f "$SYNC_LOG"' EXIT

run_sync_once() {
  local pod_name="$1"

  echo "Copying sync harness into vLLM pod ${pod_name}..."
  oc cp "$SYNC_SCRIPT_LOCAL" "${NAMESPACE}/${pod_name}:${SYNC_SCRIPT_REMOTE}" >/dev/null

  echo "Running full weight sync in-pod on ${pod_name}..."
  oc exec -n "$NAMESPACE" "$pod_name" -- python3 "$SYNC_SCRIPT_REMOTE" | tee "$SYNC_LOG"
}

sync_succeeded="false"
for sync_attempt in 1 2; do
  resolve_vllm_pod
  if run_sync_once "$VLLM_POD"; then
    sync_succeeded="true"
    break
  fi

  echo "Sync attempt ${sync_attempt} failed; refreshing rollout readiness and retrying once..." >&2
  if [[ "$sync_attempt" -lt 2 ]]; then
    wait_for_service_readiness 300 || true
  fi
done

if [[ "$sync_succeeded" != "true" ]]; then
  echo "FAIL[sync]: could not complete in-pod weight sync after retry." >&2
  exit 1
fi

python3 - "$SYNC_LOG" <<'PY'
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
required = [
    "BEFORE weight sync:",
    "AFTER weight sync:",
    "WEIGHT SYNC COMPLETE",
]
missing = [marker for marker in required if marker not in text]
if missing:
    print(f"FAIL[sync]: missing markers in sync output: {missing}", file=sys.stderr)
    sys.exit(1)
print("PASS: sync output markers present")
PY

echo "Running post-sync probe..."
POST_PROBE="$(run_probe)"
assert_probe "post-sync" "synced" "$POST_PROBE"

echo "PASS: deterministic cold-start validation completed."
