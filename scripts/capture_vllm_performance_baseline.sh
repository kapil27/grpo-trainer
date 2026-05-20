#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Capture a reproducible vLLM baseline from a notebook pod.

Usage:
  scripts/capture_vllm_performance_baseline.sh \
    --namespace <ns> \
    --notebook-pod <pod-name> \
    [--notebook-container <container-name>] \
    [--vllm-service grpo-vllm-rollout] \
    [--vllm-selector app=grpo-vllm-rollout] \
    [--model-name Qwen/Qwen2.5-0.5B-Instruct] \
    [--requests 30] \
    [--max-tokens 64] \
    [--output-dir runs_log]
EOF
}

NAMESPACE=""
NOTEBOOK_POD=""
NOTEBOOK_CONTAINER=""
VLLM_SERVICE="grpo-vllm-rollout"
VLLM_SELECTOR="app=grpo-vllm-rollout"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
REQUESTS="30"
MAX_TOKENS="64"
OUTPUT_DIR="runs_log"
PROMPT="Give a one-line definition of reinforcement learning."

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
    --requests)
      REQUESTS="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
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

if [[ -z "$NOTEBOOK_CONTAINER" ]]; then
  NOTEBOOK_CONTAINER="$(oc get pod "$NOTEBOOK_POD" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].name}')"
fi

if [[ -z "$NOTEBOOK_CONTAINER" ]]; then
  echo "Could not resolve notebook container name." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RAW_JSON="${OUTPUT_DIR}/vllm-baseline-raw-${TIMESTAMP}.json"
SUMMARY_MD="${OUTPUT_DIR}/vllm-baseline-summary-${TIMESTAMP}.md"
GPU_SMI="${OUTPUT_DIR}/vllm-baseline-nvidia-smi-${TIMESTAMP}.txt"
POD_STATE="${OUTPUT_DIR}/vllm-baseline-pod-state-${TIMESTAMP}.txt"

BASE_URL="http://${VLLM_SERVICE}.${NAMESPACE}.svc.cluster.local:8000"
VLLM_POD="$(oc get pod -n "$NAMESPACE" -l "$VLLM_SELECTOR" -o jsonpath='{.items[0].metadata.name}')"
if [[ -z "$VLLM_POD" ]]; then
  echo "Could not resolve vLLM pod for selector: ${VLLM_SELECTOR}" >&2
  exit 1
fi

echo "Capturing baseline via notebook pod=${NOTEBOOK_POD}, vllm pod=${VLLM_POD}"

oc exec -i -n "$NAMESPACE" "$NOTEBOOK_POD" -c "$NOTEBOOK_CONTAINER" -- \
  env BASE_URL="$BASE_URL" MODEL_NAME="$MODEL_NAME" REQUESTS="$REQUESTS" MAX_TOKENS="$MAX_TOKENS" PROMPT="$PROMPT" python3 - <<'PY' >"$RAW_JSON"
import json
import os
import statistics
import time

import requests
from openai import OpenAI

base_url = os.environ["BASE_URL"]
model_name = os.environ["MODEL_NAME"]
requests_n = int(os.environ["REQUESTS"])
max_tokens = int(os.environ["MAX_TOKENS"])
prompt = os.environ["PROMPT"]

client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy")
health_status = requests.get(f"{base_url}/health", timeout=20).status_code
world_size = requests.get(f"{base_url}/get_world_size", timeout=20).json()["world_size"]

latencies_ms = []
ttft_ms = []
output_tokens = []
errors = []

bench_start = time.perf_counter()
for i in range(requests_n):
    try:
        t0 = time.perf_counter()
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        )
        first_token_at = None
        content_parts = []
        for chunk in stream:
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                content_parts.append(delta)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)
        if first_token_at is None:
            ttft_ms.append(None)
        else:
            ttft_ms.append((first_token_at - t0) * 1000.0)
        token_count = len("".join(content_parts).split())
        output_tokens.append(token_count)
    except Exception as exc:  # noqa: BLE001
        errors.append({"request_index": i, "error": str(exc)})

bench_end = time.perf_counter()
total_s = bench_end - bench_start
ok = len(latencies_ms)
error_rate = (len(errors) / requests_n) if requests_n else 1.0

def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    idx = int((len(values) - 1) * p)
    return values[idx]

ttft_observed = [x for x in ttft_ms if x is not None]

result = {
    "metadata": {
        "base_url": base_url,
        "model_name": model_name,
        "requests": requests_n,
        "max_tokens": max_tokens,
        "prompt": prompt,
    },
    "signals": {
        "health_status": health_status,
        "world_size": world_size,
        "successful_requests": ok,
        "failed_requests": len(errors),
        "error_rate": error_rate,
        "total_wall_time_s": total_s,
        "throughput_rps": (ok / total_s) if total_s > 0 else None,
        "token_throughput_tokens_per_s": (sum(output_tokens) / total_s) if total_s > 0 else None,
        "latency_ms_p50": percentile(latencies_ms, 0.50),
        "latency_ms_p95": percentile(latencies_ms, 0.95),
        "latency_ms_mean": statistics.mean(latencies_ms) if latencies_ms else None,
        "ttft_ms_p50": percentile(ttft_observed, 0.50),
        "ttft_ms_p95": percentile(ttft_observed, 0.95),
        "ttft_ms_mean": statistics.mean(ttft_observed) if ttft_observed else None,
    },
    "samples": {
        "latencies_ms": latencies_ms,
        "ttft_ms": ttft_ms,
        "output_tokens_approx": output_tokens,
        "errors": errors,
    },
}

print(json.dumps(result, indent=2))
PY

oc exec -n "$NAMESPACE" "$VLLM_POD" -- nvidia-smi >"$GPU_SMI"
oc get pod "$VLLM_POD" -n "$NAMESPACE" -o wide >"$POD_STATE"
oc get pod "$VLLM_POD" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[*].restartCount}' >>"$POD_STATE"
echo >>"$POD_STATE"

python3 - "$RAW_JSON" "$SUMMARY_MD" <<'PY'
import datetime as dt
import json
import pathlib
import sys

raw = pathlib.Path(sys.argv[1])
summary = pathlib.Path(sys.argv[2])
data = json.loads(raw.read_text())

signals = data["signals"]
meta = data["metadata"]
run_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

summary.write_text(
    "\n".join(
        [
            f"# vLLM Baseline Capture ({run_at})",
            "",
            f"- Namespace service URL: `{meta['base_url']}`",
            f"- Model: `{meta['model_name']}`",
            f"- Requests: `{meta['requests']}`",
            f"- Max tokens: `{meta['max_tokens']}`",
            "",
            "## Key metrics",
            "",
            f"- Health status: `{signals['health_status']}`",
            f"- World size: `{signals['world_size']}`",
            f"- Error rate: `{signals['error_rate']:.4f}`",
            f"- Throughput (req/s): `{signals['throughput_rps']}`",
            f"- Token throughput (approx token/s): `{signals['token_throughput_tokens_per_s']}`",
            f"- Latency p50/p95 (ms): `{signals['latency_ms_p50']}` / `{signals['latency_ms_p95']}`",
            f"- TTFT p50/p95 (ms): `{signals['ttft_ms_p50']}` / `{signals['ttft_ms_p95']}`",
            "",
            "## Artifacts",
            "",
            f"- Raw JSON: `{raw}`",
            "",
            "Attach matching nvidia-smi and pod-state files captured in the same run.",
            "",
        ]
    )
)
PY

echo "Baseline capture complete."
echo "Raw metrics: $RAW_JSON"
echo "Summary:     $SUMMARY_MD"
echo "GPU stats:   $GPU_SMI"
echo "Pod state:   $POD_STATE"
