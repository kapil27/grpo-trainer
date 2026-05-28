#!/usr/bin/env python3
"""Wait for KFT TrainJob + PVC checkpoint, then NCCL-sync weights into vLLM.

Usage:
  python scripts/post_train_sync.py TRAINJOB_NAME
  python scripts/post_train_sync.py TRAINJOB_NAME --namespace grpoxtrainer

Requires: oc CLI, cluster access, vLLM deployment with shared checkpoint PVC mounted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NS = os.environ.get("GRPO_NAMESPACE", "grpoxtrainer")
DEFAULT_VLLM_LABEL = os.environ.get("GRPO_VLLM_LABEL", "app=grpo-vllm-rollout")
CHECKPOINT = os.environ.get("GRPO_CHECKPOINT_PATH", "/mnt/checkpoint/grpo-trained")
SYNC_SCRIPT = "/tmp/grpo_vllm_train_sync.py"
STATE_SCRIPT = "/tmp/create_sync_state.py"
STATE_PATH = "/tmp/grpo-sync-state.json"


def oc(ns: str, *args: str, check: bool = True) -> subprocess.CompletedResult[str]:
    cmd = ["oc", "-n", ns, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def discover_vllm_pod(ns: str, label: str) -> str:
    r = oc(ns, "get", "pods", "-l", label, "-o", "jsonpath={.items[0].metadata.name}", check=False)
    pod = (r.stdout or "").strip()
    if r.returncode != 0 or not pod:
        raise RuntimeError(f"No vLLM pod found with label {label} in {ns}: {r.stderr}")
    return pod


def wait_trainjob(ns: str, job_name: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = oc(ns, "get", "trainjob", job_name, "-o", "jsonpath={.status.conditions[-1].type}", check=False)
        state = (r.stdout or "").strip()
        if state == "Complete":
            return
        if state == "Failed":
            raise RuntimeError(f"TrainJob {job_name} failed")
        time.sleep(15)
    raise TimeoutError(f"TrainJob {job_name} not complete after {timeout_s}s")


def wait_checkpoint_ready(ns: str, vllm_pod: str, timeout_s: int) -> str:
    deadline = time.time() + timeout_s
    probe = f"test -f {CHECKPOINT}/.ready && test -f {CHECKPOINT}/model.safetensors"
    while time.time() < deadline:
        r = oc(ns, "exec", vllm_pod, "--", "bash", "-c", probe, check=False)
        if r.returncode == 0:
            du = oc(ns, "exec", vllm_pod, "--", "du", "-sh", CHECKPOINT)
            return du.stdout.strip()
        time.sleep(10)
    raise TimeoutError(f"Checkpoint not ready at {CHECKPOINT} after {timeout_s}s")


def rank0_pod(ns: str, job_name: str) -> str:
    r = oc(ns, "get", "pods", "-o", "name", check=False)
    pods = [
        p.split("/", 1)[1]
        for p in (r.stdout or "").splitlines()
        if p and f"{job_name}-node-0-0-" in p
    ]
    if pods:
        return sorted(pods)[0]
    raise RuntimeError(f"No rank-0 pod for TrainJob {job_name}")


def get_train_summary(ns: str, job_name: str) -> dict:
    pod = rank0_pod(ns, job_name)
    logs = oc(ns, "logs", pod, "-c", "node").stdout
    for line in logs.splitlines():
        if line.strip().startswith('{"model"') and '"status": "ok"' in line:
            return json.loads(line.strip())
    raise RuntimeError(f"SPIKE_SUMMARY_JSON not found in logs for {pod}")


def ensure_vllm_scripts(ns: str, vllm_pod: str) -> None:
    for local_name, remote_path in (
        ("grpo_vllm_train_sync.py", SYNC_SCRIPT),
        ("create_sync_state.py", STATE_SCRIPT),
    ):
        local = SCRIPT_DIR / local_name
        if not local.exists():
            continue
        subprocess.run(
            ["oc", "-n", ns, "cp", str(local), f"{vllm_pod}:{remote_path}"],
            check=True,
            capture_output=True,
            text=True,
        )


def ensure_sync_deps(ns: str, vllm_pod: str) -> None:
    oc(
        ns,
        "exec",
        vllm_pod,
        "--",
        "pip",
        "install",
        "-q",
        "trl",
        "transformers",
        "accelerate",
        check=False,
    )


def create_sync_state(ns: str, vllm_pod: str, world_size: int, train_elapsed: float) -> None:
    r = oc(
        ns,
        "exec",
        vllm_pod,
        "--",
        "python3",
        "-u",
        STATE_SCRIPT,
        str(world_size),
        str(train_elapsed),
        "0.0",
        "0.0",
        STATE_PATH,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"create_sync_state failed:\n{r.stderr}\n{r.stdout}")


def run_sync(ns: str, vllm_pod: str) -> str:
    r = oc(
        ns,
        "exec",
        vllm_pod,
        "--",
        "env",
        "CUDA_VISIBLE_DEVICES=0,1",
        "python3",
        "-u",
        SYNC_SCRIPT,
        "--sync-only",
        "--checkpoint",
        CHECKPOINT,
        "--state",
        STATE_PATH,
        check=False,
    )
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(f"Weight sync failed ({r.returncode}):\n{out[-4000:]}")
    return out


def parse_results_json(sync_out: str) -> dict:
    m = re.search(r"__RESULTS_JSON__(\{.*\})", sync_out, re.S)
    if not m:
        raise RuntimeError("Sync output missing __RESULTS_JSON__")
    return json.loads(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-train vLLM weight sync")
    parser.add_argument("job_name", help="KFT TrainJob name")
    parser.add_argument("--namespace", "-n", default=DEFAULT_NS)
    parser.add_argument("--vllm-label", default=DEFAULT_VLLM_LABEL)
    parser.add_argument("--train-timeout", type=int, default=3600)
    parser.add_argument("--checkpoint-timeout", type=int, default=600)
    args = parser.parse_args()

    vllm_pod = discover_vllm_pod(args.namespace, args.vllm_label)
    print(f"vLLM pod: {vllm_pod}")

    print(f"Waiting for TrainJob {args.job_name}...")
    wait_trainjob(args.namespace, args.job_name, args.train_timeout)
    print("TrainJob complete.")

    summary = get_train_summary(args.namespace, args.job_name)
    print(json.dumps(summary, indent=2))

    print("Waiting for checkpoint on PVC...")
    size = wait_checkpoint_ready(args.namespace, vllm_pod, args.checkpoint_timeout)
    print(f"Checkpoint ready: {size}")

    print("Preparing vLLM pod for sync...")
    ensure_vllm_scripts(args.namespace, vllm_pod)
    ensure_sync_deps(args.namespace, vllm_pod)

    world_size = int(summary.get("world_size", 1))
    train_elapsed = float(summary.get("train_runtime_s", summary.get("benchmarks_s", {}).get("train_wall_s", 0)))
    create_sync_state(args.namespace, vllm_pod, world_size, train_elapsed)

    print("Running NCCL weight sync...")
    sync_out = run_sync(args.namespace, vllm_pod)
    results = parse_results_json(sync_out)

    print("\n=== POST-TRAIN SYNC COMPLETE ===")
    print(f"  Sync time:      {results.get('sync_elapsed_s')}s")
    print(f"  Before accuracy: {results['pre_eval']['accuracy']:.1%}")
    print(f"  After accuracy:  {results['post_eval']['accuracy']:.1%}")
    print(f"  TrainJob:        {args.job_name}")
    print(f"  Checkpoint:      {CHECKPOINT}")


if __name__ == "__main__":
    main()
