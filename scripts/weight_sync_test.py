#!/usr/bin/env python3
"""
Weight sync test with Prometheus metrics — matches the official vLLM
rlhf_http_nccl.py pattern. Run INSIDE the vLLM pod (oc exec) so trainer
and vLLM share localhost.

vLLM uses GPU 0, trainer uses GPU 1.

Usage:
    oc exec <vllm-pod> -- python3 /tmp/weight_sync_test.py
"""

import json as _json
import threading
import time

import requests
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM

from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port

BASE_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
METRICS_URL = f"{BASE_URL}/metrics"
BENCHMARK_N = 20
BENCHMARK_PROMPTS = [
    "What is 2+2? Answer with just the number.",
    "Name three primary colors.",
    "What planet is closest to the sun?",
    "Translate 'hello' to French.",
    "What is the square root of 144?",
]

TRACKED_METRICS = [
    "vllm:avg_generation_throughput_toks_per_s",
    "vllm:avg_prompt_throughput_toks_per_s",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
]


def scrape_metrics():
    raw = requests.get(METRICS_URL, timeout=10).text
    metrics = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        for key in TRACKED_METRICS:
            if line.startswith(key):
                try:
                    metrics[key] = float(line.split()[-1])
                except ValueError:
                    pass
    return metrics


def run_benchmark(label, n=BENCHMARK_N):
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="dummy")
    latencies = []
    tok_counts = []
    for i in range(n):
        prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
        )
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        tok_counts.append(r.usage.completion_tokens)

    avg_lat = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    total_toks = sum(tok_counts)
    total_time = sum(latencies)
    tps = total_toks / total_time if total_time > 0 else 0

    result = dict(
        label=label,
        requests=n,
        avg_latency_s=round(avg_lat, 4),
        p50_latency_s=round(p50, 4),
        p99_latency_s=round(p99, 4),
        total_tokens=total_toks,
        tokens_per_sec=round(tps, 2),
    )
    print(f"\n[{label}] Benchmark ({n} requests):")
    for k, v in result.items():
        if k != "label":
            print(f"  {k}: {v}")
    return result


def main():
    inference_world_size = requests.get(
        f"{BASE_URL}/get_world_size", timeout=10
    ).json()["world_size"]
    world_size = inference_world_size + 1

    device = f"cuda:{inference_world_size}"
    torch.cuda.set_device(int(device.split(":")[1]))
    print(f"Trainer on {device}, vLLM world_size={inference_world_size}, total={world_size}")

    print(f"Loading model: {MODEL_NAME}")
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    train_model.to(device)

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="dummy")

    print("=" * 60)
    print("BEFORE weight sync (dummy weights):")
    print("=" * 60)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        max_tokens=16,
    )
    print(f"  Answer: {resp.choices[0].message.content}")
    pre_metrics = scrape_metrics()
    pre_bench = run_benchmark("BEFORE")

    master_address = get_ip()
    master_port = get_open_port()
    print(f"\nNCCL init: master={master_address}:{master_port} world_size={world_size}")

    init_thread = threading.Thread(
        target=lambda: requests.post(
            f"{BASE_URL}/init_weight_transfer_engine",
            json={"init_info": dict(
                master_address=master_address,
                master_port=master_port,
                rank_offset=1,
                world_size=world_size,
            )},
            timeout=120,
        ).raise_for_status(),
    )
    init_thread.start()

    sync_t0 = time.perf_counter()
    model_update_group = NCCLWeightTransferEngine.trainer_init(dict(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
    ))
    init_thread.join()
    print("NCCL group established")

    requests.post(f"{BASE_URL}/pause", timeout=30).raise_for_status()
    print("vLLM paused")

    requests.post(
        f"{BASE_URL}/start_weight_update",
        json={"is_checkpoint_format": True},
        timeout=60,
    ).raise_for_status()

    names, dtype_names, shapes = [], [], []
    for name, p in train_model.named_parameters():
        names.append(name)
        dtype_names.append(str(p.dtype).split(".")[-1])
        shapes.append(list(p.shape))

    update_thread = threading.Thread(
        target=lambda: requests.post(
            f"{BASE_URL}/update_weights",
            json={"update_info": dict(
                names=names, dtype_names=dtype_names, shapes=shapes, packed=True,
            )},
            timeout=300,
        ).raise_for_status(),
    )
    update_thread.start()

    print("Broadcasting weights via NCCL...")
    NCCLWeightTransferEngine.trainer_send_weights(
        iterator=train_model.named_parameters(),
        trainer_args=NCCLTrainerSendWeightsArgs(group=model_update_group, packed=True),
    )
    update_thread.join()
    print("Weights transferred")

    requests.post(f"{BASE_URL}/finish_weight_update", json={}, timeout=60).raise_for_status()
    requests.post(f"{BASE_URL}/resume", timeout=30).raise_for_status()
    sync_elapsed = time.perf_counter() - sync_t0
    print(f"vLLM resumed — weight sync took {sync_elapsed:.2f}s")

    print("\n" + "=" * 60)
    print("AFTER weight sync (real weights):")
    print("=" * 60)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        max_tokens=16,
    )
    print(f"  Answer: {resp.choices[0].message.content}")
    post_metrics = scrape_metrics()
    post_bench = run_benchmark("AFTER")

    print("\n" + "=" * 60)
    print("PROMETHEUS METRICS COMPARISON")
    print("=" * 60)
    all_keys = sorted(set(list(pre_metrics.keys()) + list(post_metrics.keys())))
    print(f"  {'Metric':<52} {'Before':>10} {'After':>10}")
    print(f"  {'-'*52} {'-'*10} {'-'*10}")
    for k in all_keys:
        pre_v = pre_metrics.get(k, 0)
        post_v = post_metrics.get(k, 0)
        print(f"  {k:<52} {pre_v:>10.4f} {post_v:>10.4f}")

    print("\n" + "=" * 60)
    print("LATENCY & THROUGHPUT COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Before':>12} {'After':>12} {'Delta':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    for field in ["avg_latency_s", "p50_latency_s", "p99_latency_s", "tokens_per_sec", "total_tokens"]:
        bv = pre_bench[field]
        av = post_bench[field]
        delta = av - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {field:<30} {str(bv):>12} {str(av):>12} {sign}{delta:>11.4f}")

    print(f"\n  Weight sync duration: {sync_elapsed:.2f}s")
    print("\n=== WEIGHT SYNC + METRICS COMPLETE ===")

    results = _json.dumps(dict(
        pre_metrics=pre_metrics, post_metrics=post_metrics,
        pre_bench=pre_bench, post_bench=post_bench,
        sync_duration_s=round(sync_elapsed, 2),
    ))
    print(f"\n__RESULTS_JSON__{results}")


if __name__ == "__main__":
    main()
