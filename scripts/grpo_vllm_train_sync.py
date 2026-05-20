#!/usr/bin/env python3
"""
GRPO multi-GPU training + NCCL weight sync to vLLM — run INSIDE the vLLM pod.

GPU 0: vLLM inference server (already running)
GPU 1-N: TRL GRPOTrainer with torchrun DDP

Modes:
  --train-only   : torchrun runs this; trains on GPUs 1-N, rank 0 saves checkpoint
  --sync-only    : plain python; loads checkpoint on GPU 1, NCCL syncs to vLLM on GPU 0
  (default)      : single-GPU train + auto sync (legacy)

Usage (multi-GPU, called by notebook):
  Phase A: CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 /tmp/grpo_vllm_train_sync.py --train-only
  Phase B: CUDA_VISIBLE_DEVICES=0,1 python3 -u /tmp/grpo_vllm_train_sync.py --sync-only
"""

from __future__ import annotations

import gc
import inspect
import json as _json
import os
import re
import threading
import time

import requests
import torch
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE_URL = "http://localhost:8000"
METRICS_URL = f"{BASE_URL}/metrics"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_N = 64
NUM_GENERATIONS = 4
PER_DEVICE_BS = 2
GRAD_ACCUM = 2
MAX_PROMPT_LEN = 256
MAX_COMPLETION_LEN = 128
EVAL_QUESTIONS_N = 10
CHECKPOINT_DIR = "/tmp/grpo-vllm-train"
STATE_PATH = "/tmp/grpo_phase1_state.json"

TRACKED_COUNTERS = [
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
]
TRACKED_GAUGES = [
    "vllm:avg_generation_throughput_toks_per_s",
    "vllm:avg_prompt_throughput_toks_per_s",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
]
TRACKED_HISTOGRAMS = [
    "vllm:e2e_request_latency_seconds",
    "vllm:time_to_first_token_seconds",
]


def scrape_metrics() -> dict:
    raw = requests.get(METRICS_URL, timeout=15).text
    out: dict = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        for key in TRACKED_COUNTERS + TRACKED_GAUGES:
            if line.startswith(key + " ") or line.startswith(key + "{"):
                try:
                    out[key] = float(line.split()[-1])
                except ValueError:
                    pass
        for key in TRACKED_HISTOGRAMS:
            if line.startswith(key) and "_bucket" not in line and "_count" in line:
                try:
                    out[f"{key}_count"] = float(line.split()[-1])
                except ValueError:
                    pass
            if line.startswith(key) and "_sum" in line:
                try:
                    out[f"{key}_sum"] = float(line.split()[-1])
                except ValueError:
                    pass
    return out


def gsm8k_reward(completions, answer, **kwargs):
    rewards = []
    for completion, ref_answer in zip(completions, answer):
        content = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        ref_match = re.search(r"####\s*([\d,]+)", ref_answer)
        ref_num = ref_match.group(1).replace(",", "") if ref_match else None
        pred_match = re.search(r"####\s*([\d,]+)", content)
        pred_num = pred_match.group(1).replace(",", "") if pred_match else None
        if ref_num and pred_num and ref_num == pred_num:
            rewards.append(1.0)
        elif pred_match:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def load_eval_questions(n: int = EVAL_QUESTIONS_N) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(n, len(ds))))
    items = []
    for ex in ds:
        ref_match = re.search(r"####\s*([\d,]+)", ex["answer"])
        ref_num = ref_match.group(1).replace(",", "") if ref_match else None
        items.append(
            {
                "question": ex["question"],
                "ref_num": ref_num,
                "prompt": [
                    {
                        "role": "system",
                        "content": "Solve the math problem. End with #### followed by the numeric answer.",
                    },
                    {"role": "user", "content": ex["question"]},
                ],
            }
        )
    return items


def extract_answer_num(text: str) -> str | None:
    m = re.search(r"####\s*([\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"\b(\d[\d,]*)\b", text)
    return nums[-1].replace(",", "") if nums else None


def run_vllm_eval(client: OpenAI, questions: list[dict], label: str) -> dict:
    correct = 0
    results = []
    latencies = []
    for i, q in enumerate(questions):
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=q["prompt"],
            max_tokens=MAX_COMPLETION_LEN,
        )
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        content = r.choices[0].message.content or ""
        pred_num = extract_answer_num(content)
        ok = bool(q["ref_num"] and pred_num and q["ref_num"] == pred_num)
        if ok:
            correct += 1
        results.append(
            {
                "i": i,
                "question": q["question"][:80],
                "ref": q["ref_num"],
                "pred": pred_num,
                "correct": ok,
                "latency_s": round(elapsed, 3),
                "snippet": content[:120].replace("\n", " "),
            }
        )
    acc = correct / len(questions) if questions else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    summary = {
        "label": label,
        "n": len(questions),
        "correct": correct,
        "accuracy": round(acc, 4),
        "avg_latency_s": round(avg_lat, 4),
        "results": results,
    }
    print(f"\n[{label}] GSM8K eval: {correct}/{len(questions)} correct ({acc:.1%}), avg latency {avg_lat:.3f}s")
    for row in results[:5]:
        mark = "OK" if row["correct"] else "MISS"
        print(f"  [{mark}] Q: {row['question']}... ref={row['ref']} pred={row['pred']}")
    if len(results) > 5:
        print(f"  ... ({len(results) - 5} more)")
    return summary


# ---------------------------------------------------------------------------
# NCCL weight sync (runs in --sync-only mode)
# ---------------------------------------------------------------------------

def sync_weights_to_vllm(train_model, trainer_device: str) -> float:
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerSendWeightsArgs,
        NCCLWeightTransferEngine,
    )
    from vllm.utils.network_utils import get_ip, get_open_port

    dev_idx = int(trainer_device.split(":")[1])
    torch.cuda.set_device(dev_idx)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(dev_idx)

    ws = requests.get(f"{BASE_URL}/get_world_size", timeout=10).json()["world_size"]
    world_size = ws + 1
    addr = get_ip()
    port = get_open_port()
    print(f"\nNCCL weight sync: master={addr}:{port} world_size={world_size} device=cuda:{dev_idx}")

    init_thread = threading.Thread(
        target=lambda: requests.post(
            f"{BASE_URL}/init_weight_transfer_engine",
            json={
                "init_info": dict(
                    master_address=addr,
                    master_port=port,
                    rank_offset=1,
                    world_size=world_size,
                )
            },
            timeout=180,
        ).raise_for_status(),
    )
    init_thread.start()

    t0 = time.perf_counter()
    group = NCCLWeightTransferEngine.trainer_init(
        dict(master_address=addr, master_port=port, world_size=world_size)
    )
    init_thread.join()
    print("NCCL group established")

    requests.post(f"{BASE_URL}/pause", timeout=30).raise_for_status()
    requests.post(
        f"{BASE_URL}/start_weight_update",
        json={"is_checkpoint_format": True},
        timeout=60,
    ).raise_for_status()

    names, dtypes, shapes = [], [], []
    for n, p in train_model.named_parameters():
        names.append(n)
        dtypes.append(str(p.dtype).split(".")[-1])
        shapes.append(list(p.shape))

    ut = threading.Thread(
        target=lambda: requests.post(
            f"{BASE_URL}/update_weights",
            json={
                "update_info": dict(
                    names=names, dtype_names=dtypes, shapes=shapes, packed=True
                )
            },
            timeout=300,
        ).raise_for_status(),
    )
    ut.start()
    print("Broadcasting trained weights via NCCL...")
    NCCLWeightTransferEngine.trainer_send_weights(
        iterator=train_model.named_parameters(),
        trainer_args=NCCLTrainerSendWeightsArgs(group=group, packed=True),
    )
    ut.join()

    requests.post(f"{BASE_URL}/finish_weight_update", json={}, timeout=60).raise_for_status()
    requests.post(f"{BASE_URL}/resume", timeout=30).raise_for_status()
    elapsed = time.perf_counter() - t0
    print(f"Weight sync complete in {elapsed:.2f}s")
    return elapsed


# ---------------------------------------------------------------------------
# --train-only: runs under torchrun with DDP
# ---------------------------------------------------------------------------

def run_train_only(checkpoint_dir: str, state_path: str) -> None:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    if not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    if rank == 0:
        print(f"Multi-GPU GRPO training: RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
        print(f"GPUs visible: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  cuda:{i} = {torch.cuda.get_device_name(i)}")

    if rank == 0:
        eval_questions = load_eval_questions()
        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="dummy")

        print("\n" + "=" * 60)
        print("PHASE 1: Baseline inference (base model weights in vLLM)")
        print("=" * 60)
        pre_metrics = scrape_metrics()
        pre_eval = run_vllm_eval(client, eval_questions, "BEFORE_TRAINING")

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"PHASE 2: GRPO training ({world_size} GPUs, DDP)")
        print("=" * 60)

    def format_prompt(example):
        return {
            "prompt": [
                {
                    "role": "system",
                    "content": "Solve the math problem. End with #### followed by the numeric answer.",
                },
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
        }

    raw_train = load_dataset("openai/gsm8k", "main", split="train")
    raw_train = raw_train.select(range(min(TRAIN_N, len(raw_train))))
    train_ds = raw_train.map(format_prompt)

    if rank == 0:
        print(f"Loading model: {MODEL_NAME} on cuda:{local_rank}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.to(f"cuda:{local_rank}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config_kwargs = dict(
        output_dir=checkpoint_dir,
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        learning_rate=5e-6,
        beta=0.0,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    if "max_prompt_length" in grpo_params:
        config_kwargs["max_prompt_length"] = MAX_PROMPT_LEN
        config_kwargs["max_completion_length"] = MAX_COMPLETION_LEN
    if "eval_strategy" in grpo_params:
        config_kwargs["eval_strategy"] = "no"
    config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=gsm8k_reward,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    train_t0 = time.perf_counter()
    train_out = trainer.train()
    train_elapsed = time.perf_counter() - train_t0
    train_metrics = getattr(train_out, "metrics", {}) or {}

    if rank == 0:
        logs = trainer.state.log_history
        reward_logs = [x for x in logs if "reward/mean" in x or "reward" in x]
        def _rv(entry):
            return entry.get("reward/mean", entry.get("reward", 0))
        r0 = _rv(reward_logs[0]) if reward_logs else 0
        r1 = _rv(reward_logs[-1]) if reward_logs else 0
        print(f"\nGRPO training done in {train_elapsed:.1f}s (WORLD_SIZE={world_size})")
        print(f"  reward/mean: {r0:.4f} -> {r1:.4f} (delta {r1 - r0:+.4f})")
        print(f"  train_runtime: {train_metrics.get('train_runtime', train_elapsed):.1f}s")

        trainer.save_model(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")

        state = dict(
            pre_eval=pre_eval,
            pre_metrics=pre_metrics,
            train_elapsed_s=round(train_elapsed, 2),
            reward_mean_first=float(r0),
            reward_mean_last=float(r1),
            eval_questions=eval_questions,
            world_size=world_size,
        )
        with open(state_path, "w") as f:
            _json.dump(state, f)
        print(f"Saved state to {state_path}")
        print("\n--- Training phase complete. Ready for weight sync. ---")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# --sync-only: fresh process, loads checkpoint, NCCL syncs to vLLM
# ---------------------------------------------------------------------------

def run_sync_only(checkpoint_dir: str, state_path: str) -> None:
    with open(state_path) as f:
        state = _json.load(f)

    pre_eval = state["pre_eval"]
    pre_metrics = state["pre_metrics"]
    train_elapsed = state["train_elapsed_s"]
    r0 = state["reward_mean_first"]
    r1 = state["reward_mean_last"]
    eval_questions = state["eval_questions"]
    train_ws = state.get("world_size", 1)

    inference_ws = requests.get(f"{BASE_URL}/get_world_size", timeout=10).json()["world_size"]
    trainer_device = f"cuda:{inference_ws}"
    torch.cuda.set_device(int(trainer_device.split(":")[1]))
    print(f"Sync phase on {trainer_device}, checkpoint={checkpoint_dir}")

    if not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=dtype)
    model.to(trainer_device)
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="dummy")

    print("\n" + "=" * 60)
    print("PHASE 3: Sync trained weights to vLLM")
    print("=" * 60)
    sync_elapsed = sync_weights_to_vllm(model, trainer_device)

    print("\n" + "=" * 60)
    print("PHASE 4: Post-training inference via vLLM")
    print("=" * 60)
    post_metrics = scrape_metrics()
    post_eval = run_vllm_eval(client, eval_questions, "AFTER_TRAINING")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Training GPUs:        {train_ws}")
    print(f"  Eval accuracy BEFORE: {pre_eval['accuracy']:.1%} ({pre_eval['correct']}/{pre_eval['n']})")
    print(f"  Eval accuracy AFTER:  {post_eval['accuracy']:.1%} ({post_eval['correct']}/{post_eval['n']})")
    print(f"  Accuracy delta:       {post_eval['accuracy'] - pre_eval['accuracy']:+.1%}")
    print(f"  GRPO train time:      {train_elapsed:.1f}s")
    print(f"  Weight sync time:     {sync_elapsed:.2f}s")

    payload = _json.dumps(
        dict(
            pre_eval=pre_eval,
            post_eval=post_eval,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            train_elapsed_s=train_elapsed,
            sync_elapsed_s=round(sync_elapsed, 2),
            reward_mean_first=r0,
            reward_mean_last=r1,
            train_world_size=train_ws,
        ),
        default=str,
    )
    print(f"\n__RESULTS_JSON__{payload}")
    print("\n=== GRPO TRAINING + WEIGHT SYNC COMPLETE ===")


# ---------------------------------------------------------------------------
# Legacy single-GPU mode (default, no flags)
# ---------------------------------------------------------------------------

def run_train_and_handoff(checkpoint_dir: str, state_path: str) -> None:
    inference_ws = requests.get(f"{BASE_URL}/get_world_size", timeout=10).json()["world_size"]
    trainer_device = f"cuda:{inference_ws}"
    torch.cuda.set_device(int(trainer_device.split(":")[1]))
    print(f"vLLM on GPU 0, trainer on {trainer_device}")

    if not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    eval_questions = load_eval_questions()
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="dummy")

    print("=" * 60)
    print("PHASE 1: Baseline inference (base model weights in vLLM)")
    print("=" * 60)
    pre_metrics = scrape_metrics()
    pre_eval = run_vllm_eval(client, eval_questions, "BEFORE_TRAINING")

    print("\n" + "=" * 60)
    print("PHASE 2: GRPO training on GPU 1")
    print("=" * 60)

    def format_prompt(example):
        return {
            "prompt": [
                {
                    "role": "system",
                    "content": "Solve the math problem. End with #### followed by the numeric answer.",
                },
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
        }

    raw_train = load_dataset("openai/gsm8k", "main", split="train")
    raw_train = raw_train.select(range(min(TRAIN_N, len(raw_train))))
    train_ds = raw_train.map(format_prompt)

    print(f"Loading model for GRPO on {trainer_device}: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.to(trainer_device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config_kwargs = dict(
        output_dir=checkpoint_dir,
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        learning_rate=5e-6,
        beta=0.0,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    if "max_prompt_length" in grpo_params:
        config_kwargs["max_prompt_length"] = MAX_PROMPT_LEN
        config_kwargs["max_completion_length"] = MAX_COMPLETION_LEN
    if "eval_strategy" in grpo_params:
        config_kwargs["eval_strategy"] = "no"
    config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=gsm8k_reward,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    train_t0 = time.perf_counter()
    train_out = trainer.train()
    train_elapsed = time.perf_counter() - train_t0
    train_metrics = getattr(train_out, "metrics", {}) or {}
    logs = trainer.state.log_history
    reward_logs = [x for x in logs if "reward/mean" in x or "reward" in x]
    def _rv(entry):
        return entry.get("reward/mean", entry.get("reward", 0))
    r0 = _rv(reward_logs[0]) if reward_logs else 0
    r1 = _rv(reward_logs[-1]) if reward_logs else 0
    print(f"GRPO training done in {train_elapsed:.1f}s")
    print(f"  reward/mean: {r0:.4f} -> {r1:.4f} (delta {r1 - r0:+.4f})")

    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Saved checkpoint to {checkpoint_dir}")

    state = dict(
        pre_eval=pre_eval,
        pre_metrics=pre_metrics,
        train_elapsed_s=round(train_elapsed, 2),
        reward_mean_first=float(r0),
        reward_mean_last=float(r1),
        eval_questions=eval_questions,
        world_size=1,
    )
    with open(state_path, "w") as f:
        _json.dump(state, f)

    import subprocess
    import sys

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    print("\nStarting fresh process for NCCL weight sync...")
    subprocess.check_call(
        [sys.executable, __file__, "--sync-only",
         "--checkpoint", checkpoint_dir, "--state", state_path],
        env=env,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true",
                        help="Run training only (for torchrun multi-GPU)")
    parser.add_argument("--sync-only", action="store_true",
                        help="Run sync only (loads checkpoint, NCCL syncs to vLLM)")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR)
    parser.add_argument("--state", default=STATE_PATH)
    args = parser.parse_args()

    if args.train_only:
        run_train_only(args.checkpoint, args.state)
    elif args.sync_only:
        run_sync_only(args.checkpoint, args.state)
    else:
        run_train_and_handoff(args.checkpoint, args.state)
