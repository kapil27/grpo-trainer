#!/usr/bin/env python3
"""Create state JSON for --sync-only (no trl/datasets import chain)."""
import json
import re
import sys
import time

import requests
from openai import OpenAI

BASE_URL = "http://localhost:8000"
METRICS_URL = f"{BASE_URL}/metrics"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_COMPLETION_LEN = 128

EVAL_QUESTIONS = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "ref_num": "18",
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "ref_num": "3",
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "ref_num": "70000",
    },
    {
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "ref_num": "540",
    },
    {
        "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her chickens 15 cups of feed. In the afternoon, she gives her chickens 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
        "ref_num": "20",
    },
]


def prompt_for(q: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "Solve the math problem. End with #### followed by the numeric answer.",
        },
        {"role": "user", "content": q["question"]},
    ]


def extract_answer_num(text: str) -> str | None:
    m = re.search(r"####\s*([\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"\b(\d[\d,]*)\b", text)
    return nums[-1].replace(",", "") if nums else None


def scrape_metrics() -> dict:
    raw = requests.get(METRICS_URL, timeout=15).text
    out: dict = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        for key in (
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:avg_generation_throughput_toks_per_s",
        ):
            if line.startswith(key + " ") or line.startswith(key + "{"):
                try:
                    out[key] = float(line.split()[-1])
                except ValueError:
                    pass
    return out


def run_vllm_eval(client: OpenAI, questions: list[dict], label: str) -> dict:
    correct = 0
    results = []
    latencies = []
    eval_questions = []
    for i, q in enumerate(questions):
        eval_questions.append(
            {"question": q["question"], "ref_num": q["ref_num"], "prompt": prompt_for(q)}
        )
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=prompt_for(q),
            max_tokens=MAX_COMPLETION_LEN,
        )
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        content = r.choices[0].message.content or ""
        pred_num = extract_answer_num(content)
        ok = bool(q["ref_num"] and pred_num and q["ref_num"] == pred_num)
        if ok:
            correct += 1
        results.append({"i": i, "ref": q["ref_num"], "pred": pred_num, "correct": ok})
    acc = correct / len(questions) if questions else 0.0
    return {
        "label": label,
        "n": len(questions),
        "correct": correct,
        "accuracy": round(acc, 4),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "results": results,
        "_eval_questions": eval_questions,
    }


def main():
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    train_elapsed = float(sys.argv[2]) if len(sys.argv) > 2 else 374.0
    r0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    r1 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    out_path = sys.argv[5] if len(sys.argv) > 5 else "/tmp/grpo-sync-state.json"

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="dummy")
    pre_metrics = scrape_metrics()
    pre_eval = run_vllm_eval(client, EVAL_QUESTIONS, "BEFORE_TRAINING")
    eval_questions = pre_eval.pop("_eval_questions")

    state = {
        "pre_eval": pre_eval,
        "pre_metrics": pre_metrics,
        "train_elapsed_s": train_elapsed,
        "reward_mean_first": r0,
        "reward_mean_last": r1,
        "eval_questions": eval_questions,
        "world_size": world_size,
    }
    with open(out_path, "w") as f:
        json.dump(state, f)
    print(f"Wrote {out_path} (pre_eval accuracy={pre_eval['accuracy']:.1%})")


if __name__ == "__main__":
    main()
