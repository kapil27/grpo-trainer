"""GRPO training function for KFT CustomTrainer."""

def grpo_train():
    """TRL GRPO on GSM8K inside KFT v2 — benchmarks, validation (test split), JSON summary."""
    import json
    import os
    import re
    import time
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # --- Tunables (small on CPU, larger on GPU) ---
    USE_GPU = torch.cuda.is_available()
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)

    TRAIN_N = 256 if USE_GPU else 32
    EVAL_N = 64 if USE_GPU else 16
    NUM_GENERATIONS = 4 if USE_GPU else 2
    MAX_PROMPT_LEN = 256 if USE_GPU else 128
    MAX_COMPLETION_LEN = 128 if USE_GPU else 32

    # TRL GRPO: global batch (per_device * WORLD_SIZE * grad_accum) must be divisible by num_generations
    def _pick_grpo_microbatch(ws: int, use_gpu: bool, g: int) -> tuple[int, int]:
        if not use_gpu:
            return 2, 1  # global 2 * ws; works for ws=1,2 with G=2
        candidates = [(2, 2), (2, 1), (1, 4), (1, 2), (4, 1), (1, 1)]
        for pd, ga in candidates:
            glob = pd * max(1, ws) * ga
            if glob > 0 and glob % g == 0:
                return pd, ga
        raise RuntimeError(
            f"No (per_device, grad_accum) in default table works for WORLD_SIZE={ws}, G={g}"
        )

    PER_DEVICE_BS, GRAD_ACCUM = _pick_grpo_microbatch(WORLD_SIZE, USE_GPU, NUM_GENERATIONS)

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    benchmarks = {}
    summary = {
        "model": MODEL_NAME,
        "rank": RANK,
        "local_rank": LOCAL_RANK,
        "world_size": WORLD_SIZE,
        "cuda": USE_GPU,
        "multi_gpu_ddp": bool(USE_GPU and WORLD_SIZE > 1),
        "train_n": TRAIN_N,
        "eval_n": EVAL_N,
        "num_generations": NUM_GENERATIONS,
        "per_device_train_batch_size": PER_DEVICE_BS,
        "gradient_accumulation_steps": GRAD_ACCUM,
    }

    def banner(title):
        if RANK == 0:
            print(f"\n{'='*64}\n  {title}\n{'='*64}")

    def log(msg):
        if RANK == 0:
            print(f"[GRPO] {msg}")

    banner("ENVIRONMENT")
    log(f"RANK={RANK} LOCAL_RANK={LOCAL_RANK} WORLD_SIZE={WORLD_SIZE}")
    log(f"CUDA available: {USE_GPU}")
    _gb = PER_DEVICE_BS * max(1, WORLD_SIZE) * GRAD_ACCUM
    log(
        f"GRPO batch: per_device={PER_DEVICE_BS} * world={WORLD_SIZE} * grad_accum={GRAD_ACCUM} "
        f"= global {_gb} (must be divisible by G={NUM_GENERATIONS})"
    )
    if USE_GPU and WORLD_SIZE > 1:
        log("Multi-GPU DDP: expect one torch process per GPU on this node (torchrun / Trainer).")
    if USE_GPU:
        log(f"torch.cuda.device_count() in this process/pod view: {torch.cuda.device_count()}")
        log(f"Current CUDA device index: {torch.cuda.current_device()}")
        log(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        log(f"GPU memory (this device): {props.total_memory / 1e9:.2f} GB")
    else:
        log("CPU mode — shrink TRAIN_N/EVAL_N in script for faster runs")

    # --- Model ---
    t0 = time.perf_counter()
    if USE_GPU and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif USE_GPU:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    if USE_GPU:
        model = model.to(torch.device(f"cuda:{LOCAL_RANK}"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if USE_GPU:
        torch.cuda.synchronize()
    benchmarks["model_load_s"] = time.perf_counter() - t0

    log(
        f"Model loaded: {model.config.num_hidden_layers} layers, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, dtype={dtype}"
    )

    # --- Data (Hub download + cache inside pod) ---
    t0 = time.perf_counter()

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

    raw_eval = load_dataset("openai/gsm8k", "main", split="test")
    raw_eval = raw_eval.select(range(min(EVAL_N, len(raw_eval))))
    eval_ds = raw_eval.map(format_prompt)

    if USE_GPU:
        torch.cuda.synchronize()
    benchmarks["dataset_prep_s"] = time.perf_counter() - t0
    log(f"Train samples: {len(train_ds)} (from gsm8k train)")
    log(f"Eval samples:  {len(eval_ds)} (from gsm8k test)")

    # --- Reward ---
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

    use_bf16 = USE_GPU and torch.cuda.is_bf16_supported()

    config = GRPOConfig(
        output_dir="/tmp/grpo-spike",
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LEN,
        learning_rate=5e-6,
        beta=0.0,
        bf16=use_bf16,
        fp16=bool(USE_GPU and not use_bf16),
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        ddp_find_unused_parameters=False,
        eval_strategy="no",
        do_eval=True,
    )

    banner("CONFIG")
    log(
        f"G={config.num_generations} batch={config.per_device_train_batch_size} "
        f"grad_accum={config.gradient_accumulation_steps} lr={config.learning_rate} "
        f"bf16={use_bf16}"
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=gsm8k_reward,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # --- Train benchmark ---
    banner("TRAINING")
    t0 = time.perf_counter()
    train_out = trainer.train()
    if USE_GPU:
        torch.cuda.synchronize()
    benchmarks["train_wall_s"] = time.perf_counter() - t0
    train_metrics = getattr(train_out, "metrics", {}) or {}
    summary["train_runtime_s"] = float(
        train_metrics.get("train_runtime", benchmarks["train_wall_s"])
    )

    CHECKPOINT_PATH = "/mnt/checkpoint/grpo-trained"
    if RANK == 0:
        import shutil
        from pathlib import Path

        ckpt_dir = Path(CHECKPOINT_PATH)
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(CHECKPOINT_PATH)
        tokenizer.save_pretrained(CHECKPOINT_PATH)
        (ckpt_dir / ".ready").write_text("ok\n")
        log(f"Saved checkpoint to {CHECKPOINT_PATH}")
    summary["checkpoint_path"] = CHECKPOINT_PATH if RANK == 0 else None

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # --- Validation benchmark ---
    banner("VALIDATION (GSM8K test subset)")
    eval_metrics = {}
    try:
        t0 = time.perf_counter()
        eval_metrics = trainer.evaluate(eval_dataset=eval_ds)
        if USE_GPU:
            torch.cuda.synchronize()
        benchmarks["eval_wall_s"] = time.perf_counter() - t0
        summary["eval_metrics"] = {
            k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))
        }
        if RANK == 0:
            log("Validation metrics (subset):")
            for k in sorted(eval_metrics.keys()):
                v = eval_metrics[k]
                if isinstance(v, (int, float)):
                    print(f"    {k}: {float(v):.6f}")
    except RuntimeError as e:
        log(f"Eval failed (known TRL multi-GPU reshape issue): {e}")
        summary["eval_error"] = str(e)

    # --- Train log stats (rank 0) ---
    logs = trainer.state.log_history
    reward_logs = [x for x in logs if "reward/mean" in x]
    loss_logs = [x.get("loss") for x in logs if x.get("loss") is not None]

    banner("TRAIN LOG METRICS")
    if RANK == 0:
        if reward_logs:
            r0 = reward_logs[0].get("reward/mean", 0)
            r1 = reward_logs[-1].get("reward/mean", 0)
            log(f"reward/mean first: {r0:.4f}  last: {r1:.4f}  delta: {r1 - r0:+.4f}")
            summary["train_reward_mean_first"] = float(r0)
            summary["train_reward_mean_last"] = float(r1)
        else:
            log("No reward/mean in log_history — check TRL version / logging.")
        if loss_logs:
            log(f"loss first: {loss_logs[0]:.4f}  last: {loss_logs[-1]:.4f}")
            summary["train_loss_first"] = float(loss_logs[0])
            summary["train_loss_last"] = float(loss_logs[-1])

    banner("BENCHMARKS (wall clock, seconds)")
    if RANK == 0:
        for k, v in sorted(benchmarks.items()):
            print(f"  {k}: {v:.3f}")
        if USE_GPU and benchmarks["train_wall_s"] > 0:
            approx_gen_groups = (
                len(train_ds)
                // max(1, WORLD_SIZE * PER_DEVICE_BS * GRAD_ACCUM)
                * int(getattr(config, "num_train_epochs", 1))
            )
            summary["approx_train_steps_hint"] = approx_gen_groups

    summary["benchmarks_s"] = benchmarks
    summary["status"] = "ok"

    banner("SPIKE_SUMMARY_JSON")
    if RANK == 0:
        line = json.dumps(summary, default=str)
        print(line)
        print("\n[GRPO] Done.")
