# E2E Validation: Multi-Node KFT GRPO → Shared PVC → vLLM NCCL Sync

**Date:** 2026-05-26  
**Cluster:** sridhartest (`grpoxtrainer`)  
**TrainJob:** `n119d164d9f9`  
**Status:** PASS (architecture validated)

---

## Acceptance Criteria

| Criterion | Result | Evidence |
|-----------|--------|----------|
| RWX PVC mounted in KFT train pods + vLLM pod | **PASS** | `grpo-shared-checkpoint` on train pods via `ClusterTrainingRuntime/grpo-torch-checkpoint`; vLLM at `/mnt/checkpoint` |
| KFT 2×2 GRPO completes, checkpoint on PVC | **PASS** | `WORLD_SIZE=4`, TrainJob Complete, 958M at `/mnt/checkpoint/grpo-trained`, `.ready` sentinel |
| Sync loads from PVC, NCCL to GPU 0 | **PASS** | `--sync-only` from vLLM pod, sync 0.48s, `Weight sync complete` |
| vLLM updated predictions after sync | **PASS** | Before/after preds differ (e.g. Q0: `3→144`, Q4: `2→75`); latency dropped 1.2s→0.2s |
| Measured timings | **PASS** | See metrics below |
| Findings documented | **PASS** | This file |

---

## Architecture (final)

```
KFT TrainJob (grpo-torch-checkpoint runtime)
  node-0 rank 0 ──writes──► /mnt/checkpoint/grpo-trained
  node-1 rank 2,3 ──DDP──►  (same PVC mounted, read-only use)
                                    │
                                    ▼
                         grpo-shared-checkpoint (RWX nfs)
                                    │
                                    ▼
vLLM pod ──reads──► /mnt/checkpoint/grpo-trained
         ──sync──►  GPU1 loads → NCCL → GPU0 vLLM
```

**Fix applied:** Replaced fragile `oc cp` race with `ClusterTrainingRuntime` that mounts PVC on all train pods. Rank 0 writes directly to `/mnt/checkpoint/grpo-trained`.

Note: `RuntimePatch` via SDK is sent but **stripped by this cluster's TrainJob CRD** (no `runtimePatches` field). Custom runtime YAML is the correct approach on RHOAI 3.3.

---

## Metrics (Run `n119d164d9f9`)

| Metric | Value |
|--------|-------|
| Training wall time | ~374s |
| Checkpoint size | 958 MB |
| NCCL weight sync | 0.48s |
| End-to-end (train + sync) | ~375s |
| Training GPUs | 4 (2 nodes × 2) |
| Eval subset accuracy | 0/5 before, 0/5 after (preds changed) |

---

## Key Commands

```bash
oc apply -f k8s/grpo-shared-checkpoint-pvc.yaml
oc apply -f k8s/grpo-kft-checkpoint-runtime.yaml
# Submit via notebook or scripts/submit_kft_grpo_multinode.py

# After training:
python scripts/post_train_sync.py TRAINJOB_NAME

# Or from the notebook: Step 5 (runs the same script after Step 3 submit)
```

---

## Jira Ticket Verdict

**Ready to mark completed.** All acceptance criteria met. No commit/push/Jira update performed per user request.
