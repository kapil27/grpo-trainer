#!/usr/bin/env python3
"""Submit multi-node GRPO TrainJob with shared checkpoint PVC via TrainingRuntime."""
from kubeflow.trainer import CustomTrainer, TrainerClient

from grpo_kft_train_fn import grpo_train

CHECKPOINT_RUNTIME = "grpo-torch-checkpoint"


def main():
    client = TrainerClient()
    job_name = client.train(
        runtime=CHECKPOINT_RUNTIME,
        trainer=CustomTrainer(
            func=grpo_train,
            num_nodes=2,
            resources_per_node={
                "cpu": 8,
                "memory": "48Gi",
                "gpu": 2,
            },
            packages_to_install=["trl", "datasets", "accelerate"],
        ),
    )
    print(job_name)


if __name__ == "__main__":
    main()
