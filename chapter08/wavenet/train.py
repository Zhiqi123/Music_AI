"""Train a small WaveNet on NSynth or synthetic waveform windows."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from _common.checkpointing import save_torch_checkpoint
from _common.config import load_yaml_config
from _common.device_utils import choose_device
from _common.tables import write_rows
from wavenet.dataset import build_wavenet_dataset
from wavenet.model import build_wavenet_from_config


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_wavenet(config: dict[str, Any]) -> list[dict[str, float | int | str]]:
    """Run a compact WaveNet training loop and return history rows."""
    set_seed(int(config.get("seed", 0)))
    device = torch.device(choose_device(str(config.get("device", "auto"))))
    dataset = build_wavenet_dataset(config)
    training_cfg = config.get("training", {})
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=True,
        drop_last=False,
    )
    model = build_wavenet_from_config(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("learning_rate", 1e-3)))
    history: list[dict[str, float | int | str]] = []
    max_steps = training_cfg.get("max_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None

    for epoch in range(1, int(training_cfg.get("epochs", 1)) + 1):
        model.train()
        total_loss = 0.0
        step_count = 0
        for step, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            step_count += 1
            if max_steps is not None and step >= max_steps:
                break
        mean_loss = total_loss / max(step_count, 1)
        history.append({"epoch": epoch, "train_loss": mean_loss, "device": str(device)})

    output_cfg = config.get("outputs", {})
    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "outputs/checkpoints/wavenet_toy"))
    save_torch_checkpoint(
        checkpoint_dir / "last.pt",
        {"model": model.state_dict(), "config": config, "history": history},
    )
    history_csv = output_cfg.get("history_csv")
    if history_csv:
        write_rows(history_csv, history)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wavenet_toy.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    history = train_wavenet(config)
    print(history[-1] if history else "no training steps")


if __name__ == "__main__":
    main()

