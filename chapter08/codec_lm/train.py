"""Train the mini Codec-LM on precomputed or synthetic token sequences."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from _common.checkpointing import save_torch_checkpoint
from _common.config import load_yaml_config
from _common.device_utils import choose_device
from _common.tables import write_rows
from codec_lm.model import MiniCodecLM
from codec_lm.token_dataset import RandomTokenDataset, TokenFileDataset, find_token_files


def train_codec_lm(config: dict[str, Any]) -> list[dict[str, float | int | str]]:
    """Run a compact next-token training loop."""
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    device = torch.device(choose_device(str(config.get("device", "auto"))))
    dataset = build_codec_lm_dataset(config)
    loader = DataLoader(dataset, batch_size=int(training_cfg.get("batch_size", 8)), shuffle=True)
    model = MiniCodecLM(
        vocab_size=int(model_cfg.get("vocab_size", 1024)),
        d_model=int(model_cfg.get("d_model", 128)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        max_length=int(model_cfg.get("max_length", 128)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get("learning_rate", 3e-4)))
    history: list[dict[str, float | int | str]] = []
    max_steps = training_cfg.get("max_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None

    for epoch in range(1, int(training_cfg.get("epochs", 1)) + 1):
        total_loss = 0.0
        steps = 0
        model.train()
        for step, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
            if max_steps is not None and step >= max_steps:
                break
        mean_loss = total_loss / max(steps, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "perplexity": float(torch.exp(torch.tensor(mean_loss)).item()),
                "device": str(device),
            }
        )

    output_cfg = config.get("outputs", {})
    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "outputs/checkpoints/codec_lm_mini"))
    save_torch_checkpoint(
        checkpoint_dir / "last.pt",
        {"model": model.state_dict(), "config": config, "history": history},
    )
    history_csv = output_cfg.get("history_csv")
    if history_csv:
        write_rows(history_csv, history)
    return history


def build_codec_lm_dataset(config: dict[str, Any]):
    """Build the configured token dataset without silently falling back."""
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    source = data_cfg.get("source", "random")
    sequence_length = int(model_cfg.get("max_length", 128))
    vocab_size = int(model_cfg.get("vocab_size", 1024))
    if source == "random":
        return RandomTokenDataset(
            num_examples=int(data_cfg.get("num_examples", 128)),
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            seed=int(config.get("seed", 0)),
        )
    if source in {"token_cache", "fma_small"}:
        cache_dir = data_cfg.get("token_cache_dir", "outputs/generated/codec_tokens")
        token_files = find_token_files(cache_dir)
        if not token_files:
            raise FileNotFoundError(
                f"No cached codec token files found in {cache_dir}. "
                "Build EnCodec tokens with codec.build_token_cache or teaching tokens with codec.teaching_token_cache first."
            )
        max_files = data_cfg.get("max_files")
        if max_files is not None:
            token_files = token_files[: int(max_files)]
        return TokenFileDataset(
            token_files,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
        )
    raise ValueError(f"Unknown Codec-LM data source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/codec_lm_mini.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    history = train_codec_lm(config)
    print(history[-1] if history else "no training steps")


if __name__ == "__main__":
    main()
