"""训练与评估循环（6.4 AST 三模式）。

run_ast_experiment 是顶层入口：给定冻结划分编号 + 迁移模式，返回指标字典。
传入 cache_dir 时使用结果级缓存：目标 pkl 存在且训练协议/超参匹配时跳过训练。
注意这不是 epoch 级 checkpoint；训练中途被打断时，未完成的 run 会重跑。
"""
from __future__ import annotations

import pickle
import re
import time
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from transformers import __version__ as transformers_version
from transformers import get_linear_schedule_with_warmup

from chapter06._common import get_device
from chapter06._common.ctmp_loader import (
    CTMP_CLASSES, build_label_map, load_ctmp_segments,
)
from chapter06._common.device_utils import auto_batch_size

from .ast_model import AST_HIDDEN, AST_LOCAL_DIR, TransferAST
from .dataset import ASTSegmentDataset
from .transfer import (
    apply_lora, fine_tune_all, freeze_backbone, count_trainable,
)

N_CLASSES = len(CTMP_CLASSES)
LABEL_MAP = build_label_map()
TRANSFER_MODES = ("feature_extraction", "lora", "fine_tune")
CACHE_VERSION = 5
CACHE_POLICY = "ast_two_token_pool_val_select_partial_accum_device_bound_v5"
FEATURE_EXTRACTION_STRATEGY = "frozen_two_token_pooled_embedding_standardized_cpu_head_v3"
SELECTION_SPLIT = "val"


@dataclass
class ASTTrainConfig:
    epochs: int = 10
    batch_size: Optional[int] = None  # None → auto_batch_size
    head_lr: float = 1e-3            # FE
    lora_lr: float = 1e-4            # LoRA + head
    full_ft_lr: float = 1e-5         # FullFT 统一 lr
    weight_decay: float = 1e-4
    num_workers: int = 0
    log_every: int = 5
    label_smoothing: float = 0.1
    warmup_ratio: float = 0.1        # 前 10% steps 线性从 0 升到目标 lr
    grad_clip: float = 1.0           # 本节固定的梯度总范数上限
    grad_accum_steps: int = 1        # micro-batch 梯度累积步数；1 等价于不累积
    early_stop_patience: Optional[int] = None  # None → 跑满 epochs
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1


MAIN_CACHE_CONFIG = ASTTrainConfig(epochs=20, batch_size=32)


def _safe_cache_key(key: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.strip())
    return key.strip("._-")


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "none"
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _config_slug(mode: str, cfg: ASTTrainConfig) -> str:
    parts = [
        f"e{cfg.epochs}",
        f"bs{cfg.batch_size}",
        f"ga{cfg.grad_accum_steps}",
        f"wd{_fmt_float(cfg.weight_decay)}",
        f"ls{_fmt_float(cfg.label_smoothing)}",
        f"wr{_fmt_float(cfg.warmup_ratio)}",
        f"gc{_fmt_float(cfg.grad_clip)}",
        f"pat{cfg.early_stop_patience}",
    ]
    if mode == "feature_extraction":
        parts.append(f"hlr{_fmt_float(cfg.head_lr)}")
    elif mode == "lora":
        parts.extend([
            f"lr{_fmt_float(cfg.lora_lr)}",
            f"r{cfg.lora_r}",
            f"a{cfg.lora_alpha}",
            f"d{_fmt_float(cfg.lora_dropout)}",
        ])
    elif mode == "fine_tune":
        parts.append(f"lr{_fmt_float(cfg.full_ft_lr)}")
    return "_".join(parts)


def _experiment_key(mode: str, cfg: ASTTrainConfig,
                    cache_key: Optional[str] = None) -> str:
    if cache_key is not None:
        return cache_key
    if asdict(cfg) == asdict(MAIN_CACHE_CONFIG):
        return mode
    return f"{mode}_{_config_slug(mode, cfg)}"


def _cache_name(mode: str, seed: int, cfg: ASTTrainConfig,
                cache_key: Optional[str] = None, device_type: str = "cpu") -> str:
    key = _experiment_key(mode, cfg, cache_key)
    return f"{_safe_cache_key(key)}_{device_type}_seed{seed}.pkl"


def _normalized_train_config(config: dict) -> dict:
    """Backfill fields added after early cache files were produced."""
    normalized = dict(config)
    normalized.setdefault("grad_accum_steps", 1)
    return normalized


def _train_config_matches(cached: dict, cfg: ASTTrainConfig) -> bool:
    cached_cfg = cached.get("train_config")
    if not isinstance(cached_cfg, dict):
        return False
    return _normalized_train_config(cached_cfg) == asdict(cfg)


def _cache_matches(
    cached: dict,
    *,
    seed: int,
    mode: str,
    cfg: ASTTrainConfig,
    cache_key: Optional[str],
    device_type: str,
    checkpoint_size: int,
) -> bool:
    if cached.get("seed") != seed or cached.get("mode") != mode:
        return False
    if cached.get("cache_version") != CACHE_VERSION:
        return False
    if cached.get("cache_policy") != CACHE_POLICY:
        return False
    if cached.get("selection_split") != SELECTION_SPLIT:
        return False
    if cached.get("device_type") != device_type:
        return False
    if cached.get("torch_version") != torch.__version__:
        return False
    if cached.get("transformers_version") != transformers_version:
        return False
    if cached.get("checkpoint_size") != checkpoint_size:
        return False
    if not _train_config_matches(cached, cfg):
        return False
    if cached.get("experiment_key") != _experiment_key(mode, cfg, cache_key):
        return False
    if mode == "feature_extraction":
        return cached.get("feature_extraction_strategy") == FEATURE_EXTRACTION_STRATEGY
    return True
def _maybe_empty_device_cache(device) -> None:
    if str(device) == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _optimizer_step(model, optimizer, scheduler, grad_clip) -> None:
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), grad_clip,
        )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _train_one_epoch(
    model, loader, criterion, optimizer, scheduler, device, grad_clip,
    *, grad_accum_steps: int = 1, mode=None,
):
    model.train()
    # Feature Extraction 现在走 frozen embedding 专用路径；这里保留保护，
    # 防止未来直接复用通用训练循环时把冻结 backbone 切回 train。
    if mode == "feature_extraction":
        model.base.eval()
    accum_steps = max(1, int(grad_accum_steps))
    total, total_loss, total_correct = 0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    finite_in_group = 0
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        group_start = (batch_idx // accum_steps) * accum_steps
        group_size = min(accum_steps, n_batches - group_start)
        if torch.isfinite(loss):
            (loss / group_size).backward()
            finite_in_group += 1
            bs = y.size(0)
            total += bs
            total_loss += loss.item() * bs
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
        if (batch_idx + 1) % accum_steps == 0 or batch_idx + 1 == n_batches:
            if finite_in_group > 0:
                _optimizer_step(model, optimizer, scheduler, grad_clip)
            else:
                optimizer.zero_grad(set_to_none=True)
            finite_in_group = 0
    n = max(1, total)
    return total_loss / n, total_correct / n


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    losses, all_y, all_pred = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y_dev)
        losses.append(loss.item() * y.size(0))
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_y.append(y.numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    n = max(1, len(y_true))
    return {
        "loss": float(np.sum(losses) / n),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "y_true": y_true,
        "y_pred": y_pred,
    }


@torch.no_grad()
def _extract_pooled_embeddings(model, loader, device):
    model.base.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        outputs = model.base(input_values=x)
        feats.append(outputs.pooler_output.detach().cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def _train_head_one_epoch(head, loader, criterion, optimizer, scheduler, device, grad_clip):
    head.train()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(x), y)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()


@torch.no_grad()
def _evaluate_head(head, loader, criterion, device):
    head.eval()
    losses, all_y, all_pred = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        logits = head(x)
        loss = criterion(logits, y_dev)
        losses.append(loss.item() * y.size(0))
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_y.append(y.numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    n = max(1, len(y_true))
    return {
        "loss": float(np.sum(losses) / n),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _standardize_embedding_splits(train_x: torch.Tensor, *splits: torch.Tensor):
    train_x = torch.nan_to_num(train_x.float())
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    normalized = [(train_x - mean) / std]
    for x in splits:
        x = torch.nan_to_num(x.float())
        normalized.append((x - mean) / std)
    return normalized


def _build_optimizer(model, mode: str, cfg: ASTTrainConfig):
    # 本实验显式使用 Adam；其 weight_decay 是与梯度耦合的 L2 项，
    # 不与 AdamW 的解耦权重衰减混为同一算法。
    if mode == "feature_extraction":
        freeze_backbone(model)
        return torch.optim.Adam(
            model.fc_transfer.parameters(),
            lr=cfg.head_lr, weight_decay=cfg.weight_decay,
        )
    if mode == "lora":
        apply_lora(model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                   lora_dropout=cfg.lora_dropout)
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(
            trainable, lr=cfg.lora_lr, weight_decay=cfg.weight_decay,
        )
    if mode == "fine_tune":
        fine_tune_all(model)
        return torch.optim.Adam(
            model.parameters(), lr=cfg.full_ft_lr, weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"unknown mode: {mode}")


def _write_cache(cache_file: Optional[Path], result: dict, tag: str,
                 action: str = "cached") -> None:
    if cache_file is None:
        return
    # 先写临时文件再原子 rename，避免训练完写盘途中被中断留下半截 pickle
    tmp = cache_file.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(result, fh)
    tmp.replace(cache_file)
    print(f"[{tag}] {action} → {cache_file.name}")


def _load_cache(cache_file: Path, tag: str) -> dict:
    with cache_file.open("rb") as fh:
        cached = pickle.load(fh)
    val = cached.get("val_acc")
    val_text = f"val_acc={val:.3f} " if val is not None else ""
    print(f"[{tag}] cache hit → {cache_file.name} "
          f"({val_text}test_acc={cached['test_acc']:.3f} "
          f"ext_acc={cached['ext_acc']:.3f})")
    return cached


def _load_compatible_cache(
    cache_file: Path,
    *,
    seed: int,
    mode: str,
    cfg: ASTTrainConfig,
    cache_key: Optional[str],
    tag: str,
    device_type: str,
    checkpoint_size: int,
) -> Optional[dict]:
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("rb") as fh:
            cached = pickle.load(fh)
    except Exception as exc:
        print(f"[{tag}] ignore unreadable cache {cache_file.name}: {exc}")
        return None
    if _cache_matches(
        cached, seed=seed, mode=mode, cfg=cfg, cache_key=cache_key,
        device_type=device_type, checkpoint_size=checkpoint_size,
    ):
        return _load_cache(cache_file, tag)
    print(f"[{tag}] stale cache ignored → {cache_file.name}")
    return None
def _run_feature_extraction_from_embeddings(
    *,
    seed: int,
    mode: str,
    tag: str,
    cfg: ASTTrainConfig,
    device,
    train_loader,
    val_loader,
    test_loader,
    ext_loader,
    class_weight_t,
    experiment_key: str,
    checkpoint_size: int,
) -> dict:
    torch.manual_seed(42)
    model = TransferAST(n_classes=N_CLASSES)
    freeze_backbone(model)
    model = model.to(device)
    print(f"  trainable params: {count_trainable(model):,}")
    print(f"  feature extraction strategy: {FEATURE_EXTRACTION_STRATEGY}")

    t0 = time.perf_counter()
    print("  extracting frozen AST two-token pooled embeddings...")
    train_x, train_y = _extract_pooled_embeddings(model, train_loader, device)
    val_x, val_y = _extract_pooled_embeddings(model, val_loader, device)
    test_x, test_y = _extract_pooled_embeddings(model, test_loader, device)
    ext_x, ext_y = _extract_pooled_embeddings(model, ext_loader, device)
    print(f"  embeddings: train={tuple(train_x.shape)} "
          f"val={tuple(val_x.shape)} test={tuple(test_x.shape)} "
          f"ext={tuple(ext_x.shape)}")
    train_x, val_x, test_x, ext_x = _standardize_embedding_splits(
        train_x, val_x, test_x, ext_x,
    )

    train_feat_ds = TensorDataset(train_x, train_y)
    val_feat_ds = TensorDataset(val_x, val_y)
    test_feat_ds = TensorDataset(test_x, test_y)
    ext_feat_ds = TensorDataset(ext_x, ext_y)
    generator = torch.Generator().manual_seed(42)
    train_head_loader = DataLoader(
        train_feat_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator,
    )
    train_eval_loader = DataLoader(train_feat_ds, batch_size=cfg.batch_size, shuffle=False)
    val_feat_loader = DataLoader(val_feat_ds, batch_size=cfg.batch_size, shuffle=False)
    test_feat_loader = DataLoader(test_feat_ds, batch_size=cfg.batch_size, shuffle=False)
    ext_feat_loader = DataLoader(ext_feat_ds, batch_size=cfg.batch_size, shuffle=False)

    # The feature-extraction baseline only trains a tiny linear probe. Keep that
    # probe on CPU to avoid MPS numerical spikes observed with frozen embeddings.
    head_device = torch.device("cpu")
    head = model.fc_transfer.to(head_device)
    del model
    _maybe_empty_device_cache(device)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, len(train_head_loader) * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight_t.detach().cpu(),
                                    label_smoothing=cfg.label_smoothing)

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_f1": []}
    best_loss = float("inf")
    best_f1 = -float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        _train_head_one_epoch(
            head, train_head_loader, criterion, optimizer, scheduler, head_device,
            cfg.grad_clip,
        )
        tr = _evaluate_head(head, train_eval_loader, criterion, head_device)
        va = _evaluate_head(head, val_feat_loader, criterion, head_device)
        for key, val in [("train_loss", tr["loss"]), ("train_acc", tr["acc"]),
                         ("val_loss", va["loss"]), ("val_acc", va["acc"]),
                         ("val_f1", va["f1"])]:
            history[key].append(val)
        improves_f1 = va["f1"] > best_f1 + 1e-9
        ties_f1 = abs(va["f1"] - best_f1) <= 1e-9
        if improves_f1 or (ties_f1 and va["loss"] < best_loss):
            best_loss = va["loss"]
            best_f1 = va["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            print(f"  epoch {epoch:02d}/{cfg.epochs} "
                  f"train_loss={tr['loss']:.3f} train_acc={tr['acc']:.3f} "
                  f"val_loss={va['loss']:.3f} val_acc={va['acc']:.3f} "
                  f"val_f1={va['f1']:.3f}")
        if cfg.early_stop_patience is not None and bad_epochs >= cfg.early_stop_patience:
            print(f"  early stop at epoch {epoch:02d}; "
                  f"best_epoch={best_epoch} best_val_f1={best_f1:.3f}")
            break

    elapsed = time.perf_counter() - t0
    print(f"[{tag}] done in {elapsed:.1f}s; best_epoch={best_epoch}")

    if best_state is not None:
        head.load_state_dict(best_state)
    val_eval = _evaluate_head(head, val_feat_loader, criterion, head_device)
    test_eval = _evaluate_head(head, test_feat_loader, criterion, head_device)
    ext_eval = _evaluate_head(head, ext_feat_loader, criterion, head_device)

    return {
        "seed": seed, "mode": mode, "tag": tag,
        "cache_version": CACHE_VERSION, "cache_policy": CACHE_POLICY,
        "experiment_key": experiment_key,
        "selection_split": SELECTION_SPLIT,
        "selection_metric": "val_f1",
        "device_type": device.type,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "checkpoint_size": checkpoint_size,
        "train_config": asdict(cfg),
        "feature_extraction_strategy": FEATURE_EXTRACTION_STRATEGY,
        "history": history, "best_epoch": best_epoch, "best_val_loss": best_loss,
        "best_val_f1": best_f1,
        "val_acc": val_eval["acc"], "val_f1": val_eval["f1"],
        "test_acc": test_eval["acc"], "test_f1": test_eval["f1"],
        "ext_acc": ext_eval["acc"], "ext_f1": ext_eval["f1"],
        "y_val": val_eval["y_true"], "pred_val": val_eval["y_pred"],
        "y_test": test_eval["y_true"], "pred_test": test_eval["y_pred"],
        "y_ext": ext_eval["y_true"], "pred_ext": ext_eval["y_pred"],
        "elapsed_sec": elapsed,
    }


def run_ast_experiment(
    seed: int,
    mode: str,
    cfg: Optional[ASTTrainConfig] = None,
    *,
    tag: str = "baseline",
    cache_dir: Optional[Path] = None,
    cache_key: Optional[str] = None,
) -> dict:
    assert mode in TRANSFER_MODES, f"unknown mode: {mode}"
    cfg = cfg or ASTTrainConfig()
    if cfg.batch_size is None:
        cfg = replace(cfg, batch_size=auto_batch_size(model_params_m=87, base=8))
    if cfg.grad_accum_steps < 1:
        cfg = replace(cfg, grad_accum_steps=1)
    experiment_key = _experiment_key(mode, cfg, cache_key)
    device = get_device()
    checkpoint_path = Path(AST_LOCAL_DIR) / "model.safetensors"
    checkpoint_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else -1

    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / _cache_name(
            mode, seed, cfg, cache_key=cache_key, device_type=device.type,
        )
        cached = _load_compatible_cache(
            cache_file, seed=seed, mode=mode, cfg=cfg, cache_key=cache_key, tag=tag,
            device_type=device.type, checkpoint_size=checkpoint_size,
        )
        if cached is not None:
            return cached

    effective_bs = cfg.batch_size * cfg.grad_accum_steps
    print(f"[{tag}] split={seed} mode={mode} key={experiment_key} "
          f"device={device} bs={cfg.batch_size} "
          f"accum={cfg.grad_accum_steps} effective_bs={effective_bs}")

    train_segs = load_ctmp_segments(seed=seed, split="train")
    val_segs = load_ctmp_segments(seed=seed, split="val")
    test_segs = load_ctmp_segments(seed=seed, split="test")
    ext_segs = load_ctmp_segments(seed=seed, split="external_test")

    train_ds = ASTSegmentDataset(train_segs, LABEL_MAP, precompute=True)
    val_ds = ASTSegmentDataset(val_segs, LABEL_MAP, precompute=True)
    test_ds = ASTSegmentDataset(test_segs, LABEL_MAP, precompute=True)
    ext_ds = ASTSegmentDataset(ext_segs, LABEL_MAP, precompute=True)
    print(f"  train={len(train_ds)} val={len(val_ds)} "
          f"test={len(test_ds)} ext={len(ext_ds)}")

    y_train = np.array([LABEL_MAP[s["family_label"]] for s in train_segs])
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y_train)
    class_weight_t = torch.tensor(cw, dtype=torch.float32, device=device)

    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    ext_loader = DataLoader(ext_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)

    if mode == "feature_extraction":
        extract_train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers,
        )
        result = _run_feature_extraction_from_embeddings(
            seed=seed, mode=mode, tag=tag, cfg=cfg, device=device,
            train_loader=extract_train_loader, val_loader=val_loader,
            test_loader=test_loader, ext_loader=ext_loader,
            class_weight_t=class_weight_t, experiment_key=experiment_key,
            checkpoint_size=checkpoint_size,
        )
        _write_cache(cache_file, result, tag)
        return result

    torch.manual_seed(42)
    model = TransferAST(n_classes=N_CLASSES)
    optimizer = _build_optimizer(model, mode, cfg)
    model = model.to(device)
    print(f"  trainable params: {count_trainable(model):,}")

    steps_per_epoch = max(1, ceil(len(train_loader) / cfg.grad_accum_steps))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight_t,
                                    label_smoothing=cfg.label_smoothing)

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_f1": []}
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    t0 = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, cfg.grad_clip,
            grad_accum_steps=cfg.grad_accum_steps, mode=mode,
        )
        va = _evaluate(model, val_loader, criterion, device)
        _maybe_empty_device_cache(device)
        for key, val in [("train_loss", tr_loss), ("train_acc", tr_acc),
                         ("val_loss", va["loss"]), ("val_acc", va["acc"]),
                         ("val_f1", va["f1"])]:
            history[key].append(val)
        if va["loss"] < best_loss:
            best_loss = va["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            print(f"  epoch {epoch:02d}/{cfg.epochs} "
                  f"train_loss={tr_loss:.3f} val_loss={va['loss']:.3f} "
                  f"val_acc={va['acc']:.3f} val_f1={va['f1']:.3f}")
        if cfg.early_stop_patience is not None and bad_epochs >= cfg.early_stop_patience:
            print(f"  early stop at epoch {epoch:02d}; "
                  f"best_epoch={best_epoch} best_val_loss={best_loss:.3f}")
            break

    elapsed = time.perf_counter() - t0
    print(f"[{tag}] done in {elapsed:.1f}s; best_epoch={best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)
    val_eval = _evaluate(model, val_loader, criterion, device)
    test_eval = _evaluate(model, test_loader, criterion, device)
    ext_eval = _evaluate(model, ext_loader, criterion, device)

    result = {
        "seed": seed, "mode": mode, "tag": tag,
        "cache_version": CACHE_VERSION, "cache_policy": CACHE_POLICY,
        "experiment_key": experiment_key,
        "selection_split": SELECTION_SPLIT,
        "device_type": device.type,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "checkpoint_size": checkpoint_size,
        "train_config": asdict(cfg),
        "history": history, "best_epoch": best_epoch, "best_val_loss": best_loss,
        "val_acc": val_eval["acc"], "val_f1": val_eval["f1"],
        "test_acc": test_eval["acc"], "test_f1": test_eval["f1"],
        "ext_acc": ext_eval["acc"], "ext_f1": ext_eval["f1"],
        "y_val": val_eval["y_true"], "pred_val": val_eval["y_pred"],
        "y_test": test_eval["y_true"], "pred_test": test_eval["y_pred"],
        "y_ext": ext_eval["y_true"], "pred_ext": ext_eval["y_pred"],
        "elapsed_sec": elapsed,
    }

    _write_cache(cache_file, result, tag)

    return result
