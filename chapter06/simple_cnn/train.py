"""训练与评估循环（6.2 简单 CNN）。

run_experiment 是顶层入口：给定 seed + AugmentConfig，在 CTMP train 上训练，
用 val 调整学习率，在训练完成后才在 test（内部）和 external_test（外部）
上评估，返回指标字典。

评估协议使用 3 份冻结的 train/val/test 划分；最终指标取均值和总体标准差。
"""
from __future__ import annotations

import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader

from chapter06._common import get_device
from chapter06._common.ctmp_loader import (
    CTMP_CLASSES,
    build_label_map,
    load_ctmp_segments,
)

from .augment import AugmentConfig
from .dataset import MelSegmentDataset
from .model import SimpleAudioCNN

N_CLASSES = len(CTMP_CLASSES)
LABEL_MAP = build_label_map()
CACHE_VERSION = 1
CACHE_POLICY = "train_val_schedule_fixed_swa_test_final_cpu_safe_v1"


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    log_every: int = 5
    label_smoothing: float = 0.1
    swa_start_frac: float = 0.75
    swa_lr: float = 1e-3


def _train_one_epoch(
    model, loader, criterion, optimizer, device,
    *, mixup_alpha: float = 0.0, rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    model.train()
    total, total_loss, total_correct = 0, 0.0, 0
    nan_batches = 0
    for x, y in loader:
        if not torch.isfinite(x).all():
            nan_batches += 1
            continue
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        if mixup_alpha > 0.0 and rng is not None and x.size(0) > 1:
            lam = float(rng.beta(mixup_alpha, mixup_alpha))
            # 将原样本权重映射到 [0.5, 1]；这是本章采用的 Mixup 变体。
            lam = max(lam, 1.0 - lam)
            perm = torch.randperm(x.size(0), device=device)
            x_mix = lam * x + (1.0 - lam) * x[perm]
            logits = model(x_mix)
            loss = lam * criterion(logits, y) + (1.0 - lam) * criterion(logits, y[perm])
        else:
            logits = model(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            nan_batches += 1
            continue
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total += bs
        total_loss += loss.item() * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
    if nan_batches > 0:
        print(f"  [warn] skipped {nan_batches} non-finite batches")
    n = max(1, total)
    return total_loss / n, total_correct / n


@torch.no_grad()
def _evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    losses = []
    all_y, all_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y_dev)
        losses.append(loss.item() * y.size(0))
        pred = logits.argmax(dim=1).cpu().numpy()
        all_pred.append(pred)
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


def run_experiment(
    seed: int,
    aug: AugmentConfig,
    cfg: Optional[TrainConfig] = None,
    *,
    tag: str = "baseline",
    cache_dir: Optional[Path] = None,
    cache_key: Optional[str] = None,
) -> dict:
    """完整跑一次：加载 CTMP 切片 → val 调度 → 最终测试。

    Args:
        seed: CTMP 冻结划分编号（0/1/2）；底层划分种子由 manifest 固定。
        aug: 增强配置。
        cfg: 训练超参。
        tag: 日志标签。

    Returns:
        dict 含 history、val_acc、test_acc、ext_acc、预测标签等。
    """
    cfg = cfg or TrainConfig()
    device = get_device()
    cache_file: Optional[Path] = None
    cache_metadata = {
        "cache_version": CACHE_VERSION,
        "cache_policy": CACHE_POLICY,
        "seed": seed,
        "device_type": device.type,
        "train_config": asdict(cfg),
        "augment_config": asdict(aug),
        "cache_key": cache_key,
    }
    if cache_dir is not None:
        safe_key = (cache_key or "run").replace(" ", "_").replace("/", "_")
        cache_file = Path(cache_dir) / f"{safe_key}_{device.type}_split{seed}.pkl"
        if cache_file.exists():
            try:
                with cache_file.open("rb") as f:
                    cached = pickle.load(f)
            except (OSError, EOFError, pickle.UnpicklingError):
                cached = None
            if isinstance(cached, dict) and all(
                cached.get(k) == v for k, v in cache_metadata.items()
            ):
                print(f"[{tag}] cache hit: {cache_file}")
                return cached
    print(f"[{tag}] split={seed}, device={device}, aug={aug.enabled}")

    # ---- 数据 ----
    train_segs = load_ctmp_segments(seed=seed, split="train")
    val_segs = load_ctmp_segments(seed=seed, split="val")
    test_segs = load_ctmp_segments(seed=seed, split="test")
    ext_segs = load_ctmp_segments(seed=seed, split="external_test")

    train_ds = MelSegmentDataset(train_segs, LABEL_MAP, mode="train", aug=aug, seed=cfg.seed)
    val_ds = MelSegmentDataset(val_segs, LABEL_MAP, mode="eval", seed=cfg.seed)
    test_ds = MelSegmentDataset(test_segs, LABEL_MAP, mode="eval", seed=cfg.seed)
    ext_ds = MelSegmentDataset(ext_segs, LABEL_MAP, mode="eval", seed=cfg.seed)

    print(
        f"  train={len(train_ds)}, val={len(val_ds)}, "
        f"test={len(test_ds)}, external_test={len(ext_ds)}"
    )

    # 类别权重（与 6.1b 一致）
    y_train = np.array([LABEL_MAP[s["family_label"]] for s in train_segs])
    classes = np.arange(N_CLASSES)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_t = torch.tensor(cw, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )
    ext_loader = DataLoader(
        ext_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )

    # ---- 模型 ----
    torch.manual_seed(cfg.seed)
    model = SimpleAudioCNN(n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight_t, label_smoothing=cfg.label_smoothing)

    # SWA：从 swa_start_frac * epochs 开始，对后段权重做等权重平均
    swa_start = max(1, int(cfg.epochs * cfg.swa_start_frac))
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=cfg.swa_lr, anneal_epochs=3, anneal_strategy="linear")

    # ---- 训练 ----
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }
    t0 = time.perf_counter()
    mixup_alpha = float(aug.mixup_alpha) if aug.enabled else 0.0
    mixup_rng = np.random.default_rng(cfg.seed + 7919) if mixup_alpha > 0.0 else None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=mixup_alpha, rng=mixup_rng,
        )
        va = _evaluate(model, val_loader, criterion, device)
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step(va["loss"])
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["val_f1"].append(va["f1"])
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            phase = "SWA" if epoch >= swa_start else "warm"
            print(f"  epoch {epoch:02d}/{cfg.epochs} [{phase}]  "
                  f"train_loss={tr_loss:.3f}  train_acc={tr_acc:.3f}  "
                  f"val_loss={va['loss']:.3f}  val_acc={va['acc']:.3f}  val_f1={va['f1']:.3f}")

    elapsed = time.perf_counter() - t0
    # SimpleAudioCNN 使用 GroupNorm，不维护 BatchNorm running statistics。
    final_model = swa_model
    print(f"[{tag}] training done in {elapsed:.1f}s, SWA averaged epochs {swa_start}..{cfg.epochs}")

    # ---- 最终评估（用 SWA 模型；test/external_test 此前未参与训练决策）----
    val_eval = _evaluate(final_model, val_loader, criterion, device)
    test_eval = _evaluate(final_model, test_loader, criterion, device)
    ext_eval = _evaluate(final_model, ext_loader, criterion, device)

    result = {
        "seed": seed,
        "tag": tag,
        **cache_metadata,
        "history": history,
        "val_acc": val_eval["acc"],
        "val_f1": val_eval["f1"],
        "test_acc": test_eval["acc"],
        "test_f1": test_eval["f1"],
        "ext_acc": ext_eval["acc"],
        "ext_f1": ext_eval["f1"],
        "pred_test": test_eval["y_pred"],
        "pred_ext": ext_eval["y_pred"],
        "y_test": test_eval["y_true"],
        "y_ext": ext_eval["y_true"],
        "elapsed_sec": elapsed,
    }
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
        with tmp_file.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_file.replace(cache_file)
        print(f"[{tag}] cache written: {cache_file}")
    return result
