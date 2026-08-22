"""训练与评估循环（6.3 预训练 CNN 迁移学习）。

run_experiment 是顶层入口：给定 seed + 迁移模式，在 CTMP train 上训练，
用 val 调整学习率并选择轮次，最后在 test（内部）和 external_test（外部）
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
from torch.utils.data import DataLoader, TensorDataset

from chapter06._common import get_device
from chapter06._common.ctmp_loader import (
    CTMP_CLASSES,
    build_label_map,
    load_ctmp_segments,
)

from .dataset import PannsSegmentDataset
from .model import TransferCnn14, download_cnn14_weights
from .transfer import freeze_backbone, fine_tune_all, layer_wise_lr

N_CLASSES = len(CTMP_CLASSES)
LABEL_MAP = build_label_map()
CACHE_VERSION = 2
CACHE_POLICY = "cnn14_block6_no_pool_val_select_frozen_embedding_cpu_head_v2"
FEATURE_EXTRACTION_STRATEGY = "frozen_eval_embedding_cpu_head"
FEATURE_EXTRACTION_SELECTION = "val_macro_f1_then_val_loss_v3"


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    num_workers: int = 0
    log_every: int = 5
    label_smoothing: float = 0.1
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5


TRANSFER_MODES = ("feature_extraction", "fine_tune", "layer_wise")


def _train_one_epoch(
    model, loader, criterion, optimizer, device,
) -> tuple[float, float]:
    model.train()
    total, total_loss, total_correct = 0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        logits = out["clipwise_output"]
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total += bs
        total_loss += loss.item() * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
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
        out = model(x)
        logits = out["clipwise_output"]
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


@torch.no_grad()
def _extract_embeddings(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.base.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats.append(model.base.forward_features(x).detach().cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def _train_head_one_epoch(head, loader, criterion, optimizer) -> tuple[float, float]:
    head.train()
    total, total_loss, total_correct = 0, 0.0, 0
    for x, y in loader:
        optimizer.zero_grad()
        logits = head(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total += bs
        total_loss += loss.item() * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
    n = max(1, total)
    return total_loss / n, total_correct / n


@torch.no_grad()
def _evaluate_head(head, loader, criterion) -> dict:
    head.eval()
    losses, all_y, all_pred = [], [], []
    for x, y in loader:
        logits = head(x)
        loss = criterion(logits, y)
        losses.append(loss.item() * y.size(0))
        all_pred.append(logits.argmax(dim=1).numpy())
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


def _write_cache(cache_file: Optional[Path], result: dict, tag: str) -> None:
    if cache_file is None:
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with tmp_file.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_file.replace(cache_file)
    print(f"[{tag}] cache written: {cache_file}")


def _is_better_feature_epoch(
    val_f1: float,
    val_loss: float,
    best_f1: float,
    best_loss: float,
) -> bool:
    improves_f1 = val_f1 > best_f1 + 1e-9
    ties_f1 = abs(val_f1 - best_f1) <= 1e-9
    return improves_f1 or (ties_f1 and val_loss < best_loss)


def run_experiment(
    seed: int,
    mode: str,
    cfg: Optional[TrainConfig] = None,
    *,
    tag: str = "baseline",
    ckpt_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> dict:
    """完整跑一次：加载 CTMP 切片 → val 选择 → 最终测试。

    Args:
        seed: CTMP 冻结划分编号（0/1/2）；底层划分种子由 manifest 固定。
        mode: 'feature_extraction' | 'fine_tune' | 'layer_wise'。
        cfg: 训练超参。
        tag: 日志标签。
        ckpt_path: CNN14 预训练权重路径；None 则自动下载。

    Returns:
        dict 含 history、test_acc、test_f1、ext_acc、ext_f1 等。
    """
    assert mode in TRANSFER_MODES, f"未知模式：{mode}（合法：{TRANSFER_MODES}）"
    cfg = cfg or TrainConfig()
    device = get_device()

    # ---- 权重 ----
    if ckpt_path is None:
        ckpt_path = download_cnn14_weights()
    ckpt_path = Path(ckpt_path)

    cache_metadata = {
        "cache_version": CACHE_VERSION,
        "cache_policy": CACHE_POLICY,
        "seed": seed,
        "mode": mode,
        "device_type": device.type,
        "torch_version": torch.__version__,
        "train_config": asdict(cfg),
        "checkpoint_name": ckpt_path.name,
        "checkpoint_size": ckpt_path.stat().st_size,
    }
    if mode == "feature_extraction":
        cache_metadata.update({
            "feature_extraction_strategy": FEATURE_EXTRACTION_STRATEGY,
            "feature_extraction_selection": FEATURE_EXTRACTION_SELECTION,
        })
    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{mode}_{device.type}_split{seed}.pkl"
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

    print(f"[{tag}] split={seed}, mode={mode}, device={device}")

    # ---- 数据 ----
    train_segs = load_ctmp_segments(seed=seed, split="train")
    val_segs = load_ctmp_segments(seed=seed, split="val")
    test_segs = load_ctmp_segments(seed=seed, split="test")
    ext_segs = load_ctmp_segments(seed=seed, split="external_test")

    train_ds = PannsSegmentDataset(train_segs, LABEL_MAP)
    val_ds = PannsSegmentDataset(val_segs, LABEL_MAP)
    test_ds = PannsSegmentDataset(test_segs, LABEL_MAP)
    ext_ds = PannsSegmentDataset(ext_segs, LABEL_MAP)

    print(
        f"  train={len(train_ds)}, val={len(val_ds)}, "
        f"test={len(test_ds)}, external_test={len(ext_ds)}"
    )

    # 类别权重
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
    torch.manual_seed(42)
    model = TransferCnn14(n_classes=N_CLASSES, freeze_base=(mode == "feature_extraction"))
    model.load_from_pretrain(ckpt_path)
    model = model.to(device)

    if mode == "feature_extraction":
        freeze_backbone(model)
        t0 = time.perf_counter()
        print("  extracting frozen CNN14 embeddings...")
        extract_train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers,
        )
        train_x, train_y = _extract_embeddings(model, extract_train_loader, device)
        val_x, val_y = _extract_embeddings(model, val_loader, device)
        test_x, test_y = _extract_embeddings(model, test_loader, device)
        ext_x, ext_y = _extract_embeddings(model, ext_loader, device)
        print(
            f"  embeddings: train={tuple(train_x.shape)} val={tuple(val_x.shape)} "
            f"test={tuple(test_x.shape)} ext={tuple(ext_x.shape)}"
        )

        generator = torch.Generator().manual_seed(42)
        head_train_loader = DataLoader(
            TensorDataset(train_x, train_y), batch_size=cfg.batch_size,
            shuffle=True, generator=generator,
        )
        head_val_loader = DataLoader(
            TensorDataset(val_x, val_y), batch_size=cfg.batch_size, shuffle=False,
        )
        head_test_loader = DataLoader(
            TensorDataset(test_x, test_y), batch_size=cfg.batch_size, shuffle=False,
        )
        head_ext_loader = DataLoader(
            TensorDataset(ext_x, ext_y), batch_size=cfg.batch_size, shuffle=False,
        )

        head = model.fc_transfer.cpu()
        criterion_head = nn.CrossEntropyLoss(
            weight=torch.tensor(cw, dtype=torch.float32),
            label_smoothing=cfg.label_smoothing,
        )
        optimizer_head = torch.optim.Adam(
            head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        scheduler_head = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_head, mode="min", factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": [],
        }
        best_loss = float("inf")
        best_f1 = -float("inf")
        best_epoch = 0
        best_state: Optional[dict] = None
        for epoch in range(1, cfg.epochs + 1):
            tr_loss, tr_acc = _train_head_one_epoch(
                head, head_train_loader, criterion_head, optimizer_head,
            )
            va = _evaluate_head(head, head_val_loader, criterion_head)
            scheduler_head.step(va["loss"])
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va["loss"])
            history["val_acc"].append(va["acc"])
            history["val_f1"].append(va["f1"])
            if _is_better_feature_epoch(
                va["f1"], va["loss"], best_f1, best_loss,
            ):
                best_loss = va["loss"]
                best_f1 = va["f1"]
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in head.state_dict().items()
                }
            if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
                print(
                    f"  epoch {epoch:02d}/{cfg.epochs} train_loss={tr_loss:.3f} "
                    f"train_acc={tr_acc:.3f} val_loss={va['loss']:.3f} "
                    f"val_acc={va['acc']:.3f} val_f1={va['f1']:.3f}"
                )
        elapsed = time.perf_counter() - t0
        if best_state is not None:
            head.load_state_dict(best_state)
        val_eval = _evaluate_head(head, head_val_loader, criterion_head)
        test_eval = _evaluate_head(head, head_test_loader, criterion_head)
        ext_eval = _evaluate_head(head, head_ext_loader, criterion_head)
        result = {
            "seed": seed,
            "mode": mode,
            "tag": tag,
            **cache_metadata,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
            "best_val_loss": best_loss,
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
        print(
            f"[{tag}] done in {elapsed:.1f}s; best epoch={best_epoch} "
            f"(val_f1={best_f1:.3f}, val_loss={best_loss:.3f})"
        )
        _write_cache(cache_file, result, tag)
        return result

    # 按模式配置优化器
    if mode == "fine_tune":
        fine_tune_all(model)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.backbone_lr, weight_decay=cfg.weight_decay,
        )
    else:  # layer_wise
        param_groups = layer_wise_lr(model, cfg.backbone_lr, cfg.lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight_t, label_smoothing=cfg.label_smoothing)

    # ---- 训练 ----
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }
    best_loss = float("inf")
    best_epoch = 0
    best_state: Optional[dict] = None
    t0 = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        va = _evaluate(model, val_loader, criterion, device)
        scheduler.step(va["loss"])
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["val_f1"].append(va["f1"])
        if va["loss"] < best_loss:
            best_loss = va["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            print(f"  epoch {epoch:02d}/{cfg.epochs}  "
                  f"train_loss={tr_loss:.3f}  train_acc={tr_acc:.3f}  "
                  f"val_loss={va['loss']:.3f}  val_acc={va['acc']:.3f}  val_f1={va['f1']:.3f}")

    elapsed = time.perf_counter() - t0
    print(f"[{tag}] training done in {elapsed:.1f}s; best epoch={best_epoch} (val_loss={best_loss:.3f})")

    # ---- 最终评估（用 val 选出的权重；test/external_test 此前未参与训练决策） ----
    if best_state is not None:
        model.load_state_dict(best_state)
    val_eval = _evaluate(model, val_loader, criterion, device)
    test_eval = _evaluate(model, test_loader, criterion, device)
    ext_eval = _evaluate(model, ext_loader, criterion, device)

    result = {
        "seed": seed,
        "mode": mode,
        "tag": tag,
        **cache_metadata,
        "history": history,
        "best_epoch": best_epoch,
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
    _write_cache(cache_file, result, tag)
    return result
