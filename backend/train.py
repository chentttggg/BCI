"""Train and cross-validate ShallowConvNet ensembles for the guess-number P300 task."""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ChannelConfig, PreprocessConfig, TrainConfig, load_configs
from .dataset import TrialDataset, make_loss_fn
from .io import load_session
from .model import ModelBundle, build_shallow_convnet
from .preprocess import prepare_session
from .scoring import aggregate_number_scores, binary_metrics, block_predictions
from .utils import atomic_write_json, configure_logging, sha256_file, utc_now_iso

logger = logging.getLogger("backend.train")


@dataclass
class PreparedData:
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    sidecars: list[dict[str, Any]] = field(default_factory=list)
    input_hashes: dict[str, str] = field(default_factory=dict)


def find_sessions(data_dir: str | Path, pattern: str = "*.edf") -> list[dict[str, Path]]:
    data_dir = Path(data_dir)
    sessions: list[dict[str, Path]] = []
    seen: set[Path] = set()
    for edf in sorted(data_dir.rglob(pattern)):
        if "derived" in edf.parts or "derivatives" in edf.parts:
            continue
        if edf in seen:
            continue
        seen.add(edf)
        stem = edf.stem
        events_candidates = [
            edf.with_name(stem + "_events.jsonl"),
            edf.with_name(stem + ".events.jsonl"),
        ]
        events = next((p for p in events_candidates if p.exists()), None)
        sessions.append({"edf": edf, "events": events})
    if not sessions:
        raise FileNotFoundError(f"No EDF files found under {data_dir}")
    return sessions


def build_prepared_data(data_dir: str | Path, pre_cfg: PreprocessConfig,
                        channel_cfg: ChannelConfig) -> PreparedData:
    prepared = PreparedData(X=np.empty((0, len(channel_cfg.channels), pre_cfg.n_times), dtype=np.float32),
                            y=np.empty((0,), dtype=np.int64),
                            meta=pd.DataFrame())
    for item in find_sessions(data_dir):
        session = load_session(item["edf"], item["events"])
        if session.target_number is None:
            logger.warning("Skipping %s: no target_number metadata (unsupervised session)", item["edf"])
            continue
        X, meta, sidecar = prepare_session(session, pre_cfg, channel_cfg.channels)
        if not sidecar.get("qc_pass", True):
            raise RuntimeError(
                f"QC gate failed for {item['edf']}: {sidecar.get('qc', {}).get('issues')}")
        # Exclude artifact-marked trials from supervised training. The full mask is
        # retained in `sidecar` so no rejection is silent (Constitution Stage 4).
        good_mask = meta["bad_trial"].to_numpy(dtype=bool) == 0
        n_before = int(len(meta))
        X = X[good_mask]
        meta = meta[good_mask].reset_index(drop=True)
        sidecar["n_trials_after_artifact_exclusion"] = int(len(meta))
        sidecar["n_trials_excluded_for_training"] = n_before - int(len(meta))
        meta["session_id"] = item["edf"].stem
        meta["target_number"] = session.target_number
        meta["is_target"] = (meta["number"] == session.target_number).astype(int)
        if len(meta) == 0:
            logger.warning("Skipping %s: all epochs rejected by artifact QC", item["edf"])
            prepared.sidecars.append(sidecar)
            continue
        prepared.sidecars.append(sidecar)
        prepared.input_hashes[item["edf"].stem] = sha256_file(item["edf"])
        logger.info("Prepared %s: %d epochs, target=%s", item["edf"].name, len(meta), session.target_number)
        if prepared.meta.empty:
            prepared.X = X
            prepared.y = meta["is_target"].to_numpy(dtype=np.int64)
            prepared.meta = meta
        else:
            prepared.X = np.concatenate([prepared.X, X], axis=0)
            prepared.y = np.concatenate([prepared.y, meta["is_target"].to_numpy(dtype=np.int64)], axis=0)
            prepared.meta = pd.concat([prepared.meta, meta], ignore_index=True)
    if prepared.meta.empty:
        raise ValueError("No labeled sessions found. Add target_number to session.json first.")
    prepared.meta = prepared.meta.reset_index(drop=True)
    return prepared


def _unique_sessions(meta: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(meta["session_id"].astype(str)))


def _block_key(meta: pd.DataFrame) -> pd.Series:
    return meta["session_id"].astype(str) + "__" + meta["block"].astype(int).astype(str)


def loso_splits(meta: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    sessions = _unique_sessions(meta)
    splits = []
    for test_session in sessions:
        test_idx = meta.index[meta["session_id"].astype(str) == test_session].to_numpy()
        train_idx = meta.index[meta["session_id"].astype(str) != test_session].to_numpy()
        splits.append((train_idx, test_idx))
    return splits


def block_kfold_splits(meta: pd.DataFrame, n_splits: int = 5,
                       seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold splits that keep whole blocks together; stratified-ish by target proportion."""
    rng = np.random.default_rng(seed)
    keys = np.asarray(_block_key(meta), dtype=object)
    unique = list(dict.fromkeys(keys.tolist()))
    target_ratio = []
    for key in unique:
        idx = meta.index[keys == key].to_numpy()
        target_ratio.append(float(meta.loc[idx, "is_target"].mean()))
    order = rng.permutation(len(unique))
    sorted_order = order[np.argsort([target_ratio[i] for i in order], kind="stable")][::-1]
    folds = [[] for _ in range(n_splits)]
    for i, pos in enumerate(sorted_order):
        folds[i % n_splits].append(unique[pos])
    splits = []
    for k in range(n_splits):
        test_keys = set(folds[k])
        test_mask = np.isin(keys, list(test_keys))
        splits.append((meta.index[~test_mask].to_numpy(), meta.index[test_mask].to_numpy()))
    return splits


def train_val_block_split(meta: pd.DataFrame, val_ratio: float = 0.15,
                          seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = np.asarray(_block_key(meta), dtype=object)
    unique = list(dict.fromkeys(keys.tolist()))
    val_keys = set(rng.choice(unique, size=max(1, int(round(len(unique) * val_ratio))), replace=False))
    val_mask = np.isin(keys, list(val_keys))
    return meta.index[~val_mask].to_numpy(), meta.index[val_mask].to_numpy()


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, train: bool,
                 cfg: TrainConfig, seed: int) -> Any:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    aug = cfg.augmentation
    ds = TrialDataset(
        X, y,
        train=train,
        time_shift_samples=aug.time_shift_samples if aug.enable else 0,
        amplitude_scale_range=tuple(aug.amplitude_scale_range) if aug.enable else (1.0, 1.0),
        channel_dropout_prob=aug.channel_dropout_prob if aug.enable else 0.0,
        noise_std=aug.noise_std if aug.enable else 0.0,
        seed=seed,
    )
    if train:
        y_flat = np.asarray(y).ravel()
        n_pos = max(1, int(y_flat.sum()))
        n_neg = max(1, int((1 - y_flat).sum()))
        weights = np.where(y_flat == 1, 1.0 / n_pos, 1.0 / n_neg)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(y_flat), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _predict_logits(model: Any, loader: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    outs = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            outs.append(logits.numpy())
            ys.append(yb.numpy())
    return np.concatenate(outs), np.concatenate(ys)


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_epoch(model: Any, loader: Any, optimizer: Any, loss_fn: Any) -> float:
    import torch

    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits[:, 1:2], yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.item()) * len(xb)
        n += len(xb)
    return total / max(1, n)


def fit_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              pre_cfg: PreprocessConfig, train_cfg: TrainConfig, seed: int,
              scaler: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    import torch

    _set_seed(seed)
    mean, std = scaler if scaler is not None else (X_train.mean(axis=(0, 2)), X_train.std(axis=(0, 2)) + 1e-6)
    Xt = ((X_train - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    Xv = ((X_val - mean[None, :, None]) / std[None, :, None]).astype(np.float32)

    loader = _make_loader(Xt, y_train, train_cfg.batch_size, True, train_cfg, seed)
    val_loader = _make_loader(Xv, y_val, max(32, train_cfg.batch_size), False, train_cfg, seed)

    model = build_shallow_convnet(X_train.shape[1], X_train.shape[2], 2, train_cfg,
                                  model_sfreq=pre_cfg.downsample_sfreq)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)
    if train_cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10)

    loss_fn = make_loss_fn(train_cfg.loss, train_cfg.focal_gamma, train_cfg.focal_alpha)

    best_state = None
    best_metric = -1.0
    best_epoch = 0
    patience_left = train_cfg.early_stop_patience
    train_losses: list[float] = []
    val_metrics: list[dict[str, float]] = []
    epochs = train_cfg.epochs

    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, loader, optimizer, loss_fn)
        train_losses.append(loss)
        logits, yb = _predict_logits(model, val_loader)
        probs = torch.sigmoid(torch.from_numpy(logits[:, 1] - logits[:, 0])).numpy()
        m = binary_metrics(yb.ravel().astype(int), probs)
        m["loss"] = loss
        val_metrics.append(m)
        metric = m["balanced_accuracy"] if not np.isnan(m["balanced_accuracy"]) else 0.0

        if train_cfg.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(metric)

        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            patience_left = train_cfg.early_stop_patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1

        if epoch >= train_cfg.min_epochs and patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model, "best_epoch": best_epoch,
            "best_val_balanced_accuracy": float(best_metric),
            "train_losses": train_losses, "val_metrics": val_metrics}


def _eval_model(model: Any, X: np.ndarray, y: np.ndarray, scaler: tuple[np.ndarray, np.ndarray],
                batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    import torch

    mean, std = scaler
    Xn = ((X - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    tensor = torch.from_numpy(Xn[:, None, :, :])
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            outs.append(model(tensor[i:i + batch_size]).numpy())
    logits = np.concatenate(outs)
    probs = torch.sigmoid(torch.from_numpy(logits[:, 1] - logits[:, 0])).numpy()
    return probs, y


def run_cross_validation(prepared: PreparedData, pre_cfg: PreprocessConfig,
                         train_cfg: TrainConfig, output_dir: Path) -> dict[str, Any]:
    splits = loso_splits(prepared.meta) if len(_unique_sessions(prepared.meta)) > 1 else \
        block_kfold_splits(prepared.meta, n_splits=5, seed=0)
    fold_reports = []
    all_block_preds = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        if len(val_idx) < 2 or len(train_idx) < 8:
            logger.warning("Skipping fold %d: too few trials", fold_idx)
            continue
        Xtr, ytr = prepared.X[train_idx], prepared.y[train_idx]
        Xval, yval = prepared.X[val_idx], prepared.y[val_idx]
        scaler = (Xtr.mean(axis=(0, 2)), Xtr.std(axis=(0, 2)) + 1e-6)
        fold_seed_reports = []
        fold_probs = []
        for seed in train_cfg.cv_seeds:
            result = fit_model(Xtr, ytr, Xval, yval, pre_cfg, train_cfg, seed, scaler)
            probs, _ = _eval_model(result["model"], Xval, yval, scaler)
            fold_probs.append(probs)
            fold_seed_reports.append({
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "best_val_balanced_accuracy": result["best_val_balanced_accuracy"],
            })
        probs = np.mean(fold_probs, axis=0)
        bm = binary_metrics(yval.astype(int), probs)
        block_df = block_predictions(prepared.meta.loc[val_idx].reset_index(drop=True), probs)
        block_acc = float(block_df["correct"].mean()) if len(block_df) else float("nan")
        fold_reports.append({"fold": fold_idx, "binary": bm, "block_accuracy": block_acc,
                             "seeds": fold_seed_reports,
                             "n_train": int(len(train_idx)), "n_val": int(len(val_idx))})
        block_df["fold"] = fold_idx
        all_block_preds.append(block_df)
        logger.info("Fold %d: AUC=%s balAcc=%s blockAcc=%s", fold_idx,
                    bm.get("auc"), bm.get("balanced_accuracy"), block_acc)

    summary = {
        "cv_type": "loso" if len(_unique_sessions(prepared.meta)) > 1 else "block_5fold",
        "mean_auc": float(np.nanmean([f["binary"].get("auc", np.nan) for f in fold_reports])),
        "mean_balanced_accuracy": float(np.nanmean([f["binary"]["balanced_accuracy"] for f in fold_reports])),
        "mean_block_accuracy": float(np.nanmean([f["block_accuracy"] for f in fold_reports])),
        "folds": fold_reports,
    }
    atomic_write_json(output_dir / "cv_report.json", summary)
    if all_block_preds:
        pd.concat(all_block_preds, ignore_index=True).to_json(
            output_dir / "cv_block_predictions.jsonl", orient="records", lines=True, force_ascii=False)
    return summary


def train_production_ensemble(prepared: PreparedData, pre_cfg: PreprocessConfig,
                              train_cfg: TrainConfig, output_dir: Path,
                              channels: list[str] | None = None) -> ModelBundle:
    X, y, meta = prepared.X, prepared.y, prepared.meta
    mean = X.mean(axis=(0, 2))
    std = X.std(axis=(0, 2)) + 1e-6
    channels = [c.upper() for c in (channels or [])]

    models = []
    reports = []
    for seed in train_cfg.production_seeds:
        # 1) internal block-wise validation selects the epoch count
        tr_idx, va_idx = train_val_block_split(meta, val_ratio=0.15, seed=seed)
        if len(va_idx) < 2:
            va_idx = np.random.default_rng(seed).choice(
                len(meta), size=max(2, int(0.15 * len(meta))), replace=False)
            tr_idx = np.setdiff1d(np.arange(len(meta)), va_idx)
        held = fit_model(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
                         pre_cfg, train_cfg, seed, (mean, std))
        best_epoch = max(1, held["best_epoch"])
        # 2) retrain on all data for the selected epoch count
        exact = _fit_exact_epochs(X, y, X, y, pre_cfg, train_cfg, seed, (mean, std), best_epoch)
        models.append(exact["model"])
        reports.append({
            "seed": seed,
            "internal_best_epoch": best_epoch,
            "internal_val_balanced_accuracy": held["best_val_balanced_accuracy"],
            "final_train_loss": float(exact["train_losses"][-1]) if exact["train_losses"] else None,
        })
        logger.info("Production seed %d trained (epochs=%d)", seed, best_epoch)

    bundle = ModelBundle(models=models, preprocess_cfg=pre_cfg, train_cfg=train_cfg,
                         scaler_mean=mean.astype(np.float32), scaler_std=std.astype(np.float32),
                         channels=channels,
                         extra={"production_reports": reports,
                                "input_hashes": prepared.input_hashes,
                                "created_utc": utc_now_iso()})
    bundle.save(output_dir)
    atomic_write_json(output_dir / "production_report.json", {
        "n_train_epochs": int(len(meta)),
        "n_target": int(y.sum()),
        "seeds": reports,
    })
    return bundle


def _fit_exact_epochs(X_train, y_train, X_val, y_val, pre_cfg, train_cfg, seed,
                      scaler, epochs: int) -> dict[str, Any]:
    import torch

    _set_seed(seed)
    mean, std = scaler
    Xt = ((X_train - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    Xv = ((X_val - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    loader = _make_loader(Xt, y_train, train_cfg.batch_size, True, train_cfg, seed)
    val_loader = _make_loader(Xv, y_val, max(32, train_cfg.batch_size), False, train_cfg, seed)
    model = build_shallow_convnet(X_train.shape[1], X_train.shape[2], 2, train_cfg,
                                  model_sfreq=pre_cfg.downsample_sfreq)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_fn = make_loss_fn(train_cfg.loss, train_cfg.focal_gamma, train_cfg.focal_alpha)
    train_losses = []
    for _ in range(epochs):
        train_losses.append(_train_epoch(model, loader, optimizer, loss_fn))
        scheduler.step()
    return {"model": model, "train_losses": train_losses}


def run(args: argparse.Namespace) -> None:
    pre_cfg, train_cfg = load_configs(args.preprocess_config, args.train_config)
    channel_cfg = ChannelConfig.from_json(args.channel_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "preprocess_params.json", pre_cfg.to_dict())
    atomic_write_json(output_dir / "train_params.json", train_cfg.to_dict())

    prepared = build_prepared_data(args.data_dir, pre_cfg, channel_cfg)
    atomic_write_json(output_dir / "input_files.json", prepared.input_hashes)

    if args.cv:
        summary = run_cross_validation(prepared, pre_cfg, train_cfg, output_dir)
        print(json.dumps({"cv_summary": summary}, ensure_ascii=False, indent=2))

    if args.production:
        bundle = train_production_ensemble(prepared, pre_cfg, train_cfg, output_dir,
                                           channel_cfg.channels)
        print(f"Saved production ensemble to {output_dir}")
    elif args.cv:
        # also save a compact production ensemble with cv_seeds on all data
        old = train_cfg.production_seeds
        train_cfg.production_seeds = train_cfg.cv_seeds
        try:
            bundle = train_production_ensemble(prepared, pre_cfg, train_cfg, output_dir,
                                               channel_cfg.channels)
            print(f"Saved CV-seed ensemble to {output_dir}")
        finally:
            train_cfg.production_seeds = old
    else:
        logger.warning("Nothing to do: use --cv and/or --production")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ShallowConvNet guess-number decoder")
    p.add_argument("--data-dir", default="data/raw", help="directory containing raw *_eeg.edf sessions")
    p.add_argument("--output-dir", default="data/derived/models/guess_number")
    p.add_argument("--preprocess-config", default="config/preprocessing.json")
    p.add_argument("--train-config", default="config/train.json")
    p.add_argument("--channel-config", default="config/channel_config.json")
    p.add_argument("--cv", action="store_true", help="run leave-one-session-out/block CV evaluation")
    p.add_argument("--no-cv", action="store_false", dest="cv")
    p.add_argument("--production", action="store_true", help="train final ensemble on all labeled data")
    p.add_argument("--no-production", action="store_false", dest="production")
    p.add_argument("--log-file", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging(logging.INFO, args.log_file)
    t0 = time.time()
    run(args)
    logger.info("Training finished in %.1f seconds", time.time() - t0)


if __name__ == "__main__":
    main()
