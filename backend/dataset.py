"""Loss functions and PyTorch datasets for binary target/non-target trials."""
from __future__ import annotations

from typing import Any

import numpy as np


def focal_binary_cross_entropy(logits: Any, targets: Any, gamma: float = 2.0,
                               alpha: float = 0.75) -> Any:
    """Focal loss for binary classification with rare positive class.

    alpha is the weight assigned to positive (target) trials.
    """
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * ((1.0 - p_t) ** gamma) * ce
    return loss.mean()


def class_balanced_bce(logits: Any, targets: Any, pos_weight: float | None = None) -> Any:
    import torch
    import torch.nn.functional as F

    if pos_weight is None:
        with torch.no_grad():
            n_pos = targets.sum().clamp(min=1.0)
            n_neg = (1.0 - targets).sum().clamp(min=1.0)
            pos_weight = n_neg / n_pos
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor(float(pos_weight)))


def make_loss_fn(loss_name: str, gamma: float, alpha: float) -> Any:
    if loss_name == "focal":
        return lambda logits, targets: focal_binary_cross_entropy(logits, targets, gamma, alpha)
    if loss_name in ("bce", "balanced_bce"):
        return lambda logits, targets: class_balanced_bce(logits, targets)
    raise ValueError(f"Unknown loss {loss_name!r}")


class TrialDataset:
    """Minimal dataset with deterministic, documented augmentation for training."""

    def __init__(self, X: np.ndarray, y: np.ndarray, train: bool = True,
                 time_shift_samples: int = 0, amplitude_scale_range: tuple[float, float] = (1.0, 1.0),
                 channel_dropout_prob: float = 0.0,
                 channel_dropout_max_channels: int = 1,
                 noise_std: float = 0.0,
                 mixup_alpha: float = 0.0, seed: int | None = None) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.train = bool(train)
        self.time_shift_samples = int(time_shift_samples)
        self.amp_range = tuple(amplitude_scale_range)
        self.channel_dropout_prob = float(channel_dropout_prob)
        self.channel_dropout_max_channels = int(max(1, channel_dropout_max_channels))
        self.noise_std = float(noise_std)
        self.mixup_alpha = float(mixup_alpha)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        import torch

        x = self.X[idx].copy()
        y = float(self.y[idx])
        if self.train:
            if self.mixup_alpha > 0 and self.rng.random() < 0.5:
                j = int(self.rng.integers(0, len(self.X)))
                lam = float(self.rng.beta(self.mixup_alpha, self.mixup_alpha))
                x = lam * self._augment(x) + (1.0 - lam) * self._augment(self.X[j].copy())
                y = lam * y + (1.0 - lam) * float(self.y[j])
            else:
                x = self._augment(x)
        x = x[np.newaxis, :, :]
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.time_shift_samples > 0:
            shift = int(self.rng.integers(-self.time_shift_samples, self.time_shift_samples + 1))
            if shift != 0:
                x = np.roll(x, shift, axis=-1)
        if self.amp_range != (1.0, 1.0):
            lo, hi = self.amp_range
            x = x * float(self.rng.uniform(lo, hi))
        if self.noise_std > 0:
            x = x + float(self.noise_std) * self.rng.standard_normal(x.shape).astype(np.float32)
        if self.channel_dropout_prob > 0:
            n_channels = x.shape[0]
            drop_mask = self.rng.random(n_channels) < self.channel_dropout_prob
            if drop_mask.any():
                # With only 8 original channels (+3 xDAWN), dropping many entire
                # electrodes at once is too destructive: cap at one channel.
                drop_idx = np.flatnonzero(drop_mask)
                if len(drop_idx) > self.channel_dropout_max_channels:
                    drop_idx = self.rng.choice(
                        drop_idx, size=self.channel_dropout_max_channels, replace=False)
                mask = np.ones(n_channels, dtype=np.float32)
                mask[drop_idx] = 0.0
                x = x * mask[:, None].astype(np.float32)
        return x
