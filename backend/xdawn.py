"""xDAWN spatial filtering for single-trial ERP SNR enhancement.

xDAWN estimates spatial filters that maximise the ratio between the
class-conditional averaged ERP and the residual ongoing EEG.  It is a
well-established P300 preprocessing step (Rivet et al., 2009) and has been
combined with EEGNet/CNN classifiers successfully (Cascade xDAWN EEGNet).

We use it as *additional channels* for ShallowConvNet rather than replacing
the original channels, so the network can still learn raw-space patterns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import eigh


@dataclass
class XdawnProjector:
    target_filters: np.ndarray | None = None      # (n_channels, n_target_components)
    nontarget_filters: np.ndarray | None = None   # (n_channels, n_nontarget_components)
    reg: float = 1e-6
    include_original: bool = True

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_target_components: int = 2,
            n_nontarget_components: int = 1) -> "XdawnProjector":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.target_filters = self._fit_class(X, y == 1, n_target_components)
        if n_nontarget_components > 0:
            self.nontarget_filters = self._fit_class(X, y == 0, n_nontarget_components)
        else:
            self.nontarget_filters = None
        return self

    def _fit_class(self, X: np.ndarray, mask: np.ndarray, n_components: int) -> np.ndarray:
        if int(mask.sum()) == 0:
            raise ValueError("xDAWN needs at least one trial per class")
        Xc = X[mask]
        avg = Xc.mean(axis=0)                     # (C, T)
        residual = X - avg[None, :, :]            # all trials vs target-class average
        n, c, t = X.shape
        noise = np.zeros((c, c), dtype=np.float64)
        for i in range(n):
            r = residual[i]
            noise += r @ r.T
        noise /= max(1, n * t)
        signal_cov = avg @ avg.T

        signal_cov = signal_cov + self.reg * np.eye(c)
        noise = noise + self.reg * np.eye(c)
        eigvals, eigvecs = eigh(signal_cov, noise, check_finite=True)
        order = np.argsort(eigvals)[::-1][:n_components]
        return np.asarray(eigvecs[:, order], dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        parts = [X] if self.include_original else []
        if self.target_filters is not None:
            parts.append(self._apply(X, self.target_filters))
        if self.nontarget_filters is not None:
            parts.append(self._apply(X, self.nontarget_filters))
        return np.concatenate(parts, axis=1)

    @staticmethod
    def _apply(X: np.ndarray, filters: np.ndarray) -> np.ndarray:
        return np.einsum("ck,nct->nkt", filters.astype(np.float32), X).astype(np.float32)

    @property
    def original_channels(self) -> int:
        if self.target_filters is not None:
            return int(self.target_filters.shape[0])
        if self.nontarget_filters is not None:
            return int(self.nontarget_filters.shape[0])
        return 0

    @property
    def n_output_channels(self) -> int:
        return (self.original_channels if self.include_original else 0) + \
            int(0 if self.target_filters is None else self.target_filters.shape[1]) + \
            int(0 if self.nontarget_filters is None else self.nontarget_filters.shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_filters": None if self.target_filters is None else self.target_filters.tolist(),
            "nontarget_filters": None if self.nontarget_filters is None else self.nontarget_filters.tolist(),
            "reg": self.reg,
            "include_original": self.include_original,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "XdawnProjector":
        target = obj.get("target_filters")
        nontarget = obj.get("nontarget_filters")
        return cls(
            target_filters=None if target is None else np.asarray(target, dtype=np.float32),
            nontarget_filters=None if nontarget is None else np.asarray(nontarget, dtype=np.float32),
            reg=float(obj.get("reg", 1e-6)),
            include_original=bool(obj.get("include_original", True)),
        )

    def fit_from_config(self, X: np.ndarray, y: np.ndarray, cfg: Any) -> "XdawnProjector":
        if not cfg.xdawn_enable:
            self.target_filters = None
            self.nontarget_filters = None
            self.include_original = True
            return self
        return self.fit(X, y, int(cfg.xdawn_target_components),
                        int(cfg.xdawn_nontarget_components))
