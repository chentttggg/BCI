"""ShallowConvNet model (Schirrmeister et al., 2017) and model bundle IO."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PreprocessConfig, TrainConfig


def _safe_conv_out(length: int, kernel: int, stride: int = 1) -> int:
    return (length - kernel) // stride + 1


def _safe_pool_out(length: int, kernel: int, stride: int) -> int:
    return (length - kernel) // stride + 1


def build_shallow_convnet(n_channels: int, n_times: int, n_classes: int = 2,
                          train_cfg: TrainConfig | None = None,
                          model_sfreq: float = 250.0) -> Any:
    """Build a faithful ShallowConvNet adapted to the current time length.

    Paper: temporal conv (40 filters, 1x25) -> spatial conv (40 filters, Cx1)
           -> batch norm -> square -> avg pool (1x75, stride 1x15) -> log
           -> dropout -> conv classifier.
    """
    import torch
    from torch import nn

    cfg = train_cfg or TrainConfig()
    kernel = max(3, int(round(cfg.kernel_time_s * model_sfreq)))
    if kernel % 2 == 0:
        kernel += 1  # odd temporal kernel is closer to the paper and avoids padding issues
    pool_size = max(2, int(round(cfg.pool_time_s * model_sfreq)))
    pool_stride = max(1, int(round(cfg.pool_stride_s * model_sfreq)))
    filters = int(cfg.temporal_filters)

    conv_out = _safe_conv_out(n_times, kernel)
    if conv_out < 1:
        raise ValueError(f"temporal kernel {kernel} larger than input {n_times}")
    pool_out = _safe_pool_out(conv_out, pool_size, pool_stride)
    while pool_out < 2 and pool_stride > 1:
        pool_stride = max(1, pool_stride // 2)
        pool_out = _safe_pool_out(conv_out, pool_size, pool_stride)
    if pool_out < 1:
        pool_size = max(1, conv_out // 2)
        pool_stride = 1
        pool_out = _safe_pool_out(conv_out, pool_size, pool_stride)

    class ShallowConvNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal = nn.Conv2d(1, filters, (1, kernel), padding=0, bias=True)
            self.spatial = nn.Conv2d(filters, filters, (n_channels, 1), padding=0, bias=False)
            self.batchnorm = nn.BatchNorm2d(filters)
            self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride))
            self.dropout = nn.Dropout(p=float(cfg.dropout))
            self.classifier = nn.Conv2d(filters, n_classes, (1, pool_out), bias=True)
            self.pool_out = int(pool_out)
            self._reset_parameters()

        def _reset_parameters(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

        def forward(self, x: Any) -> Any:
            x = self.temporal(x)
            x = self.spatial(x)
            x = self.batchnorm(x)
            x = x * x
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            x = self.dropout(x)
            x = self.classifier(x)
            return x.squeeze(-1).squeeze(-1)

    return ShallowConvNet()


@dataclass
class ModelBundle:
    models: list[Any]
    preprocess_cfg: PreprocessConfig
    train_cfg: TrainConfig
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    channels: list[str]
    extra: dict[str, Any]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(target) for each epoch, averaged over the ensemble."""
        import torch

        X = X.astype(np.float32)
        Xn = (X - self.scaler_mean[None, :, None]) / self.scaler_std[None, :, None]
        tensor = torch.from_numpy(Xn[:, None, :, :])
        probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(tensor)
                probs.append(torch.sigmoid(logits[:, 1] - logits[:, 0]).numpy())
        return np.mean(np.stack(probs), axis=0)

    def save(self, directory: str | Path) -> None:
        import torch
        from .utils import atomic_write_json

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        states = []
        for i, model in enumerate(self.models):
            path = directory / f"ensemble_{i}.pt"
            torch.save({"model_state": model.state_dict()}, path)
            states.append(str(path))
        atomic_write_json(directory / "bundle.json", {
            "channels": self.channels,
            "preprocess": self.preprocess_cfg.to_dict(),
            "train": self.train_cfg.to_dict(),
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_std": self.scaler_std.tolist(),
            "extra": self.extra,
            "model_files": states,
        })

    @classmethod
    def load(cls, directory: str | Path, map_location: str = "cpu") -> "ModelBundle":
        import torch
        from .utils import read_json

        directory = Path(directory)
        obj = read_json(directory / "bundle.json")
        pre_cfg = PreprocessConfig.from_dict(obj["preprocess"])
        train_cfg = TrainConfig.from_dict(obj["train"])
        n_channels = len(obj["channels"])
        n_times = pre_cfg.n_times
        model_files = [Path(s) for s in obj["model_files"]]
        models = []
        for path in model_files:
            if not path.is_absolute():
                path = directory / path.name
            model = build_shallow_convnet(n_channels, n_times, 2, train_cfg,
                                          model_sfreq=pre_cfg.downsample_sfreq)
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            models.append(model)
        return cls(models=models,
                   preprocess_cfg=pre_cfg,
                   train_cfg=train_cfg,
                   scaler_mean=np.asarray(obj["scaler_mean"], dtype=np.float32),
                   scaler_std=np.asarray(obj["scaler_std"], dtype=np.float32),
                   channels=list(obj["channels"]),
                   extra=dict(obj.get("extra", {})))

