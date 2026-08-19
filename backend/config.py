"""Project dataclasses and JSON config loading."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import read_json


@dataclass
class ChannelConfig:
    channels: list[str]
    ref_label: str
    gnd_label: str
    indices: list[int]

    @classmethod
    def from_json(cls, path: str | Path) -> "ChannelConfig":
        obj = read_json(path)
        items = obj["eeg_channels"]
        labels = [str(x["label"]) for x in items]
        indices = [int(x["index"]) for x in items]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Duplicate channel labels in {path}")
        return cls(channels=labels, ref_label=str(obj.get("ref_label", "")),
                   gnd_label=str(obj.get("gnd_label", "")), indices=indices)


@dataclass
class PreprocessConfig:
    raw_sfreq: float = 500.0
    downsample_sfreq: float = 250.0
    highpass_hz: float = 0.5
    lowpass_hz: float = 20.0
    notch_hz: float = 50.0
    notch_harmonics: list[float] = field(default_factory=lambda: [100.0])
    tmin_s: float = -0.2
    tmax_s: float = 1.0
    baseline_s: list[float] = field(default_factory=lambda: [-0.2, 0.0])
    reref: str = "car"
    artifact_abs_uv: float = 120.0
    artifact_ptp_uv: float = 150.0
    max_bad_epoch_ratio: float = 0.30
    max_bad_channels: int = 1
    leadoff_normal_value: int = 255
    xdawn_enable: bool = True
    xdawn_target_components: int = 2
    xdawn_nontarget_components: int = 1
    xdawn_reg: float = 1e-6

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "PreprocessConfig":
        known = {k: v for k, v in obj.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessConfig":
        return cls.from_dict(read_json(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def n_times(self) -> int:
        return int(round((self.tmax_s - self.tmin_s) * self.downsample_sfreq))

    @property
    def baseline_start_sample(self) -> int:
        return int(round((0.0 - self.tmin_s) * self.downsample_sfreq))


@dataclass
class AugmentationConfig:
    enable: bool = True
    time_shift_samples: int = 20
    amplitude_scale_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    channel_dropout_prob: float = 0.15
    noise_std: float = 0.20
    mixup_alpha: float = 0.20

    @classmethod
    def from_dict(cls, obj: dict[str, Any] | None) -> "AugmentationConfig":
        obj = obj or {}
        known = {k: v for k, v in obj.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class TrainConfig:
    epochs: int = 150
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    temporal_filters: int = 60
    kernel_time_s: float = 0.20
    pool_time_s: float = 0.30
    pool_stride_s: float = 0.06
    dropout: float = 0.5
    loss: str = "focal"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    early_stop_patience: int = 25
    min_epochs: int = 30
    cv_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    production_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    pos_weight_mode: str = "balanced"

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "TrainConfig":
        aug = AugmentationConfig.from_dict(obj.get("augmentation"))
        known = {k: v for k, v in obj.items() if k in cls.__dataclass_fields__ and k != "augmentation"}
        return cls(augmentation=aug, **known)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(read_json(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_configs(preprocess_path: str | Path = "config/preprocessing.json",
                 train_path: str | Path = "config/train.json") -> tuple[PreprocessConfig, TrainConfig]:
    return PreprocessConfig.load(preprocess_path), TrainConfig.load(train_path)


def save_json_config(path: str | Path, obj: Any) -> None:
    from .utils import atomic_write_json
    atomic_write_json(path, obj)
