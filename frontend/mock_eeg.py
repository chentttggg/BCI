"""Synthetic 8-channel EEG generator for offline tests and --mock mode."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal


@dataclass
class Stimulus:
    onset_sample: int
    duration_samples: int
    number: int
    block: int
    trial: int


def _make_background_noise(n_samples: int, sfreq: int, n_channels: int,
                           seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n_channels, n_samples))
    sos = signal.butter(4, [0.5, 40.0], btype="bandpass", fs=sfreq, output="sos")
    noise = signal.sosfiltfilt(sos, white, axis=-1)
    # A few microvolts of slow, spatially coherent drift.
    t = np.arange(n_samples) / sfreq
    drift = 8.0 * np.sin(2 * np.pi * 0.08 * t + rng.uniform(0, 2 * np.pi, size=(n_channels, 1)))
    drift += 4.0 * np.sin(2 * np.pi * 0.21 * t + rng.uniform(0, 2 * np.pi, size=(n_channels, 1)))
    noise = noise / np.std(noise) * 2.2 + drift
    # Occipital/central alpha.
    alpha = 5.0 * np.sin(2 * np.pi * 10.0 * t)
    alpha_env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t + 0.7)
    weights = np.array([0.15, 0.25, 0.45, 0.55, 0.45, 0.85, 0.85, 1.0], dtype=float)[:, None]
    noise += weights * alpha * alpha_env
    return noise.astype(np.float32)


def _erp_templates(sfreq: int, channels: list[str]) -> dict[str, np.ndarray]:
    """Target and non-target template in channel x time."""
    duration_s = 1.0
    n = int(round(duration_s * sfreq))
    t = np.arange(n) / sfreq

    def gauss(center: float, width: float, amp: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    weights = np.array([0.15, 0.55, 0.65, 1.00, 0.65, 0.55, 0.55, 0.60], dtype=float)
    target = (gauss(0.22, 0.025, -2.0) + gauss(0.33, 0.035, 7.5) +
              gauss(0.45, 0.10, 3.5) + gauss(0.62, 0.14, 1.5))
    target = target[None, :] * weights[:, None]

    nontarget = (gauss(0.16, 0.025, 2.2) + gauss(0.26, 0.04, -2.0) +
                 gauss(0.38, 0.10, 0.8))
    nontarget = nontarget[None, :] * (weights[:, None] * 0.45)
    return {"target": target.astype(np.float32), "nontarget": nontarget.astype(np.float32)}


def generate_session_data(sfreq: int, channels: list[str], stimuli: list[Stimulus],
                          target_number: int, seed: int = 0,
                          noise_uv: float = 3.0) -> np.ndarray:
    """Generate continuous synthetic EEG for a complete paradigm."""
    n_channels = len(channels)
    if not stimuli:
        raise ValueError("stimuli list is empty")
    n_samples = int(max(s.onset_sample + s.duration_samples for s in stimuli) + sfreq * 2)
    raw = _make_background_noise(n_samples, sfreq, n_channels, seed).astype(np.float64)
    templates = _erp_templates(sfreq, channels)
    rng = np.random.default_rng(seed + 1000)
    for stim in stimuli:
        tpl = templates["target"] if stim.number == target_number else templates["nontarget"]
        # Small per-trial amplitude/latency jitter makes training non-trivial.
        amp = float(rng.uniform(0.85, 1.15))
        shift = int(rng.integers(-8, 9))
        start = stim.onset_sample + shift
        length = tpl.shape[1]
        if start < 0 or start + length > n_samples:
            continue
        raw[:, start:start + length] += tpl * amp
    raw += noise_uv * np.random.default_rng(seed + 7).standard_normal(raw.shape)
    return np.asarray(raw, dtype=np.float32)


def build_stimulus_list(schedule: list[dict[str, Any]], sfreq: int) -> list[Stimulus]:
    """Convert paradigm schedule dicts to sample-indexed stimuli."""
    out = []
    for ev in schedule:
        if ev.get("kind") != "stim_on":
            continue
        out.append(Stimulus(
            onset_sample=int(round(float(ev["time_sec"]) * sfreq)),
            duration_samples=max(1, int(round(float(ev["duration_sec"]) * sfreq))),
            number=int(ev["number"]),
            block=int(ev["block"]),
            trial=int(ev["trial"]),
        ))
    return out
