"""Conservative preprocessing and epoch extraction.

Every processing step is deterministic and parameterised.  Artifact detection
only *marks* epochs/channels; it never silently deletes or interpolates data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from .config import PreprocessConfig
from .io import SessionData, reorder_channels


def validate_session(session: SessionData, expected_channels: list[str]) -> dict[str, Any]:
    """Return basic QC facts for one raw session."""
    issues: list[str] = []
    warnings: list[str] = []
    expected = [c.upper() for c in expected_channels]
    actual = [c.upper() for c in session.ch_names]
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    if missing:
        issues.append(f"missing channels: {missing}")
    if extra:
        warnings.append(f"extra/unmapped channels: {extra}")
    if session.events.empty:
        issues.append("no stim_on events found")
    if session.target_number is None:
        warnings.append("target_number not recorded; supervised training will skip/use metadata")
    if not np.isfinite(session.raw).all():
        issues.append("raw data contains non-finite values")
    return {
        "source": str(session.source_path),
        "sfreq": float(session.sfreq),
        "n_channels": session.n_channels,
        "n_samples": session.n_samples,
        "n_stim_events": int(len(session.events)),
        "target_number": session.target_number,
        "issues": issues,
        "warnings": warnings,
    }


def _butter_bandpass(data: np.ndarray, sfreq: float, low_hz: float, high_hz: float,
                     order: int = 4) -> np.ndarray:
    nyq = 0.5 * sfreq
    low = max(low_hz, 0.01) if low_hz > 0 else None
    high = min(high_hz, nyq * 0.95) if high_hz > 0 else None
    out = data.astype(np.float64, copy=True)
    if low is not None and low > 0:
        sos = signal.butter(order, low / nyq, btype="highpass", output="sos")
        out = signal.sosfiltfilt(sos, out, axis=-1)
    if high is not None and high < nyq:
        sos = signal.butter(order, high / nyq, btype="lowpass", output="sos")
        out = signal.sosfiltfilt(sos, out, axis=-1)
    return out


def _notch_filter(data: np.ndarray, sfreq: float, freqs: list[float], q: float = 30.0) -> np.ndarray:
    nyq = 0.5 * sfreq
    out = data.astype(np.float64, copy=True)
    for freq in freqs:
        if freq <= 0 or freq >= nyq:
            continue
        b, a = signal.iirnotch(freq, q, sfreq)
        out = signal.filtfilt(b, a, out, axis=-1)
    return out


def _resample(data: np.ndarray, src_sfreq: float, dst_sfreq: float) -> np.ndarray:
    if np.isclose(src_sfreq, dst_sfreq):
        return data
    ratio = Fraction(int(round(dst_sfreq * 100)), int(round(src_sfreq * 100)))
    up, down = ratio.numerator, ratio.denominator
    return signal.resample_poly(data, up, down, axis=-1)


def preprocess_continuous(data: np.ndarray, sfreq: float,
                          cfg: PreprocessConfig) -> tuple[np.ndarray, float]:
    """High-pass -> notch -> low-pass -> downsample. No channel dropping here."""
    data = _butter_bandpass(data, sfreq, cfg.highpass_hz, cfg.lowpass_hz)
    data = _notch_filter(data, sfreq, [cfg.notch_hz] + list(cfg.notch_harmonics))
    data = _resample(data, sfreq, cfg.downsample_sfreq)
    return np.asarray(data, dtype=np.float32), cfg.downsample_sfreq


def apply_reref(data: np.ndarray, mode: str = "car",
                bad_channel_mask: np.ndarray | None = None) -> np.ndarray:
    """Re-reference. `car` subtracts the mean of good channels, documented in sidecar."""
    mode = mode.strip().lower()
    if mode in ("none", "original"):
        return data.copy()
    if mode in ("car", "common", "average"):
        out = data.astype(np.float64, copy=True)
        if bad_channel_mask is not None and bad_channel_mask.any():
            good = ~bad_channel_mask
            ref = out[good, :].mean(axis=0, keepdims=True)
            out = out - ref
            out[bad_channel_mask, :] = np.nan
            return out
        return out - out.mean(axis=0, keepdims=True)
    raise ValueError(f"Unknown reref mode: {mode!r}")


def _baseline_correct(epoch: np.ndarray, start: int, stop: int) -> np.ndarray:
    base = epoch[:, start:stop].mean(axis=-1, keepdims=True)
    return epoch - base


def epoch_data(data: np.ndarray, sfreq: float, events: pd.DataFrame,
               tmin_s: float, tmax_s: float, baseline_s: tuple[float, float]) -> tuple[np.ndarray, pd.DataFrame, list[int]]:
    """Cut fixed windows around stim_on events.

    Returns X (n_epochs, n_channels, n_time), metadata, dropped event indices.
    """
    n_before = int(round(-tmin_s * sfreq))
    n_after = int(round(tmax_s * sfreq))
    n_time = n_before + n_after
    base_start = int(round((baseline_s[0] - tmin_s) * sfreq))
    base_stop = int(round((baseline_s[1] - tmin_s) * sfreq))
    base_start = max(0, min(base_start, n_time))
    base_stop = max(base_start, min(base_stop, n_time))

    epochs = []
    metas = []
    dropped: list[int] = []
    for idx, row in events.iterrows():
        onset = int(row["onset_sample"])
        start = onset - n_before
        stop = start + n_time
        if start < 0 or stop > data.shape[-1]:
            dropped.append(int(idx))
            continue
        ep = data[:, start:stop].astype(np.float32, copy=True)
        ep = _baseline_correct(ep, base_start, base_stop)
        epochs.append(ep)
        record = {
            "event_index": int(idx),
            "number": int(row["number"]),
            "block": float(row["block"]) if pd.notna(row["block"]) else -1,
            "trial": float(row["trial"]) if pd.notna(row["trial"]) else -1,
            "onset_sample": onset,
        }
        for col in ["leadoff_status", "trig_in_status", "is_impedance_mode",
                    "packet_seq", "packet_delta_time_us"]:
            if col in events.columns:
                record[col] = row[col]
        metas.append(record)
    if not epochs:
        raise ValueError("No stim_on event produced a full epoch window")
    X = np.stack(epochs)
    meta = pd.DataFrame(metas)
    return X, meta, dropped


@dataclass
class ArtifactInfo:
    bad_trial: np.ndarray
    bad_channel: np.ndarray
    reasons: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_bad_trials(self) -> int:
        return int(self.bad_trial.sum())

    @property
    def n_bad_channels(self) -> int:
        return int(self.bad_channel.sum())


def detect_artifacts(X: np.ndarray, cfg: PreprocessConfig) -> ArtifactInfo:
    """Conservative thresholding. Returns masks, never modifies X."""
    n_epochs, n_channels, _ = X.shape
    ptp = np.ptp(X, axis=-1)
    abs_max = np.max(np.abs(X), axis=-1)
    std = np.std(X, axis=-1)

    channel_ptp = np.nanmedian(ptp, axis=0)
    bad_channel = channel_ptp > cfg.artifact_ptp_uv * 2.0

    bad_by_ptp = ptp > cfg.artifact_ptp_uv
    bad_by_abs = abs_max > cfg.artifact_abs_uv
    # A trial is globally bad when any good channel violates thresholds.
    good_channels = ~bad_channel[None, :]
    bad_trial = np.any((bad_by_ptp | bad_by_abs) & good_channels, axis=1)

    bad_by_channel_mask = bad_by_ptp | bad_by_abs

    reasons = []
    for i in range(n_epochs):
        reasons.append({
            "epoch": int(i),
            "ptp_uv": [float(v) for v in ptp[i]],
            "abs_max_uv": [float(v) for v in abs_max[i]],
            "std_uv": [float(v) for v in std[i]],
            "bad_trial": bool(bad_trial[i]),
            "bad_channels": [int(j) for j in range(n_channels) if bad_by_channel_mask[i, j]],
        })

    metrics = {
        "n_epochs": n_epochs,
        "n_bad_trials": int(bad_trial.sum()),
        "bad_trial_ratio": float(bad_trial.mean()),
        "n_bad_channels": int(bad_channel.sum()),
        "channel_ptp_uv": [float(v) for v in channel_ptp],
        "channel_abs_median_uv": [float(v) for v in np.nanmedian(abs_max, axis=0)],
        "channel_std_median_uv": [float(v) for v in np.nanmedian(std, axis=0)],
    }
    return ArtifactInfo(bad_trial=bad_trial, bad_channel=bad_channel,
                        reasons=reasons, metrics=metrics)


def prepare_session(session: SessionData, cfg: PreprocessConfig,
                    expected_channels: list[str]) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Full fixed preprocessing for one session: validate, reorder, filter, CAR, epoch, mark artifacts."""
    raw, used_channels, problems = reorder_channels(
        session.raw, session.ch_names, expected_channels)
    qc = validate_session(session, expected_channels)
    qc["used_channels"] = used_channels
    qc["channel_problems"] = problems

    if len(used_channels) < len(expected_channels):
        qc.setdefault("issues", []).append(
            f"channel count mismatch: used={len(used_channels)}, expected={len(expected_channels)}")

    filtered, new_sfreq = preprocess_continuous(raw, session.sfreq, cfg)

    # CAR re-reference on continuous data, before epoching. Original reference is kept in metadata.
    bad_cont_channel = _continuous_bad_channel(filtered)
    filtered = apply_reref(filtered, cfg.reref, bad_channel_mask=bad_cont_channel)
    if bad_cont_channel.any():
        # Channels with broad-band continuous failure are reported. They are zeroed after
        # CAR (documented repair) and the session is failed by QC if too many channels fail.
        qc["continuous_bad_channels"] = [int(i) for i in np.where(bad_cont_channel)[0]]
        if len(qc["continuous_bad_channels"]) > cfg.max_bad_channels:
            qc.setdefault("issues", []).append(
                f"bad channels {qc['continuous_bad_channels']} exceed gate max={cfg.max_bad_channels}")
        filtered = np.where(np.isnan(filtered), 0.0, filtered)

    # Events were stored at raw sfreq; convert onset sample to downsampled time base.
    events = session.events.copy()
    if not events.empty and not np.isclose(new_sfreq, session.sfreq):
        events["onset_sample"] = np.floor(events["onset_sample"] * new_sfreq / session.sfreq).astype(int)

    X, meta, dropped = epoch_data(filtered, new_sfreq, events, cfg.tmin_s, cfg.tmax_s,
                                  tuple(cfg.baseline_s))
    artifact = detect_artifacts(X, cfg)

    # BrainSync packet status at marker time: 0xff/255 means all leads connected.
    # A trial is marked bad when the device reported lead-off or impedance mode.
    if "leadoff_status" in meta.columns:
        status = pd.to_numeric(meta["leadoff_status"], errors="coerce")
        leadoff_bad = status.notna() & (status.astype(int) != int(cfg.leadoff_normal_value))
        if "is_impedance_mode" in meta.columns:
            impedance_bad = meta["is_impedance_mode"].astype(bool).to_numpy()
            leadoff_bad = leadoff_bad | impedance_bad
        artifact.bad_trial = artifact.bad_trial | leadoff_bad.to_numpy()
        artifact.metrics["n_bad_trials"] = int(artifact.bad_trial.sum())
        artifact.metrics["bad_trial_ratio"] = float(artifact.bad_trial.mean())
        artifact.metrics["n_leadoff_or_impedance_bad_trials"] = int(leadoff_bad.sum())
        qc["leadoff_or_impedance_bad_trials"] = int(leadoff_bad.sum())

    meta["epoch_idx"] = np.arange(len(meta), dtype=int)
    meta["bad_trial"] = artifact.bad_trial.astype(int)
    meta["is_target"] = (meta["number"] == session.target_number).astype(int) \
        if session.target_number is not None else 0
    meta["session_id"] = str(session.metadata.get("session_id", session.source_path.stem))
    meta["target_number"] = session.target_number

    if artifact.metrics["bad_trial_ratio"] > cfg.max_bad_epoch_ratio:
        qc.setdefault("issues", []).append(
            f"bad trial ratio {artifact.metrics['bad_trial_ratio']:.2%} exceeds "
            f"gate {cfg.max_bad_epoch_ratio:.0%}")

    sidecar = {
        "qc": qc,
        "artifact": artifact.metrics,
        "dropped_event_indices": dropped,
        "n_epochs": int(len(meta)),
        "sfreq_after": float(new_sfreq),
        "tmin_s": cfg.tmin_s,
        "tmax_s": cfg.tmax_s,
        "baseline_s": cfg.baseline_s,
        "reref": cfg.reref,
        "original_reference": session.metadata.get("ref_label"),
        "original_ground": session.metadata.get("gnd_label"),
        "qc_pass": not bool(qc.get("issues")),
        "bad_channel_repair": "zeroed after CAR; channels listed in qc.continuous_bad_channels",
    }
    return X, meta, sidecar


def _continuous_bad_channel(data: np.ndarray, max_abs: float = 500.0,
                            flat_max_std: float = 0.01) -> np.ndarray:
    """Detect grossly failed channels on continuous filtered data. Marker only."""
    bad = np.zeros(data.shape[0], dtype=bool)
    with np.errstate(invalid="ignore"):
        bad |= np.nanmax(np.abs(data), axis=1) > max_abs
        bad |= np.nanstd(data, axis=1) < flat_max_std
    return bad


def save_prepared_cache(path: str | Path, X: np.ndarray, meta: pd.DataFrame,
                        sidecar: dict[str, Any]) -> None:
    from .utils import atomic_write_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), X.astype(np.float32))
    meta.to_json(path.with_suffix(".meta.jsonl"), orient="records", lines=True, force_ascii=False)
    atomic_write_json(path.with_suffix(".sidecar.json"), sidecar)
