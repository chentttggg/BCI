"""Load raw sessions (EDF+/BDF or synthetic NPZ) and event logs into memory.

Physical-unit policy
--------------------
- EDF signals are read with pyedflib in their stored physical dimension.
  If dimension is not microvolts, values are converted to microvolts.
- ``events.jsonl`` produced by the frontend is authoritative for stimulus onsets
  because it contains the EEG sample index at marker time.  If absent, EDF+
  annotations are used as a fallback.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import read_json

_UV_ALIASES = {"uv", "µv", "μv", "microvolt", "microvolts"}
_MV_ALIASES = {"mv", "millivolt", "millivolts"}


@dataclass
class SessionData:
    raw: np.ndarray
    sfreq: float
    ch_names: list[str]
    events: pd.DataFrame
    target_number: int | None
    metadata: dict[str, Any]
    source_path: Path
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return int(self.raw.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.raw.shape[1])


def _dimension_to_uv_scale(dimension: str | None) -> float:
    if not dimension:
        return 1.0
    d = dimension.strip().lower().replace(" ", "")
    if d in _UV_ALIASES:
        return 1.0
    if d in _MV_ALIASES:
        return 1000.0
    if d in {"v", "volt", "volts"}:
        return 1e6
    # Unknown dimension: keep values unchanged and let QC/user verify.
    return 1.0


def read_edf(path: str | Path) -> tuple[np.ndarray, float, list[str], dict[str, Any]]:
    """Read EDF/BDF into (channels x samples, sfreq, labels, header).

    MNE-Python is the preferred reader (Constitution tech stack); pyedflib is a
    fallback when MNE is unavailable.
    """
    path = Path(path)

    try:
        import mne
        import pyedflib

        # MNE normalises EDF physical units to SI volts; consult the stored
        # physical dimension so the returned array is always microvolts.
        with pyedflib.EdfReader(str(path)) as probe:
            dimension = probe.getPhysicalDimension(0)

        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
        # MNE returns EDF data in SI volts; convert to microvolts for the pipeline.
        data = np.asarray(raw.get_data(), dtype=np.float64) * 1e6
        return data, float(raw.info["sfreq"]), list(raw.ch_names), {
            "reader": "mne",
            "edf_dimension": dimension,
            "annotations": [(a["onset"], a["duration"], a["description"])
                            for a in raw.annotations],
        }
    except Exception as mne_exc:
        logger = logging.getLogger("backend.io")
        logger.debug("MNE read failed (%s); falling back to pyedflib", mne_exc)

    from pyedflib import highlevel

    try:
        signals, signal_headers, header = highlevel.read_edf(str(path), digital=False)
    except Exception as exc:
        raise ValueError(f"Cannot read EDF {path}: {exc}") from exc

    if isinstance(signals, list):
        data = np.vstack([np.asarray(s, dtype=np.float64) for s in signals])
    else:
        data = np.asarray(signals, dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]

    sfreq = float(signal_headers[0].get("sample_frequency", np.nan))
    labels = [str(h.get("label", f"Ch{i}")) for i, h in enumerate(signal_headers)]
    scale = _dimension_to_uv_scale(str(signal_headers[0].get("dimension", "")))
    if not np.isclose(scale, 1.0):
        data = data * scale

    return data, sfreq, labels, {"reader": "pyedflib", "header": header,
                                  "signal_headers": signal_headers}


def _read_edf_annotations(path: str | Path) -> pd.DataFrame:
    """Return EDF+ annotations as an events-like DataFrame."""
    import pyedflib

    path = Path(path)
    with pyedflib.EdfReader(str(path)) as reader:
        if reader.annotations_in_file == 0:
            return pd.DataFrame(columns=["onset_sec", "duration_sec", "type"])
        onsets, durations, descriptions = reader.readAnnotations()
    rows = []
    for onset, dur, desc in zip(onsets, durations, descriptions):
        rows.append({"onset_sec": float(onset), "duration_sec": float(dur), "type": str(desc)})
    return pd.DataFrame(rows)


def read_events_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _parse_stim_number(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value)
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    try:
        return int(s)
    except ValueError:
        return None


def _events_from_df(df: pd.DataFrame, sfreq: float) -> pd.DataFrame:
    """Normalize event records: keep only stim_on rows and derive onset sample."""
    empty = pd.DataFrame(columns=["onset_sample", "onset_sec", "number", "block", "trial", "type"])
    if df.empty:
        return empty
    out = df.copy()
    out = out[out["type"].astype(str).str.startswith("stim_on")].copy()
    if out.empty:
        return empty

    out["number"] = out["type"].map(_parse_stim_number)
    out = out.dropna(subset=["number"]).copy()
    out["number"] = out["number"].astype(int)

    if "onset_sample" not in out.columns:
        out["onset_sample"] = pd.NA
    if "recording_sample" in out.columns:
        out["onset_sample"] = pd.to_numeric(out["recording_sample"], errors="coerce")
    elif "eeg_sample" in out.columns:
        out["onset_sample"] = pd.to_numeric(out["eeg_sample"], errors="coerce")
    if "onset_sec" in out.columns:
        out["onset_sec"] = pd.to_numeric(out["onset_sec"], errors="coerce")
    else:
        out["onset_sec"] = np.nan

    mask = out["onset_sample"].isna()
    out.loc[mask, "onset_sample"] = (out.loc[mask, "onset_sec"] * sfreq).round()
    out = out.dropna(subset=["onset_sample"]).copy()
    out["onset_sample"] = out["onset_sample"].astype(int)

    for col in ["block", "trial"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    # Preserve BrainSync packet-level quality fields when the frontend logged them.
    quality_cols = ["leadoff_status", "trig_in_status", "is_impedance_mode",
                    "packet_seq", "packet_delta_time_us"]
    for col in quality_cols:
        if col in out.columns:
            if col == "is_impedance_mode":
                out[col] = out[col].astype(str).str.lower().isin(["true", "1", "yes"])
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce")

    cols = ["onset_sample", "onset_sec", "number", "block", "trial", "type"]
    timing_cols = ["eeg_sample", "recording_sample", "recording_onset_sec",
                   "edf_annotation_onset_sec", "alignment_source"]
    for col in timing_cols:
        if col in out.columns:
            if col == "alignment_source":
                out[col] = out[col].astype(str)
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    cols += [c for c in timing_cols if c in out.columns]
    cols += [c for c in quality_cols if c in out.columns]
    out = out[cols].sort_values("onset_sample").reset_index(drop=True)
    return out


def _find_session_json(edf_path: Path) -> Path | None:
    stem = edf_path.stem
    candidates = [
        edf_path.with_name(stem + ".session.json"),
        edf_path.with_name(stem.replace("_eeg", "_session") + ".json"),
        edf_path.with_name(stem + "_session.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_session(edf_path: str | Path, events_path: str | Path | None = None,
                 session_json: str | Path | None = None) -> SessionData:
    """Load one session. events_path defaults to ``<edf stem>_events.jsonl``."""
    edf_path = Path(edf_path)

    if edf_path.suffix.lower() == ".npz":
        with np.load(edf_path, allow_pickle=False) as z:
            raw = z["raw"]
            sfreq = float(z["sfreq"])
            ch_names = list(z["ch_names"])
            target = int(z["target_number"]) if "target_number" in z.files else None
        events_df = pd.DataFrame()
        if events_path is not None:
            if Path(events_path).suffix.lower() == ".csv":
                events_df = pd.read_csv(events_path)
            else:
                events_df = read_events_jsonl(events_path)
        events = _events_from_df(events_df, sfreq)
        metadata: dict[str, Any] = {"target_number": target, "synthetic": True}
        return SessionData(raw=raw, sfreq=sfreq, ch_names=ch_names, events=events,
                           target_number=target, metadata=metadata, source_path=edf_path)

    if events_path is None:
        default_events = edf_path.with_name(edf_path.stem + "_events.jsonl")
        events_path = default_events if default_events.exists() else None

    raw, sfreq, ch_names, header_info = read_edf(edf_path)
    if events_path is not None:
        events = _events_from_df(read_events_jsonl(events_path), sfreq)
    else:
        events = pd.DataFrame()
    if events.empty:
        events = _events_from_df(_read_edf_annotations(edf_path), sfreq)

    metadata: dict[str, Any] = {}
    target_number: int | None = None
    if session_json is None:
        session_json = _find_session_json(edf_path)
    if session_json is not None:
        metadata = read_json(session_json)
        target_number = metadata.get("target_number")
        if target_number is None:
            target_number = metadata.get("target")

    if target_number is None:
        for record in metadata.get("markers", []):
            text = str(record)
            if text.startswith("target/"):
                target_number = _parse_stim_number(text)
                if target_number is not None:
                    break

    metadata["source_path"] = str(edf_path)
    return SessionData(raw=raw, sfreq=sfreq, ch_names=ch_names, events=events,
                       target_number=target_number, metadata=metadata,
                       source_path=edf_path, provenance={"edf_header": header_info})


def reorder_channels(data: np.ndarray, src_names: list[str],
                     expected_names: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Reorder source channels to canonical expected order.

    Missing channels are reported, never silently interpolated (Constitution red line).
    """
    src_names = [str(x).strip().upper() for x in src_names]
    expected_names = [str(x).strip().upper() for x in expected_names]
    missing = [c for c in expected_names if c not in src_names]
    extra = [c for c in src_names if c not in expected_names]
    order = [src_names.index(c) for c in expected_names if c in src_names]
    if not order:
        raise ValueError(f"No expected channels found. EDF labels={src_names}, expected={expected_names}")
    used = [expected_names[i] for i in range(len(expected_names)) if expected_names[i] in src_names]
    return data[order, :], used, missing + extra
