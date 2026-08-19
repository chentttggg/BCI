"""Stage 0/1: raw data inventory, hashing, and integrity check (Constitution Stage 0)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .config import ChannelConfig, PreprocessConfig
from .io import load_session
from .utils import atomic_write_json, configure_logging, sha256_file, utc_now_iso

logger = logging.getLogger("backend.ingest")


def scan_raw_dir(data_dir: str | Path) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".edf", ".bdf", ".jsonl", ".json"}:
            continue
        if "derived" in path.parts:
            continue
        rows.append({
            "path": str(path.resolve()),
            "relative_path": str(path.relative_to(data_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "ingested_utc": utc_now_iso(),
        })
    return rows


def _raw_channel_qc(session) -> dict[str, Any]:
    """Detect flat/railed/duplicated channels directly from the raw session.

    This catches the failure mode seen with BrainSync EDF recordings where
    several channels are bit-identical copies or saturated at the ADC rails.
    """
    import numpy as np

    raw = np.asarray(session.raw, dtype=np.float64)
    n_ch, n = raw.shape
    issues = []
    per_channel = []
    for i, name in enumerate(session.ch_names):
        x = raw[i]
        ptp = float(np.ptp(x)) if n else 0.0
        flat_frac = float(np.mean(np.diff(x) == 0)) if n > 1 else 1.0
        lo, hi = float(np.min(x)), float(np.max(x))
        rail_frac = float(np.mean((x <= lo + 1e-9) | (x >= hi - 1e-9))) if n else 1.0
        per_channel.append({
            "index": int(i), "label": str(name), "min": lo, "max": hi,
            "ptp": ptp, "flat_frac": flat_frac, "rail_frac": rail_frac,
        })
        if flat_frac > 0.20:
            issues.append(f"{name} flat_frac={flat_frac:.2%}")
        if rail_frac > 0.20:
            issues.append(f"{name} rail_frac={rail_frac:.2%}")
    # Exact duplicate detection.
    groups: dict[str, list[str]] = {}
    for i, name in enumerate(session.ch_names):
        key = raw[i].tobytes()
        groups.setdefault(key, []).append(str(name))
    duplicate_groups = [v for v in groups.values() if len(v) > 1]
    if duplicate_groups:
        issues.append(f"exact-duplicate channels: {duplicate_groups}")
    return {"per_channel": per_channel, "distinct_channels": len(groups),
            "duplicate_groups": duplicate_groups, "issues": issues}


def validate_sessions(data_dir: str | Path, channel_cfg: ChannelConfig,
                      pre_cfg: PreprocessConfig) -> list[dict[str, Any]]:
    reports = []
    for row in scan_raw_dir(data_dir):
        if not row["path"].lower().endswith((".edf", ".bdf")):
            continue
        try:
            session = load_session(row["path"])
            n_events = len(session.events)
            expected = sorted([c.upper() for c in channel_cfg.channels])
            actual = sorted([c.upper() for c in session.ch_names])
            channel_ok = actual == expected
            raw_qc = _raw_channel_qc(session)
            issues = [] if channel_ok else [f"channel labels {actual} != {expected}"]
            if n_events == 0:
                issues.append("no stim events")
            issues.extend(raw_qc["issues"])
            report = {
                "edf": row["relative_path"],
                "sha256": row["sha256"],
                "sfreq": session.sfreq,
                "n_channels": session.n_channels,
                "n_samples": session.n_samples,
                "n_stim_events": n_events,
                "target_number": session.target_number,
                "channel_labels_ok": channel_ok,
                "expected_sfreq": pre_cfg.raw_sfreq,
                "sample_rate_ok": bool(abs(session.sfreq - pre_cfg.raw_sfreq) < 0.5),
                "raw_channel_qc": raw_qc,
                "issues": issues,
            }
        except Exception as exc:
            report = {"edf": row["relative_path"], "sha256": row["sha256"],
                      "read_error": str(exc), "issues": [str(exc)]}
        reports.append(report)
    return reports


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Raw data inventory and integrity check")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--manifest", default="data/manifest.jsonl")
    p.add_argument("--channel-config", default="config/channel_config.json")
    p.add_argument("--preprocess-config", default="config/preprocessing.json")
    args = p.parse_args(argv)
    configure_logging(logging.INFO)

    channel_cfg = ChannelConfig.from_json(args.channel_config)
    pre_cfg = PreprocessConfig.load(args.preprocess_config)
    inventory = scan_raw_dir(args.data_dir)
    reports = validate_sessions(args.data_dir, channel_cfg, pre_cfg)

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest, {"generated_utc": utc_now_iso(),
                                 "files": inventory,
                                 "session_reports": reports})
    bad = [r for r in reports if r.get("issues")]
    logger.info("Manifest written: %s (%d files, %d sessions with issues)",
                manifest, len(inventory), len(bad))
    if bad:
        logger.warning("Do not proceed past QC until these are resolved: %s",
                       json.dumps(bad, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
