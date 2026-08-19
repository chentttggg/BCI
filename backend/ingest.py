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
                "issues": [] if (channel_ok and n_events > 0) else
                    ([f"channel labels {actual} != {expected}"] if not channel_ok else []) +
                    (["no stim events"] if n_events == 0 else []),
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
