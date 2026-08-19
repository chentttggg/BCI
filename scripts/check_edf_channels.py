"""Inspect one EDF recording: flat/rail/duplicate channel diagnostics.

Usage:
    python scripts/check_edf_channels.py \
        brainsync-sdk-example-main/python/recording_complete.bdf/streaming_EEG_20260819_182453.edf \
        --plot data/derived/reports/recording_complete_qc.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_edf_signals(path: str | Path) -> tuple[np.ndarray, list[str], float]:
    from pyedflib import EdfReader

    with EdfReader(str(path)) as reader:
        labels = [str(x) for x in reader.getSignalLabels()]
        fs = float(reader.getSampleFrequencies()[0])
        physical = np.stack([reader.readSignal(i, digital=False)
                             for i in range(reader.signals_in_file)]).astype(np.float64)
        digital = np.stack([reader.readSignal(i, digital=True)
                            for i in range(reader.signals_in_file)]).astype(np.int64)
    return physical, digital, labels, fs


def channel_qc(physical: np.ndarray, digital: np.ndarray,
               labels: list[str]) -> dict[str, Any]:
    per_channel = []
    for i, name in enumerate(labels):
        xp = physical[i]
        xd = digital[i]
        n = len(xp)
        per_channel.append({
            "index": int(i),
            "label": name,
            "mean_uv": float(np.mean(xp)),
            "std_uv": float(np.std(xp)),
            "min_uv": float(np.min(xp)),
            "max_uv": float(np.max(xp)),
            "ptp_uv": float(np.ptp(xp)),
            "flat_frac": float(np.mean(np.diff(xd) == 0)) if n > 1 else 1.0,
            "rail_frac": float(np.mean((xd <= np.min(xd)) | (xd >= np.max(xd)))) if n else 1.0,
        })

    groups: dict[bytes, list[str]] = {}
    for i, name in enumerate(labels):
        groups.setdefault(physical[i].tobytes(), []).append(name)
    duplicate_groups = [v for v in groups.values() if len(v) > 1]
    dead = [c["label"] for c in per_channel if c["flat_frac"] > 0.20]
    railed = [c["label"] for c in per_channel if c["rail_frac"] > 0.20]
    return {
        "per_channel": per_channel,
        "distinct_channels": int(len(groups)),
        "duplicate_groups": duplicate_groups,
        "flat_or_near_flat_channels": dead,
        "railed_channels": railed,
        "verdict": "OK" if len(duplicate_groups) == 0 and not dead and not railed else "BAD",
    }


def make_plot(physical: np.ndarray, labels: list[str], fs: float,
              out_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = physical.shape[1]
    t = np.arange(n) / fs
    fig, axes = plt.subplots(8, 1, figsize=(16, 13), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, physical[i], lw=0.6)
        ax.set_ylabel(labels[i], fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-6000, 6000)
    axes[0].set_title(f"{Path(out_path).stem} - EDF waveform QC")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="EDF channel flat/rail/duplicate QC")
    p.add_argument("edf")
    p.add_argument("--plot", default=None)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    physical, digital, labels, fs = read_edf_signals(args.edf)
    report = channel_qc(physical, digital, labels)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.plot:
        make_plot(physical, labels, fs, args.plot)
        print(f"plot: {args.plot}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
