"""Generate per-session QC report: PSD, ERP, artifact masks, pass/fail gates."""
from __future__ import annotations

import argparse

from guess_number.paths import default_channel_config_path, default_preprocess_config_path
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as scipy_signal

from .config import ChannelConfig, PreprocessConfig
from .io import load_session
from .preprocess import prepare_session, preprocess_continuous
from guess_number.utils import atomic_write_json, configure_logging, sha256_file, utc_now_iso

logger = logging.getLogger("backend.qc")


def _psd(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    freqs, psd = scipy_signal.welch(data, fs=sfreq, nperseg=min(1024, data.shape[-1]),
                                    axis=-1, detrend="constant")
    return freqs, psd


def make_qc_report(edf: str | Path, output_dir: str | Path,
                   channel_cfg: ChannelConfig, pre_cfg: PreprocessConfig) -> dict[str, Any]:
    edf = Path(edf)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = edf.stem.replace("_eeg", "")
    session = load_session(edf)
    X, meta, sidecar = prepare_session(session, pre_cfg, channel_cfg.channels)

    raw_psd_freqs, raw_psd = _psd(session.raw, session.sfreq)
    filt, filt_sfreq = preprocess_continuous(
        session.raw[:len(channel_cfg.channels), :], session.sfreq, pre_cfg)
    filt_psd_freqs, filt_psd = _psd(filt, filt_sfreq)

    target_idx = meta.index[meta["is_target"] == 1]
    nontarget_idx = meta.index[meta["is_target"] == 0]
    if len(target_idx) == 0 or len(nontarget_idx) == 0:
        raise ValueError("Need both target and non-target epochs for QC report")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    for i, ch in enumerate(channel_cfg.channels):
        ax.semilogy(raw_psd_freqs, raw_psd[i], alpha=0.7, label=ch)
    ax.set_title("Raw PSD (0-80 Hz)")
    ax.set_xlim(0, min(80, session.sfreq / 2 - 1))
    ax.set_xlabel("Hz")
    ax.set_ylabel("uV^2/Hz")
    ax.legend(fontsize=7, ncol=4)

    ax = axes[0, 1]
    for i, ch in enumerate(channel_cfg.channels):
        ax.semilogy(filt_psd_freqs, filt_psd[i], alpha=0.7, label=ch)
    ax.set_title("After HP/notch/LP + downsample PSD")
    ax.set_xlim(0, min(80, filt_sfreq / 2 - 1))
    ax.set_xlabel("Hz")
    ax.set_ylabel("uV^2/Hz")
    ax.legend(fontsize=7, ncol=4)

    time_axis = np.arange(X.shape[-1]) / pre_cfg.downsample_sfreq + pre_cfg.tmin_s
    pz_idx = channel_cfg.channels.index("Pz") if "Pz" in channel_cfg.channels else 3
    ax = axes[1, 0]
    ax.plot(time_axis, X[target_idx, pz_idx, :].mean(axis=0), color="red", label="target")
    ax.plot(time_axis, X[nontarget_idx, pz_idx, :].mean(axis=0), color="gray", label="non-target")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"Pz ERP ({len(target_idx)} target / {len(nontarget_idx)} non-target epochs)")
    ax.set_xlabel("s")
    ax.set_ylabel("uV (CAR)")
    ax.legend()

    ax = axes[1, 1]
    img = X[target_idx, :, :].mean(axis=0)
    vmax = np.nanpercentile(np.abs(img), 99) or 1.0
    im = ax.imshow(img, aspect="auto", origin="lower",
                   extent=[pre_cfg.tmin_s, pre_cfg.tmax_s, -0.5, len(channel_cfg.channels) - 0.5],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(channel_cfg.channels)))
    ax.set_yticklabels(channel_cfg.channels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Target ERP image")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    png = output_dir / (stem + "_qc.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>QC {stem}</title></head>
<body><h2>{stem}</h2><img src="{png.name}" width="1000"><pre>{json.dumps(sidecar, ensure_ascii=False, indent=2)}</pre>
</body></html>"""
    (output_dir / (stem + "_qc.html")).write_text(html, encoding="utf-8")

    report = {
        "edf": str(edf.resolve()),
        "edf_sha256": sha256_file(edf),
        "target_number": session.target_number,
        "n_epochs": int(len(meta)),
        "n_target": int(meta["is_target"].sum()),
        "n_nontarget": int((meta["is_target"] == 0).sum()),
        "bad_trial_ratio": sidecar["artifact"]["bad_trial_ratio"],
        "bad_trials": int(sidecar["artifact"]["n_bad_trials"]),
        "bad_channels": int(sidecar["artifact"]["n_bad_channels"]),
        "qc_issues": sidecar["qc"].get("issues", []),
        "qc_warnings": sidecar["qc"].get("warnings", []),
        "png": str(png),
        "html": str(output_dir / (stem + "_qc.html")),
        "generated_utc": utc_now_iso(),
    }
    atomic_write_json(output_dir / (stem + "_qc.json"), report)
    logger.info("QC report written: %s", report["html"])
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="report for every EDF under this directory")
    p.add_argument("--edf", default=None)
    p.add_argument("--output-dir", default="data/derived/reports")
    p.add_argument("--channel-config", default=default_channel_config_path())
    p.add_argument("--preprocess-config", default=default_preprocess_config_path())
    args = p.parse_args(argv)
    configure_logging(logging.INFO)
    ch = ChannelConfig.from_json(args.channel_config)
    pre = PreprocessConfig.load(args.preprocess_config)
    if args.edf:
        paths = [Path(args.edf)]
    elif args.data_dir:
        paths = sorted(Path(args.data_dir).rglob("*.edf"))
    else:
        raise SystemExit("provide --edf or --data-dir")
    for edf in paths:
        try:
            make_qc_report(edf, args.output_dir, ch, pre)
        except Exception as exc:
            logger.error("QC failed for %s: %s", edf, exc)


if __name__ == "__main__":
    main()
