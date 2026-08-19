"""Run a trained ShallowConvNet ensemble on a new session and guess the number."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ChannelConfig
from .io import load_session
from .model import ModelBundle
from .preprocess import prepare_session
from .scoring import aggregate_number_scores, block_predictions
from .utils import atomic_write_json, configure_logging, sha256_file, utc_now_iso

logger = logging.getLogger("backend.predict")


def process_session(edf: str | Path, bundle: ModelBundle, channel_cfg: ChannelConfig,
                    events_path: str | Path | None = None,
                    session_json: str | Path | None = None) -> dict[str, Any]:
    session = load_session(edf, events_path, session_json)
    X, meta, sidecar = prepare_session(session, bundle.preprocess_cfg, channel_cfg.channels)
    if not sidecar.get("qc_pass", True):
        raise RuntimeError(f"QC gate failed: {sidecar.get('qc', {}).get('issues')}")

    # The model was trained on the same canonical channel order.
    if tuple(ch.upper() for ch in channel_cfg.channels) != tuple(bundle.channels):
        raise ValueError(
            f"Model channels {bundle.channels} do not match config "
            f"{[c.upper() for c in channel_cfg.channels]}")

    probs = bundle.predict_proba(X)
    meta = meta.copy()
    meta["prob_target"] = probs
    good = (~meta["bad_trial"].astype(bool)).to_numpy()
    probs_masked = np.where(good, probs, np.nan)

    # Score using only non-artifact trials.
    valid_idx = meta.index[good]
    if len(valid_idx) == 0:
        raise ValueError("All trials were rejected by artifact QC; inspect raw session manually")
    valid_meta = meta.loc[valid_idx].reset_index(drop=True)
    scores, ranking = aggregate_number_scores(valid_meta, probs_masked[good], method="mean_logit")
    scores_prob, ranking_prob = aggregate_number_scores(valid_meta, probs_masked[good], method="mean_prob")
    block_df = block_predictions(valid_meta, probs_masked[good], method="mean_logit")
    block_records = []
    for _, row in block_df.iterrows():
        block_records.append({
            "session_id": row["session_id"],
            "block": row["block"],
            "n_trials": row["n_trials"],
            "target": row["target"],
            "predicted": row["predicted"],
            "top3": row["top3"],
            "scores": row["scores"],
            "correct": row["correct"],
        })

    trial_predictions = []
    for i, row in meta.iterrows():
        trial_predictions.append({
            "trial": int(row["trial"]) if pd.notna(row["trial"]) else None,
            "block": int(row["block"]) if pd.notna(row["block"]) else None,
            "number": int(row["number"]),
            "is_target": bool(row["is_target"]) if pd.notna(row["is_target"]) else None,
            "prob_target": float(probs[i]) if not np.isnan(probs[i]) else None,
            "excluded_by_artifact": bool(row["bad_trial"]),
        })

    result = {
        "source_edf": str(Path(edf).resolve()),
        "source_edf_sha256": sha256_file(edf),
        "target_number_truth": session.target_number,
        "prediction": int(ranking[0]),
        "prediction_alt_prob_mean": int(ranking_prob[0]),
        "top3": [int(x) for x in ranking[:3]],
        "scores_mean_logit": {str(k): v for k, v in scores.items()},
        "scores_mean_prob": {str(k): v for k, v in scores_prob.items()},
        "block_predictions": block_records,
        "n_trials_total": int(len(meta)),
        "n_trials_used": int(good.sum()),
        "n_trials_rejected": int((~good).sum()),
        "correct": bool(ranking[0] == session.target_number) if session.target_number is not None else None,
        "preprocess_sidecar": sidecar,
        "trial_predictions": trial_predictions,
        "model_dir": str(bundle.extra.get("model_dir", "")),
        "generated_utc": utc_now_iso(),
    }
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Predict the thought number for one session")
    p.add_argument("--edf", required=True)
    p.add_argument("--events", default=None)
    p.add_argument("--session-json", default=None)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--channel-config", default="config/channel_config.json")
    p.add_argument("--output-json", default=None)
    p.add_argument("--log-file", default=None)
    args = p.parse_args(argv)
    configure_logging(logging.INFO, args.log_file)

    bundle = ModelBundle.load(args.model_dir)
    bundle.extra["model_dir"] = str(Path(args.model_dir).resolve())
    channel_cfg = ChannelConfig.from_json(args.channel_config)
    result = process_session(args.edf, bundle, channel_cfg, args.events, args.session_json)

    out_path = Path(args.output_json) if args.output_json else \
        Path(args.edf).with_name(Path(args.edf).stem + "_prediction.json")
    atomic_write_json(out_path, result)
    print(json.dumps({
        "prediction": result["prediction"],
        "top3": result["top3"],
        "correct": result["correct"],
        "scores": result["scores_mean_logit"],
        "output": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
