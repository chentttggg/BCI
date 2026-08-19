"""Trial-to-number aggregation and evaluation metrics for the 9-way guess."""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_number_scores(meta: pd.DataFrame, probabilities: np.ndarray,
                            method: str = "mean_logit") -> tuple[dict[int, float], list[int]]:
    """Aggregate per-trial P(target) into one score per displayed number.

    method:
      - ``mean_logit``: mean of log(p/(1-p)), more stable for rare targets (default)
      - ``mean_prob``: arithmetic mean probability
      - ``sum_prob``: sum of probabilities (equivalent ranking to mean_prob)
    """
    eps = 1e-8
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    numbers = np.asarray(meta["number"], dtype=int)
    scores: dict[int, float] = {}
    for n in range(1, 10):
        idx = numbers == n
        if not idx.any():
            scores[n] = -np.inf if method == "mean_logit" else 0.0
            continue
        if method == "mean_logit":
            logits = np.log(probs[idx] / (1.0 - probs[idx]))
            scores[n] = float(np.nanmean(logits))
        elif method == "sum_prob":
            scores[n] = float(np.nansum(probs[idx]))
        else:
            scores[n] = float(np.nanmean(probs[idx]))
    ranking = sorted(scores, key=lambda k: (scores[k], -k), reverse=True)
    return scores, ranking


def block_predictions(meta: pd.DataFrame, probabilities: np.ndarray,
                      method: str = "mean_logit") -> pd.DataFrame:
    """One prediction per (session, block)."""
    rows = []
    for (session_id, block), idx in meta.groupby(["session_id", "block"], sort=True).groups.items():
        sub_meta = meta.loc[idx].reset_index(drop=True)
        scores, ranking = aggregate_number_scores(sub_meta, probabilities[idx], method=method)
        target = sub_meta["target_number"].iloc[0]
        # Bayesian MAP confidence: mean-logit scores are log posterior odds up to a
        # constant, so softmax over the 9 scores gives the approximate P(number|block).
        z = np.asarray([scores.get(n, -np.inf if method == "mean_logit" else 0.0)
                        for n in range(1, 10)], dtype=np.float64)
        z = z - np.nanmax(z)
        p = np.exp(z)
        p = p / p.sum()
        confidence = float(p[np.argmax(p)])
        margin = float(np.sort(z)[-1] - np.sort(z)[-2]) if np.isfinite(z).sum() >= 2 else float("nan")

        rows.append({
            "session_id": session_id,
            "block": block,
            "n_trials": int(len(idx)),
            "target": int(target) if pd.notna(target) else -1,
            "predicted": int(ranking[0]),
            "top3": [int(x) for x in ranking[:3]],
            "scores": scores,
            "confidence": confidence,
            "margin_logit": margin,
            "correct": bool(ranking[0] == target) if pd.notna(target) else None,
        })
    return pd.DataFrame(rows)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """AUC + threshold metrics. AUC is None when a class is missing."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    out: dict[str, float] = {}
    out["n_trials"] = int(len(y_true))
    out["n_target"] = int(y_true.sum())
    out["n_nontarget"] = int((1 - y_true).sum())
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_prob >= 0.5))
    try:
        if len(np.unique(y_true)) == 2:
            out["auc"] = float(roc_auc_score(y_true, y_prob))
        else:
            out["auc"] = float("nan")
    except ValueError:
        out["auc"] = float("nan")
    return out
