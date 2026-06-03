"""Pooled failure-prediction PR-AUC helpers for combined multi-dataset eval."""
from typing import List

import numpy as np
from sklearn.metrics import auc, precision_recall_curve


def pooled_pr_auc_failure_prediction(
    matched_top1: np.ndarray,
    confidence_higher_is_better: np.ndarray,
) -> float:
    """Failure-prediction PR-AUC after global min–max on concatenated scores."""
    y = np.asarray(matched_top1, dtype=np.float64).reshape(-1)
    s = np.asarray(confidence_higher_is_better, dtype=np.float64).reshape(-1)
    if y.size == 0 or s.size != y.size:
        return float("nan")
    lo, hi = float(np.min(s)), float(np.max(s))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")
    if hi <= lo + 1e-15:
        s_norm = np.full_like(s, 0.49995)
    else:
        s_norm = np.interp(s, (lo, hi), (0.0, 0.9999))
    precision, recall, _ = precision_recall_curve(y, s_norm)
    return float(auc(recall, precision))


def _minmax_confidence_to_unit_interval(s: np.ndarray) -> np.ndarray:
    """Map raw confidence (higher = more certain) to [0, 0.9999]; constant vector → midpoint."""
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return s
    lo, hi = float(np.min(s)), float(np.max(s))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.full_like(s, 0.49995)
    if hi <= lo + 1e-15:
        return np.full_like(s, 0.49995)
    return np.interp(s, (lo, hi), (0.0, 0.9999))


def pooled_pr_auc_failure_prediction_per_dataset(
    shards_matched: List[np.ndarray],
    shards_confidence: List[np.ndarray],
) -> float:
    """Pool queries from many datasets: min–max within each dataset, then one PR-AUC."""
    if len(shards_matched) != len(shards_confidence) or not shards_matched:
        return float("nan")
    parts_y: List[np.ndarray] = []
    parts_s: List[np.ndarray] = []
    for y, s in zip(shards_matched, shards_confidence):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        if y.size == 0 or y.size != s.size:
            return float("nan")
        parts_y.append(y)
        parts_s.append(_minmax_confidence_to_unit_interval(s))
    y_all = np.concatenate(parts_y)
    s_all = np.concatenate(parts_s)
    precision, recall, _ = precision_recall_curve(y_all, s_all)
    return float(auc(recall, precision))


def combined_failure_prediction_pr_auc(
    norm_mode: str,
    shards_matched: List[np.ndarray],
    shards_confidence: List[np.ndarray],
) -> float:
    """Combined PR-AUC: ``norm_mode`` is ``per_dataset`` or ``global``."""
    mode = (norm_mode or "per_dataset").strip().lower()
    if mode == "global":
        if not shards_matched:
            return float("nan")
        y = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in shards_matched])
        s = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in shards_confidence])
        return pooled_pr_auc_failure_prediction(y, s)
    return pooled_pr_auc_failure_prediction_per_dataset(shards_matched, shards_confidence)
