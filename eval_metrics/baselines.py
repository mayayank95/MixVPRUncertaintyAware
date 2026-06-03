"""Baseline confidence scores (L2 / PA-score / SUE) for calibration and predication_failure comparsion.

All baselines operate on the same retrieval results (predictions, distances,
ground-truth) so they are directly comparable to the model's own uncertainty
(kappa / variance).

Implementation follows sue.py from the SUE paper exactly.

Baselines implemented:
  - L2 distance (match_scores)
  - PA-score (pa_scores)
  - SUE (sue_scores)
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from eval_metrics.eval_ece_sh import compute_ece, compute_ece_pairwise
from eval_metrics.log_utils import log_baseline_summary
from eval_metrics.uncertainty import compute_auc_pr, compute_variance_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-baseline score extractors
# ---------------------------------------------------------------------------

def _to_l2(distances: np.ndarray) -> np.ndarray:
    """FAISS IndexFlatL2 returns squared L2; convert to L2 (same scale as sue.py)."""
    return np.sqrt(np.maximum(distances, 0))


def compute_l2_scores(distances: np.ndarray) -> Dict[str, np.ndarray]:
    """L2 baseline scores: top-1 distance per query and full top-K pair distances."""
    dists = _to_l2(distances)
    return {"per_query": dists[:, 0], "pair": dists}


def compute_pa_scores(distances: np.ndarray) -> Optional[np.ndarray]:
    """PA-score = d1/d2 per query. Returns None when fewer than 2 retrievals available."""
    dists = _to_l2(distances)
    if dists.shape[1] < 2:
        return None
    return dists[:, 0] / dists[:, 1]


def compute_sue_scores(
    distances: np.ndarray,
    predictions: np.ndarray,
    database_utms: np.ndarray,
    *,
    num_nn: int = 10,
    slope: float = 350.0,
) -> np.ndarray:
    """SUE spatial variance score per query (sue.py lines 95-131)."""
    dists = _to_l2(distances)
    total_queries = len(predictions)
    sue_scores = np.zeros(total_queries)
    weights = np.ones(num_nn)

    for itr in range(total_queries):
        top_preds = predictions[itr][:num_nn]
        nn_poses = database_utms[top_preds]

        for itr2 in range(num_nn):
            weights[itr2] = math.e ** ((-1 * abs(dists[itr][itr2])) * slope)
        weights = weights / sum(abs(weights))

        mean_pose = np.asarray([
            np.average(nn_poses[:, 0], weights=weights),
            np.average(nn_poses[:, 1], weights=weights),
        ])

        variance_lat_lat = 0.0
        variance_lon_lon = 0.0
        for k in range(num_nn):
            diff_lat = min(500, nn_poses[k, 0] - mean_pose[0])
            diff_lon = min(500, nn_poses[k, 1] - mean_pose[1])
            variance_lat_lat += weights[k] * diff_lat ** 2
            variance_lon_lon += weights[k] * diff_lon ** 2

        sue_scores[itr] = (variance_lat_lat + variance_lon_lon) / 2

    return sue_scores


# ---------------------------------------------------------------------------
# General per-baseline evaluation
# ---------------------------------------------------------------------------

def _evaluate_baseline(
    name: str,
    scores: np.ndarray,
    predictions: np.ndarray,
    positives_per_query: List,
    *,
    n_values: Optional[List[int]] = None,
    output_dir=None,
    zoom_threshold: float = 0.001,
    bin_mode: str = "zoom",
    pair_scores: Optional[np.ndarray] = None,
    compute_aucpr: bool = True,
    log_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one baseline: variance stats, ECE, optional pairwise ECE, optional AUC-PR.

    Args:
        name: Short key used in result dict (e.g. "l2", "pa", "sue").
        scores: 1D per-query scores where higher = more uncertain.
        pair_scores: Optional 2D top-K scores for pairwise ECE.
        compute_aucpr: When True, also computes AUC-PR (failure prediction).
        log_name: Pretty name used in the log summary header.
    """
    log_name = log_name or name
    score_stats = compute_variance_summary(scores)
    out: Dict[str, Any] = {f"{name}_score_stats": score_stats}

    auc_pr = None
    if compute_aucpr:
        auc_pr = compute_auc_pr(
            predictions, positives_per_query, scores,
            higher_is_confidence=False, label=log_name,
        )
        out[name] = auc_pr

    ece_result = None
    if n_values:
        ece_result = compute_ece(
            predictions, positives_per_query, scores[:, None],
            n_values=n_values,
            uncertainty_loss="gaussian_nll",
            output_dir=output_dir,
            plot_filename=f"ece_{name}.png",
            zoom_threshold=zoom_threshold,
            bin_mode=bin_mode,
            percentile_two_sided=False,
        )
        out[f"ece_{name}"] = ece_result["ece_recall"]

    log_baseline_summary(log_name, ece_result, scores, auc_pr)

    if pair_scores is not None and n_values:
        ece_pairwise = compute_ece_pairwise(
            predictions, positives_per_query, pair_scores,
            n_values=n_values,
            output_dir=output_dir,
            plot_filename=f"ece_pairwise_{name}.png",
            zoom_threshold=zoom_threshold,
            bin_mode=bin_mode,
            uncertainty_loss="gaussian_nll",
            percentile_two_sided=False,
        )
        out[f"ece_{name}_pairwise"] = ece_pairwise
        log_baseline_summary(f"pairwise_{log_name}", ece_pairwise, pair_scores, ece_only=True)

    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_baselines(
    predictions,
    positives_per_query,
    distances,
    database_utms=None,
    n_values=None,
    output_dir=None,
    zoom_threshold: float = 0.001,
    bin_mode: str = "zoom",
    skip_auc_pr: bool = False,
):
    """Compute baseline confidence methods (AUC-PR + ECE + raw scores), exactly as in sue.py.

    Parameters
    ----------
    predictions : ndarray, shape [num_queries, K]
        Top-K retrieved indices (from FAISS).
    positives_per_query : list of arrays
        Ground-truth positive indices per query.
    distances : ndarray, shape [num_queries, K]
        Squared L2 distances from FAISS. We take sqrt to get L2.
    database_utms : ndarray, shape [num_database, 2]
        UTM coordinates of database images (needed for SUE).
    output_dir : Path or str, optional
        If provided, save ECE plots here.

    Returns
    -------
    dict  {baseline_name: scalar_or_dict}
        Includes AUC-PR values, ECE dictionaries, and raw scores per baseline.
    """
    logger.debug("baselines: Computing for %d queries. output_dir=%s", len(predictions), output_dir)
    compute_aucpr = not skip_auc_pr

    results: Dict[str, Any] = {}
    raw: Dict[str, Optional[np.ndarray]] = {
        "l2": None, "pa": None, "sue": None, "sue_log": None, "l2_pairwise": None,
    }

    # Build (name, log_name, scores, pair_scores, run_aucpr) for each available baseline.
    baseline_specs = []

    l2 = compute_l2_scores(distances)
    raw["l2"] = l2["per_query"]
    raw["l2_pairwise"] = l2["pair"]
    baseline_specs.append(("l2", "L2 distance", l2["per_query"], l2["pair"], compute_aucpr))

    pa = compute_pa_scores(distances)
    if pa is not None:
        raw["pa"] = pa
        baseline_specs.append(("pa", "PA-score", pa, None, compute_aucpr))

    if database_utms is not None:
        sue = compute_sue_scores(distances, predictions, database_utms)
        raw["sue"] = sue
        baseline_specs.append(("sue", "SUE", sue, None, compute_aucpr))
        sue_log = np.log(sue + 1e-10)
        raw["sue_log"] = sue_log
        baseline_specs.append(("sue_log", "SUE-log", sue_log, None, compute_aucpr))
    else:
        logger.warning("No UTM coordinates found; skipping SUE baseline.")

    for name, log_name, scores, pair_scores, run_aucpr in baseline_specs:
        results.update(_evaluate_baseline(
            name, scores, predictions, positives_per_query,
            n_values=n_values, output_dir=output_dir,
            zoom_threshold=zoom_threshold, bin_mode=bin_mode,
            pair_scores=pair_scores, compute_aucpr=run_aucpr,
            log_name=log_name,
        ))

    if compute_aucpr and raw.get("l2_pairwise") is not None:
        pair_top1 = np.asarray(raw["l2_pairwise"], dtype=np.float64)[:, 0]
        results["l2_pairwise"] = compute_auc_pr(
            predictions,
            positives_per_query,
            pair_top1,
            higher_is_confidence=False,
            label="pairwise_l2",
        )

    positive_rate = float(np.mean([
        float(np.any(np.isin(predictions[i][:1], positives_per_query[i])))
        for i in range(len(predictions))
    ]))
    results["random_baseline"] = positive_rate
    results["raw_scores"] = raw

    logger.debug("Baselines done (positive rate / random baseline ~= %.3f)", positive_rate)

    return results
