import logging
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_variance_summary(variances):
    """Return scalar summary statistics for an array of variances."""
    return {
        "mean": float(np.mean(variances)),
        "min": float(np.min(variances)),
        "max": float(np.max(variances)),
        "std": float(np.std(variances)),
        "median": float(np.median(variances)),
    }


def compute_score_statistics(scores) -> dict:
    """Per-query summary stats for a metric score tensor (ECE / AUC-PR inputs).

    For 2D+ scores (e.g. kappa ``[n_queries, dim]``), uses the mean per query.
    For 1D per-query scores (baselines), uses the vector directly.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return {}

    if arr.ndim >= 2 and arr.shape[0] > 0:
        per_query = np.mean(arr, axis=tuple(range(1, arr.ndim)))
    else:
        per_query = arr.reshape(-1)

    q_stats = compute_variance_summary(per_query.reshape(-1))
    return {
        "q_mean": q_stats["mean"],
        "q_std": q_stats["std"],
        "q_min": q_stats["min"],
        "q_max": q_stats["max"],
    }


def plot_variance_distribution(all_variances, output_dir, num_database=None):
    """
    Save a figure with per-vector mean variance: left = Database, right = Queries.

    Fallback when `num_database` is not provided (or invalid): single panel over
    all vectors combined.
    """
    try:
        import matplotlib.pyplot as plt
        out_path = Path(output_dir).resolve()
        if out_path.exists() and not out_path.is_dir():
            logger.warning("Cannot save variance distribution: %s exists as a file, not a directory.", out_path)
            return
        out_path.mkdir(parents=True, exist_ok=True)

        mean_per_vector = np.mean(all_variances, axis=1)
        has_split = num_database is not None and 0 < num_database < len(mean_per_vector)

        if has_split:
            db_means = mean_per_vector[:num_database]
            q_means = mean_per_vector[num_database:]
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].hist(db_means, bins=40, alpha=0.8, color='green', edgecolor='black')
            axs[0].set_title(f"Database per-vector mean variance (N={len(db_means)})")
            axs[0].set_xlabel("Mean variance σ²")
            axs[0].set_ylabel("Frequency")
            axs[0].grid(True, alpha=0.3)

            axs[1].hist(q_means, bins=40, alpha=0.8, color='coral', edgecolor='black')
            axs[1].set_title(f"Queries per-vector mean variance (N={len(q_means)})")
            axs[1].set_xlabel("Mean variance σ²")
            axs[1].set_ylabel("Frequency")
            axs[1].grid(True, alpha=0.3)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.hist(mean_per_vector, bins=50, alpha=0.8, color='steelblue', edgecolor='black')
            ax.set_title(f"Per-vector mean variance (N={len(mean_per_vector)})")
            ax.set_xlabel("Mean variance σ²")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path / "variance_distribution.png", dpi=150)
        plt.close()
        logger.debug(f"Variance distribution plot saved to {out_path / 'variance_distribution.png'}")
    except ImportError:
        logger.warning("matplotlib not installed, skipping variance distribution plot.")
    except Exception as e:
        logger.warning("Error saving variance distribution plot: %s", e)

def compute_uncertainty_statistics(all_variances, output_dir=None, num_database=None):
    """
    Computes and logs statistics about the uncertainty values, and saves variance distribution plot.
    """
    all_stats = compute_variance_summary(all_variances)

    logger.debug(
        "Variance Statistics (database + query) - Mean: %.4e, Min: %.4e, Max: %.4e, Std: %.4e, Median: %.4e",
        all_stats["mean"],
        all_stats["min"],
        all_stats["max"],
        all_stats["std"],
        all_stats["median"],
    )

    if output_dir:
        plot_variance_distribution(all_variances, output_dir, num_database=num_database)

    stats: dict = {}

    mean_per_vector = np.mean(all_variances, axis=1)
    if num_database is not None and num_database < len(mean_per_vector):
        db_stats = compute_variance_summary(mean_per_vector[:num_database])
        q_stats = compute_variance_summary(mean_per_vector[num_database:])
        stats.update({
            "db_mean": db_stats["mean"],
            "db_std": db_stats["std"],
            "db_min": db_stats["min"],
            "db_max": db_stats["max"],
            "q_mean": q_stats["mean"],
            "q_std": q_stats["std"],
            "q_min": q_stats["min"],
            "q_max": q_stats["max"],
            "q_median": q_stats["median"],
        })
    else:
        vector_stats = compute_variance_summary(mean_per_vector)
        stats.update({
            "mean": vector_stats["mean"],
            "std": vector_stats["std"],
            "min": vector_stats["min"],
            "max": vector_stats["max"],
            "median": vector_stats["median"],
        })
    
    return stats


def compute_auc_pr(
    predictions,
    positives_per_query,
    scores,
    *,
    higher_is_confidence: bool,
    label: str,
) -> float:
    """
    Compute AUC-PR for failure prediction from one score vector.

    This follows the same pattern as SUE's sue.py:
      1. Binary labels: 1 if top-1 retrieval is correct, 0 otherwise.
      2. Convert the input into confidence scores (higher = more confident).
      3. Normalize confidence to [0.0, 0.9999].
      4. precision_recall_curve -> auc.
    """
    total_queries = len(predictions)

    # 1. Binary labels: 1 if top-1 is a ground-truth match.
    matched = np.zeros(total_queries, dtype="float32")
    for i, preds in enumerate(predictions):
        if np.any(np.isin(preds[:1], positives_per_query[i])):
            matched[i] = 1.0

    # 2. Convert to confidence scores. Some inputs are uncertainty values, where
    # lower is better, so negate them before computing precision/recall.
    confidence = np.asarray(scores, dtype=np.float64)
    if not higher_is_confidence:
        confidence = -1 * confidence

    # 3. Normalize to [0.0, 0.9999] (same range as SUE in sue.py).
    confidence = np.interp(confidence, (confidence.min(), confidence.max()), (0.0, 0.9999))

    # 4. Precision-recall curve and AUC.
    precision, recall, _ = precision_recall_curve(matched, confidence)
    return float(auc(recall, precision))