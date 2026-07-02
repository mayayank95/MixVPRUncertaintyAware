"""Expected Calibration Error (ECE) for uncertainty-aware VPR.

Bins queries by predicted uncertainty (mean variance), computes recall@K and
mAP@K per bin, and measures how well uncertainty tracks retrieval performance.

A well-calibrated model should have high recall in low-uncertainty bins and low
recall in high-uncertainty bins, with a smooth monotonic relationship.

ECE = Σ (bin_weight * |bin_metric - expected_metric|)  over all bins.

Adapted from: https://github.com/ramdrop/stun
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def vmf_like_uncertainty_loss(uncertainty_loss: str) -> bool:
    """vMF and kappa_ms both use softplus κ where higher = more confident."""
    return str(uncertainty_loss).lower() in ("vmf", "kappa_ms")


def mean_query_uncertainty(
    query_variances: np.ndarray,
    uncertainty_loss: str,
    vmf_kappa_floor: bool = False,
) -> np.ndarray:
    """Per-query scalar uncertainty (higher = more uncertain)."""
    mu = np.mean(query_variances, axis=-1)
    if vmf_like_uncertainty_loss(uncertainty_loss):
        if vmf_kappa_floor:
            mu = np.maximum(mu, 1.0)
        return 1.0 / (mu + 1e-6)
    return mu


# ``percentile_two_sided`` is selected per-call by the caller (via the ``--ece_two_sided``
# config flag for learned heads; baselines always pass False explicitly).
PERCENTILE_CLIP_PCT = 1.0  # clip 1% tail(s): two-sided → [p1, p99]; one-sided → hi at p99


def _get_zoomed_bins(variances: np.ndarray, num_bins: int, zoom_threshold: float = 0.001):
    """Equal-width bins with adaptive zoom to handle long-tailed distributions.

    Iteratively narrows the upper range until the last bin has at least zoom_threshold%
    of the data (default 0.1%), preventing empty high-uncertainty bins from outliers.

    Returns:
        indices: list of arrays, each containing query indices for that bin.
        k: number of zoom steps applied.
    """
    s_min, s_max = np.min(variances), np.max(variances)
    bins_parent = np.linspace(s_min, s_max, num=num_bins)
    k = 0
    while True:
        indices = []
        bins_child = np.linspace(bins_parent[0], bins_parent[-1 - k], num=num_bins)
        for i in range(num_bins - 1):
            left = np.where(variances >= bins_child[i])
            if i != num_bins - 2:
                right = np.where(variances < bins_child[i + 1])
            else:
                right = np.where(variances <= bins_child[i + 1])
            indices.append(np.intersect1d(left[0], right[0]))
        if len(indices[-1]) > int(variances.shape[0] * zoom_threshold) or k == num_bins - 2:
            break
        k += 1
    
    bin_sizes = [len(b) for b in indices]
    logger.debug(f"ECE Binning: zoom_k={k}, bin_sizes={bin_sizes} (threshold={int(variances.shape[0]*zoom_threshold)})")
    
    return indices, k


def _get_percentile_bins(variances: np.ndarray, num_bins: int,
                          two_sided: bool = False,
                          clip_pct: float = PERCENTILE_CLIP_PCT):
    """Percentile bounds + clip, then uniform bins (every sample is assigned).

    Percentiles define bounds ``[lo, hi]`` on the **raw** scores. Each score is then
    ``np.clip``ped into ``[lo, hi]``, and equal-width bins are built on ``[lo, hi]``.
    No sample is omitted: outliers are folded into the lowest/highest bins.

    Args:
        variances: 1D array of scores.
        num_bins: number of bin edges (so num_bins - 1 actual bins).
        two_sided: if True, ``lo``/``hi`` are the ``clip_pct`` and ``100 - clip_pct``
            percentiles (``PERCENTILE_CLIP_PCT`` default 1.0 → 1st and 99th).
        clip_pct: Tail fraction defining the clip window, in percent.
            Default ``PERCENTILE_CLIP_PCT`` (1.0): one-sided uses min–(100−clip_pct);
            two-sided uses clip_pct–(100−clip_pct).

    Returns:
        indices: list of length num_bins - 1, each entry is the array of indices in that bin.
        k: 0 (kept for return-shape compatibility with _get_zoomed_bins).
    """
    if two_sided:
        lo, hi = np.percentile(variances, [clip_pct, 100.0 - clip_pct])
    else:
        lo = float(np.min(variances))
        hi = float(np.percentile(variances, 100.0 - clip_pct))
    if hi <= lo:
        hi = float(np.max(variances))
    v = np.clip(np.asarray(variances, dtype=np.float64), lo, hi)
    clip_stats = {
        "before_min": float(np.min(variances)),
        "before_max": float(np.max(variances)),
        "before_mean": float(np.mean(variances)),
        "before_median": float(np.median(variances)),
        "after_min": float(np.min(v)),
        "after_max": float(np.max(v)),
        "after_mean": float(np.mean(v)),
        "after_median": float(np.median(v)),
        "clip_lo": float(lo),
        "clip_hi": float(hi),
    }
    edges = np.linspace(lo, hi, num=num_bins)
    indices = []
    for i in range(num_bins - 1):
        left = np.where(v >= edges[i])
        if i != num_bins - 2:
            right = np.where(v < edges[i + 1])
        else:
            right = np.where(v <= edges[i + 1])
        indices.append(np.intersect1d(left[0], right[0]))
    bin_sizes = [len(b) for b in indices]
    n = int(variances.shape[0])
    assigned = int(sum(bin_sizes))
    if assigned != n:
        logger.warning(
            "ECE percentile bins: assigned %d/%d samples (unexpected; check for NaNs in uncertainty)",
            assigned, n,
        )
    logger.debug(f"ECE Binning (percentile, two_sided={two_sided}, clip-to-bounds): bin_sizes={bin_sizes} "
                 f"[lo={lo:.4g}, hi={hi:.4g}, clip_pct={clip_pct}%]")
    return indices, 0, clip_stats


def _select_bins(variances: np.ndarray, num_bins: int, bin_mode: str,
                 zoom_threshold: float = 0.001, percentile_two_sided: bool = False):
    """Dispatch helper: pick zoom or percentile binning based on ``bin_mode``."""
    if bin_mode == "percentile":
        return _get_percentile_bins(variances, num_bins, two_sided=percentile_two_sided)
    indices, zoom_k = _get_zoomed_bins(variances, num_bins, zoom_threshold=zoom_threshold)
    return indices, zoom_k, None


def _score_summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def _clamp_scores_for_logging(scores: np.ndarray, clip_stats: Optional[Dict[str, float]]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if not clip_stats:
        return scores
    return np.clip(scores, clip_stats["clip_lo"], clip_stats["clip_hi"])


def _build_score_stats(
    variances_before_clamp: np.ndarray,
    variances_after_clamp: np.ndarray,
    confidence_before_clamp: np.ndarray,
    confidence_after_clamp: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    return {
        "variances_before_clamp": _score_summary(variances_before_clamp),
        "variances_after_clamp": _score_summary(variances_after_clamp),
        "confidence_before_clamp": _score_summary(confidence_before_clamp),
        "confidence_after_clamp": _score_summary(confidence_after_clamp),
    }


def _cal_recall(predictions: np.ndarray, positives_per_query: List, n_values: List[int]) -> np.ndarray:
    """Compute recall@K. Returns array of shape [len(n_values)], values in [0, 100]."""
    recalls = np.zeros(len(n_values))
    num_queries = predictions.shape[0]
    if num_queries == 0:
        return recalls
    for q_idx in range(num_queries):
        for i, n in enumerate(n_values):
            if np.sum(np.isin(predictions[q_idx, :n], positives_per_query[q_idx])) > 0:
                recalls[i:] += 1
                break
    return recalls / num_queries * 100.0


def _cal_apk(positives, prediction, k):
    """Average precision at K for a single query."""
    if len(positives) == 0 or k == 0:
        return 0.0
    if len(prediction) > k:
        prediction = prediction[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(prediction):
        if p in positives and p not in prediction[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(positives), k) * 100.0


def _cal_mapk(predictions, positives_per_query, k):
    """Mean average precision at K across all queries."""
    return np.mean([_cal_apk(pos, pred, k) for pos, pred in zip(positives_per_query, predictions)])


def _bin_pr(predictions: np.ndarray, distances: np.ndarray, positives_per_query: List):
    """Precision-recall curve by sweeping distance threshold. Returns (recalls, precisions) for AP integration."""
    dists_u = np.linspace(np.min(distances[:, 0]), np.max(distances[:, 0]), num=100)
    recalls, precisions = [], []
    for th in dists_u:
        tp = fp = fn = tn = 0
        for q in range(distances.shape[0]):
            if distances[q, 0] < th:
                if np.any(np.isin(predictions[q, 0], positives_per_query[q])):
                    tp += 1
                else:
                    fp += 1
            else:
                if np.any(np.isin(predictions[q, 0], positives_per_query[q])):
                    fn += 1
                else:
                    tn += 1
        if (tp + fn) == 0 or (tp + fp) == 0:
            continue
        recalls.append(tp / (tp + fn))
        precisions.append(tp / (tp + fp))
    return recalls, precisions


def compute_ece(
    predictions: np.ndarray,
    positives_per_query: List,
    query_variances: np.ndarray,
    n_values: List[int] = [1, 5, 10],
    num_bins: int = 11,
    output_dir: Optional[Path] = None,
    metrics: Optional[List[str]] = None,
    distances: Optional[np.ndarray] = None,
    uncertainty_loss: str = "gaussian_nll",
    plot_filename: str = "ece_plot.png",
    zoom_threshold: float = 0.001,
    bin_mode: str = "zoom",
    vmf_kappa_floor: bool = False,
    percentile_two_sided: bool = False,
) -> Dict:
    """Compute Expected Calibration Error for uncertainty-aware retrieval.

    Args:
        predictions: [num_queries, max_k] predicted DB indices per query.
        positives_per_query: list of arrays, ground-truth positive DB indices per query.
        query_variances: [num_queries, D] variance vectors for each query.
        n_values: recall@K values to evaluate.
        num_bins: number of bin edges (actual bins = num_bins - 1).
        output_dir: if provided, save ECE plot here.
        metrics: list of metrics to compute: 'recall', 'map', 'ap'. Default ['recall'].
        distances: [num_queries, max_k] L2 distances per prediction. Required for 'ap'.
        uncertainty_loss: type of uncertainty loss used (e.g., 'gaussian_nll', 'vmf').
        plot_filename: Base filename for calibration curves of the **first** metric in
            ``metrics`` (usually recall → ``ece_plot.png``). Also writes ``{stem}_bins.png``,
            ``{stem}_weights.png``, and extra ``{stem}_map.png`` / ``{stem}_ap.png`` when
            those metrics are included.
        zoom_threshold: Fraction of data required in the last bin for adaptive zoom (default 0.001).
        bin_mode: 'zoom' (legacy, adaptive) or 'percentile' (define ``[lo, hi]`` from percentiles—
            default 1% tails: two-sided p1–p99; one-sided min–p99—then ``np.clip`` scores into that
            interval and form equal-width bins so every query is counted).
        vmf_kappa_floor: when True, clip kappa < 1 to 1 before inversion (vMF only) to
            avoid huge 1/kappa outliers from low-confidence queries.
        percentile_two_sided: when bin_mode='percentile', clip both the low and high tails
            (p_clip and p_{100-clip}) before forming equal-width bins. Default is one-sided
            clipping (high tail only, min to p_{100-clip}). Use when uncertainty scores have
            outliers on both ends.

    Returns:
        dict with ece_recall, ece_map, ece_ap (when included), bin_*.
    """
    metrics = metrics or ["recall"]
    logger.debug(f"compute_ece: Saving {plot_filename} to {output_dir}")
    variances_before_clamp = np.mean(query_variances, axis=-1)
    variances_after_clamp = variances_before_clamp
    mean_var = mean_query_uncertainty(
        query_variances,
        uncertainty_loss,
        vmf_kappa_floor=vmf_kappa_floor,
    )
    if vmf_like_uncertainty_loss(uncertainty_loss):
        variances_after_clamp = np.maximum(variances_before_clamp, 1.0) if vmf_kappa_floor else variances_before_clamp
        
    bin_indices, zoom_k, clip_stats = _select_bins(
        mean_var, num_bins, bin_mode,
        zoom_threshold=zoom_threshold,
        percentile_two_sided=percentile_two_sided,
    )
    mean_var_after_clamp = _clamp_scores_for_logging(mean_var, clip_stats)
    num_actual_bins = num_bins - 1
    num_queries = len(mean_var)

    bin_recalls = np.zeros((num_actual_bins, len(n_values)))
    bin_map = np.zeros((num_actual_bins, len(n_values)))
    bin_ap = np.zeros(num_actual_bins) if "ap" in metrics else None
    bin_weights = np.zeros(num_actual_bins)

    ece_recall = np.zeros(len(n_values))
    ece_map = np.zeros(len(n_values))
    ece_ap = 0.0 if "ap" in metrics else None

    for b, q_in_bin in enumerate(bin_indices):
        if len(q_in_bin) == 0:
            continue
        bin_weights[b] = len(q_in_bin) / num_queries

        bin_preds = predictions[q_in_bin]
        bin_positives = [positives_per_query[i] for i in q_in_bin]

        # Expected performance: linearly decreasing from 1.0 (low uncertainty) to 0.0 (high uncertainty)
        expected = (num_actual_bins - 1 - b) / (num_actual_bins - 1)

        if "recall" in metrics:
            recall_at_n = _cal_recall(bin_preds, bin_positives, n_values)
            bin_recalls[b] = recall_at_n
            for i in range(len(n_values)):
                ece_recall[i] += bin_weights[b] * abs(recall_at_n[i] / 100.0 - expected)

        if "map" in metrics:
            map_at_n = [_cal_mapk(bin_preds, bin_positives, n) for n in n_values]
            bin_map[b] = map_at_n
            for i in range(len(n_values)):
                ece_map[i] += bin_weights[b] * abs(map_at_n[i] / 100.0 - expected)

        if "ap" in metrics and distances is not None:
            bin_dists = distances[q_in_bin]
            recalls_pr, precisions_pr = _bin_pr(bin_preds, bin_dists, bin_positives)
            ap = 0.0
            for j in range(len(recalls_pr) - 1):
                ap += precisions_pr[j] * (recalls_pr[j + 1] - recalls_pr[j])
            bin_ap[b] = ap
            ece_ap += bin_weights[b] * abs(ap - expected)

    if output_dir:
        _plot_ece(bin_recalls, bin_map, bin_ap, bin_weights, bin_indices, n_values, num_actual_bins,
                 output_dir, metrics, plot_filename)

    result = {
        "bin_recalls": bin_recalls,
        "bin_map": bin_map,
        "bin_weights": bin_weights,
        "score_stats": _build_score_stats(
            variances_before_clamp,
            variances_after_clamp,
            mean_var,
            mean_var_after_clamp,
        ),
    }
    if clip_stats:
        result["clip_stats"] = clip_stats
    if "recall" in metrics:
        result["ece_recall"] = {n: e for n, e in zip(n_values, ece_recall)}
    if "map" in metrics:
        result["ece_map"] = {n: e for n, e in zip(n_values, ece_map)}
    if "ap" in metrics:
        result["ece_ap"] = ece_ap
        result["bin_ap"] = bin_ap
    return result


def _get_paper_rcparams():
    """Return matplotlib rcParams for publication-quality fonts matching reference figure."""
    return {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "cm",
        "font.size": 40,
        "axes.titlesize": 40,
        "axes.labelsize": 48,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40,
        "legend.fontsize": 30,
        "legend.frameon": True,
    }


# Calibration curves: thick lines + large markers (readable in slides / downsized figures).
_ECE_DATA_LINEWIDTH = 3.0
_ECE_REF_LINEWIDTH = 2.8
_ECE_MARKERSIZE = 11


def _plot_ece(bin_recalls, bin_map, bin_ap, bin_weights, bin_indices, n_values, num_bins, output_dir, metrics, plot_filename):
    """Save ECE figures as separate PNGs (bin histogram, weights, one file per metric curve)."""
    try:
        import matplotlib.pyplot as plt
        with plt.rc_context(_get_paper_rcparams()):
            out_dir = Path(output_dir)
            stem = Path(plot_filename).stem
            bins_png = f"{stem}_bins.png"
            weights_png = f"{stem}_weights.png"
            x = np.arange(1, num_bins + 1)
            perfect_x = np.array([1, num_bins])
            perfect_y_pct = np.array([100.0, 0.0])
            even_ticks = [t for t in x if t % 2 == 0]

            saved: List[str] = []

            # --- 1. Bin distribution ---
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.bar(x, [len(b) for b in bin_indices])
            ax.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
            ax.set_ylabel("Number of samples")
            ax.set_xticks(even_ticks)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / bins_png, dpi=300)
            plt.close(fig)
            saved.append(bins_png)

            # --- 2. Bin weights ---
            fig_w, ax_w = plt.subplots(figsize=(12, 10))
            ax_w.bar(x, bin_weights)
            ax_w.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
            ax_w.set_ylabel("Bin weight (fraction of queries)")
            ax_w.set_xticks(even_ticks)
            ax_w.grid(True, alpha=0.3)
            fig_w.tight_layout()
            fig_w.savefig(out_dir / weights_png, dpi=300)
            plt.close(fig_w)
            saved.append(weights_png)

            # --- 3. Calibration curves: first metric uses ``plot_filename``; others use ``{stem}_<metric>.png``
            primary_used = False

            if "recall" in metrics:
                outp = out_dir / plot_filename if not primary_used else out_dir / f"{stem}_recall.png"
                fig_r, ax_r = plt.subplots(figsize=(12, 10))
                ax_r.plot(
                    perfect_x,
                    perfect_y_pct,
                    color="grey",
                    linestyle="--",
                    linewidth=_ECE_REF_LINEWIDTH,
                    label="Perfect calibration",
                    zorder=0,
                )
                for i, n in enumerate(n_values):
                    ax_r.plot(
                        x,
                        bin_recalls[:, i],
                        marker="o",
                        markersize=_ECE_MARKERSIZE,
                        linewidth=_ECE_DATA_LINEWIDTH,
                        label=f"R@{n}",
                    )
                ax_r.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
                ax_r.set_ylabel("Recall@K (%)")
                ax_r.set_xticks(even_ticks)
                # ax_r.legend(loc="lower left", prop={"weight": "bold"})
                ax_r.grid(True, alpha=0.3)
                fig_r.tight_layout()
                fig_r.savefig(outp, dpi=300)
                plt.close(fig_r)
                saved.append(outp.name)
                primary_used = True

            if "map" in metrics:
                outp = out_dir / plot_filename if not primary_used else out_dir / f"{stem}_map.png"
                fig_m, ax_m = plt.subplots(figsize=(12, 10))
                ax_m.plot(
                    perfect_x,
                    perfect_y_pct,
                    color="grey",
                    linestyle="--",
                    linewidth=_ECE_REF_LINEWIDTH,
                    label="Perfect calibration",
                    zorder=0,
                )
                for i, n in enumerate(n_values):
                    ax_m.plot(
                        x,
                        bin_map[:, i],
                        marker="o",
                        markersize=_ECE_MARKERSIZE,
                        linewidth=_ECE_DATA_LINEWIDTH,
                        label=f"mAP@{n}",
                    )
                ax_m.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
                ax_m.set_ylabel("mAP@K (%)")
                ax_m.set_xticks(even_ticks)
                ax_m.legend(loc="lower left", prop={"weight": "bold"})
                ax_m.grid(True, alpha=0.3)
                fig_m.tight_layout()
                fig_m.savefig(outp, dpi=300)
                plt.close(fig_m)
                saved.append(outp.name)
                primary_used = True

            if "ap" in metrics and bin_ap is not None:
                outp = out_dir / plot_filename if not primary_used else out_dir / f"{stem}_ap.png"
                fig_a, ax_a = plt.subplots(figsize=(12, 10))
                ax_a.plot(
                    perfect_x,
                    np.array([1.0, 0.0]),
                    color="grey",
                    linestyle="--",
                    linewidth=_ECE_REF_LINEWIDTH,
                    label="Perfect calibration",
                    zorder=0,
                )
                ax_a.plot(
                    x,
                    bin_ap,
                    marker="o",
                    markersize=_ECE_MARKERSIZE,
                    linewidth=_ECE_DATA_LINEWIDTH,
                    label="AP",
                )
                ax_a.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
                ax_a.set_ylabel("AP")
                ax_a.set_xticks(even_ticks)
                ax_a.legend(loc="lower left", prop={"weight": "bold"})
                ax_a.grid(True, alpha=0.3)
                fig_a.tight_layout()
                fig_a.savefig(outp, dpi=300)
                plt.close(fig_a)
                saved.append(outp.name)

            logger.debug("ECE plots saved to %s: %s", out_dir, saved)
    except ImportError:
        logger.warning("matplotlib not installed, skipping ECE plot.")


def _pair_scores_to_uncertainty(pair_scores: np.ndarray, uncertainty_loss: str,
                                vmf_kappa_floor: bool) -> np.ndarray:
    """Map raw pair scores to ``uncertainty`` (higher = more uncertain), shape preserved."""
    ps = pair_scores.astype(np.float64, copy=False)
    if vmf_like_uncertainty_loss(uncertainty_loss):
        # Joint-kappa S / concentration-like scores: higher score → more confident.
        # vmf_kappa_floor applies to κ-like inputs; for resultant S it is usually left False.
        if vmf_kappa_floor:
            ps = np.maximum(ps, 1.0)
        return 1.0 / (ps + 1e-6)
    # L2 / Gaussian-style: larger distance → more uncertain
    return ps


def _pairwise_flat_pool(
    predictions: np.ndarray,
    positives_per_query: List,
    uncertainty_2d: np.ndarray,
    n_rank: int,
):
    """Collect rank positions 0 .. n_rank-1 into flat uncertainty + binary GT-hit labels."""
    n_q, k_max = predictions.shape
    us, ys = [], []
    k_lim = min(n_rank, k_max)
    for q in range(n_q):
        pos = positives_per_query[q]
        for k in range(k_lim):
            us.append(float(uncertainty_2d[q, k]))
            pid = predictions[q, k]
            ys.append(1.0 if np.any(np.isin(pid, pos)) else 0.0)
    return np.asarray(us, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def compute_ece_pairwise(
    predictions: np.ndarray,
    positives_per_query: List,
    pair_scores: np.ndarray,
    n_values: Optional[List[int]] = None,
    num_bins: int = 11,
    output_dir: Optional[Path] = None,
    plot_filename: str = "ece_pairwise.png",
    zoom_threshold: float = 0.001,
    bin_mode: str = "zoom",
    uncertainty_loss: str = "gaussian_nll",
    vmf_kappa_floor: bool = False,
    percentile_two_sided: bool = False,
    metrics: Optional[List[str]] = None,
) -> Dict:
    """Match-level calibration: bin each retrieved pair (Q, R) by U_{Q↔R}.

    For each recall cutoff N in ``n_values``, pool all pairs at ranks 1..N (i.e. columns
    0..N-1), bin them by uncertainty, and measure acc(B_i) = fraction of pairs that are
    ground-truth positives. ECE uses the same linear perfect-calibration target as
    ``compute_ece`` (1.0 at low-uncertainty bins → 0.0 at high).

    Logs::
        Pairwise ECE_R@1/5/10/20: ... (variant: <plot_filename>)

    Returns a dict compatible with ``compute_ece`` (``ece_recall`` / ``ece_map`` keys).
    """
    metrics = metrics or ["recall"]
    n_values = n_values or [1, 5, 10]

    if pair_scores.shape != predictions.shape:
        raise ValueError(
            f"pair_scores shape {pair_scores.shape} must match predictions {predictions.shape}"
        )

    variances_before_clamp = pair_scores.astype(np.float64, copy=False)
    variances_after_clamp = variances_before_clamp
    if vmf_like_uncertainty_loss(uncertainty_loss) and vmf_kappa_floor:
        variances_after_clamp = np.maximum(variances_before_clamp, 1.0)
    U = _pair_scores_to_uncertainty(pair_scores, uncertainty_loss, vmf_kappa_floor)
    num_actual_bins = num_bins - 1

    ece_recall = np.zeros(len(n_values))
    ece_map = np.zeros(len(n_values))

    # For plotting: per–Top-K bin assignments (boundaries differ per pool); curves overlay all R@K.
    bin_indices_per_n: List[Optional[List[np.ndarray]]] = []
    bin_accs_all: List[np.ndarray] = []

    for ni, n in enumerate(n_values):
        u_flat, y_flat = _pairwise_flat_pool(predictions, positives_per_query, U, n)
        t = len(u_flat)
        if t == 0:
            bin_accs_all.append(np.full(num_actual_bins, np.nan))
            bin_indices_per_n.append(None)
            continue

        bin_indices, _, clip_stats = _select_bins(
            u_flat, num_bins, bin_mode,
            zoom_threshold=zoom_threshold,
            percentile_two_sided=percentile_two_sided,
        )
        if ni == 0 and clip_stats:
            first_clip_stats = clip_stats
        if ni == 0:
            variance_before_flat, _ = _pairwise_flat_pool(
                predictions,
                positives_per_query,
                variances_before_clamp,
                n,
            )
            variance_after_flat, _ = _pairwise_flat_pool(
                predictions,
                positives_per_query,
                variances_after_clamp,
                n,
            )
            first_score_stats = _build_score_stats(
                variance_before_flat,
                variance_after_flat,
                u_flat,
                _clamp_scores_for_logging(u_flat, clip_stats),
            )
        bin_indices_per_n.append(bin_indices)

        bin_accs_curve = np.full(num_actual_bins, np.nan)
        for b, idx_in_bin in enumerate(bin_indices):
            if len(idx_in_bin) == 0:
                continue
            w = len(idx_in_bin) / t
            acc = float(np.mean(y_flat[idx_in_bin]))
            bin_accs_curve[b] = acc
            expected = (num_actual_bins - 1 - b) / max(num_actual_bins - 1, 1)
            if "recall" in metrics:
                ece_recall[ni] += w * abs(acc - expected)
            if "map" in metrics:
                ece_map[ni] += w * abs(acc - expected)

        bin_accs_all.append(bin_accs_curve)

    if output_dir and bin_accs_all and any(b is not None for b in bin_indices_per_n):
        _plot_pairwise_ece(
            bin_indices_per_n,
            bin_accs_all,
            n_values,
            num_actual_bins,
            output_dir,
            plot_filename,
        )

    result: Dict = {}
    if "first_clip_stats" in locals():
        result["clip_stats"] = first_clip_stats
    if "first_score_stats" in locals():
        result["score_stats"] = first_score_stats
    if "recall" in metrics:
        result["ece_recall"] = {int(n): float(ece_recall[i]) for i, n in enumerate(n_values)}
    if "map" in metrics:
        result["ece_map"] = {int(n): float(ece_map[i]) for i, n in enumerate(n_values)}
    return result


def _plot_pairwise_ece(
    bin_indices_per_n: List[Optional[List[np.ndarray]]],
    bin_accs_per_topk: List[np.ndarray],
    n_values: List[int],
    num_bins: int,
    output_dir: Path,
    plot_filename: str,
):
    """Save pairwise ECE figures as separate PNGs (matches ``_plot_ece``-style splits).

    - ``plot_filename``: calibration curves (Perfect + R@K lines).
    - ``{stem}_bins.png`` / ``{stem}_weights.png``: counts / weights for the last non-empty Top-K pool
      in ``n_values`` order (same ``_bins`` convention as before; adds ``_weights`` like ``_plot_ece``).
    - ``{stem}_bins_rK.png`` / ``{stem}_weights_rK.png``: one pair-count / weight plot per Top-K ``K``.
    """
    try:
        import matplotlib.pyplot as plt
        with plt.rc_context(_get_paper_rcparams()):
            out_dir = Path(output_dir)
            stem = Path(plot_filename).stem
            bins_filename = f"{stem}_bins.png"
            weights_filename = f"{stem}_weights.png"
            x = np.arange(1, num_bins + 1, dtype=float)
            perfect_x = np.array([1.0, float(num_bins)])
            perfect_y = np.array([1.0, 0.0])
            even_ticks = [int(t) for t in x if int(t) % 2 == 0]

            def _legacy_bins_weights():
                """Prefer last entry in ``n_values`` order with a non-empty pool."""
                for idx in range(len(bin_indices_per_n) - 1, -1, -1):
                    b_list = bin_indices_per_n[idx]
                    if b_list is None:
                        continue
                    t_pairs = sum(len(b) for b in b_list)
                    if t_pairs <= 0:
                        continue
                    counts = [len(b) for b in b_list]
                    weights = [len(b) / t_pairs for b in b_list]
                    return int(n_values[idx]), counts, weights
                return None, None, None

            # --- Per–Top-K distribution images (separate file per N) ---
            for ni, n in enumerate(n_values):
                b_list = bin_indices_per_n[ni] if ni < len(bin_indices_per_n) else None
                if b_list is None:
                    continue
                t_pairs = sum(len(b) for b in b_list)
                if t_pairs <= 0:
                    continue
                counts = [len(b) for b in b_list]
                weights = [len(b) / t_pairs for b in b_list]
                bins_rn = f"{stem}_bins_r{n}.png"
                weights_rn = f"{stem}_weights_r{n}.png"

                fig_b, ax_b = plt.subplots(figsize=(12, 10))
                ax_b.bar(x, counts, width=0.8, align="center")
                ax_b.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
                ax_b.set_ylabel("Number of pairs")
                ax_b.set_xticks(even_ticks)
                ax_b.grid(True, alpha=0.3)
                fig_b.tight_layout()
                fig_b.savefig(out_dir / bins_rn, dpi=300)
                plt.close(fig_b)

                fig_w, ax_w = plt.subplots(figsize=(12, 10))
                ax_w.bar(x, weights, width=0.8, align="center")
                ax_w.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
                ax_w.set_ylabel("Bin weight (fraction of pairs)")
                ax_w.set_xticks(even_ticks)
                ax_w.grid(True, alpha=0.3)
                fig_w.tight_layout()
                fig_w.savefig(out_dir / weights_rn, dpi=300)
                plt.close(fig_w)

            # --- Legacy filenames (W&B / scripts): last non-empty Top-K in n_values order ---
            n_legacy, counts_l, weights_l = _legacy_bins_weights()
            if counts_l is not None and weights_l is not None and n_legacy is not None:
                fig1, ax1 = plt.subplots(figsize=(12, 10))
                ax1.bar(x, counts_l, width=0.8, align="center")
                ax1.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
                ax1.set_ylabel("Number of pairs")
                ax1.set_xticks(even_ticks)
                ax1.grid(True, alpha=0.3)
                fig1.tight_layout()
                fig1.savefig(out_dir / bins_filename, dpi=300)
                plt.close(fig1)

                fig_w0, ax_w0 = plt.subplots(figsize=(12, 10))
                ax_w0.bar(x, weights_l, width=0.8, align="center")
                ax_w0.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
                ax_w0.set_ylabel("Bin weight (fraction of pairs)")
                ax_w0.set_xticks(even_ticks)
                ax_w0.grid(True, alpha=0.3)
                fig_w0.tight_layout()
                fig_w0.savefig(out_dir / weights_filename, dpi=300)
                plt.close(fig_w0)

            # --- Calibration curves (all Top-K on one axis) ---
            fig2, ax2 = plt.subplots(figsize=(12, 10))
            ax2.plot(
                perfect_x,
                perfect_y,
                color="grey",
                linestyle="--",
                linewidth=_ECE_REF_LINEWIDTH,
                label="Perfect calibration",
                zorder=0,
            )
            for i, n in enumerate(n_values):
                if i >= len(bin_accs_per_topk):
                    break
                curve = bin_accs_per_topk[i]
                # Same default color cycle as _plot_ece ("Recall per bin"); do not pass color= so C0,C1,… match R@1,R@5,…
                ax2.plot(
                    x,
                    curve,
                    marker="o",
                    markersize=_ECE_MARKERSIZE,
                    linewidth=_ECE_DATA_LINEWIDTH,
                    linestyle="-",
                    label=f"R@{n}",
                    zorder=1,
                )
            ax2.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
            ax2.set_ylabel("Pair hit rate")
            ax2.set_xticks(even_ticks)
            ax2.set_ylim(-0.02, 1.02)
            # ax2.legend(loc="lower left", prop={"weight": "bold"})
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(out_dir / plot_filename, dpi=300)
            plt.close(fig2)

            logger.debug(
                "Pairwise ECE plots saved under %s (curves: %s; bins/weights per R@K + legacy %s / %s)",
                out_dir,
                plot_filename,
                bins_filename,
                weights_filename,
            )
    except ImportError:
        logger.warning("matplotlib not installed, skipping pairwise ECE plot.")
