"""Shared logging helpers for evaluation metric summaries."""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from eval_metrics.uncertainty import compute_variance_summary

logger = logging.getLogger(__name__)


def format_ece(ece_result: Optional[Dict]) -> str:
    if not ece_result:
        return "N/A"
    parts = []
    if "ece_recall" in ece_result:
        values = ece_result["ece_recall"]
        parts.append(
            "ECE_R@"
            + "/".join(str(k) for k in values)
            + ": "
            + "/".join(f"{float(v):.3f}" for v in values.values())
        )
    if "ece_map" in ece_result:
        values = ece_result["ece_map"]
        parts.append(
            "ECE_mAP@"
            + "/".join(str(k) for k in values)
            + ": "
            + "/".join(f"{float(v):.3f}" for v in values.values())
        )
    if "ece_ap" in ece_result:
        parts.append(f"ap={float(ece_result['ece_ap']):.3f}")
    return ", ".join(parts) if parts else str(ece_result)


def log_stats(prefix: str, stats: Dict[str, float], *, debug: bool = False) -> None:
    log_fn = logger.debug if debug else logger.info
    log_fn(
        "  %s: min=%.4g, max=%.4g, mean=%.4g, std=%.4g, median=%.4g",
        prefix,
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["std"],
        stats["median"],
    )


def log_metric_summary(
    metric_name: str,
    ece_result: Optional[Dict[str, Any]],
    scores: np.ndarray,
    auc_pr: Optional[float],
    *,
    ece_only: bool = False,
) -> None:
    """Log summary for a learned uncertainty metric (variances + confidence rows)."""
    logger.info("\033[1m%s\033[0m", metric_name)
    logger.info("  %s", format_ece(ece_result))
    if ece_only:
        return
    score_stats = ece_result.get("score_stats") if isinstance(ece_result, dict) else None
    if score_stats:
        log_stats("variances (after clamp)", score_stats["variances_after_clamp"])
        log_stats("confidence (after clamp)", score_stats["confidence_after_clamp"])
        log_stats("variances (before clamp)", score_stats["variances_before_clamp"], debug=True)
        log_stats("confidence (before clamp)", score_stats["confidence_before_clamp"], debug=True)
    else:
        logger.info(
            "  variances (after clamp): min=%.4g, max=%.4g, mean=%.4g, std=%.4g",
            float(np.min(scores)),
            float(np.max(scores)),
            float(np.mean(scores)),
            float(np.std(scores)),
        )
    if auc_pr is not None:
        logger.info("  AUC-PR: %.4f", auc_pr)


def _panel_variant_names(panel: Dict[str, Any]) -> List[str]:
    """Variant order follows ``ece_recall`` insertion order from combined eval, then any AUC-only keys."""
    names = list(panel.get("ece_recall") or {})
    auc_only = (set(panel.get("auc_pr_norm_per_dataset") or {}) | set(panel.get("auc_pr_norm_global") or {})) - set(names)
    names.extend(sorted(auc_only))
    return names


def log_combined_panel(name: str, panel: Dict[str, Any]) -> None:
    """Log combined ECE and dual-normalization AUC-PR per variant."""
    logger.info("\033[1m%s\033[0m", name)

    recalls = panel.get("recalls") or {}
    if recalls:
        keys = sorted(recalls)
        logger.info(
            "  Recall@%s: %s",
            "/".join(str(k) for k in keys),
            "/".join(f"{float(recalls[k]):.2f}" for k in keys),
        )

    auc_pd = panel.get("auc_pr_norm_per_dataset") or {}
    auc_g = panel.get("auc_pr_norm_global") or {}

    for variant in _panel_variant_names(panel):
        ece_payload: Dict[str, Any] = {}
        if variant in panel.get("ece_recall", {}):
            ece_payload["ece_recall"] = panel["ece_recall"][variant]
        if variant in panel.get("ece_map", {}):
            ece_payload["ece_map"] = panel["ece_map"][variant]
        if variant in panel.get("ece_ap", {}):
            ece_payload["ece_ap"] = panel["ece_ap"][variant]

        logger.info("\033[1m  %s\033[0m", variant)
        if ece_payload:
            logger.info("    %s", format_ece(ece_payload))
        if variant in auc_pd:
            logger.info("    AUC-PR (norm_per_dataset): %.4f", float(auc_pd[variant]))
        if variant in auc_g:
            logger.info("    AUC-PR (norm_global): %.4f", float(auc_g[variant]))


def log_baseline_summary(
    name: str,
    ece_result: Optional[Dict[str, Any]],
    scores: np.ndarray,
    auc_pr: Optional[float] = None,
    *,
    ece_only: bool = False,
) -> None:
    """Log summary for a baseline metric (confidence rows only)."""
    logger.info("\033[1m%s\033[0m", name)
    logger.info("  %s", format_ece(ece_result))
    if ece_only:
        return
    score_stats = ece_result.get("score_stats") if isinstance(ece_result, dict) else None
    if score_stats:
        log_stats("confidence (after clamp)", score_stats["confidence_after_clamp"])
        log_stats("confidence (before clamp)", score_stats["confidence_before_clamp"], debug=True)
    else:
        log_stats("confidence (after clamp)", compute_variance_summary(scores))
        log_stats("confidence (before clamp)", compute_variance_summary(scores), debug=True)
    if auc_pr is not None:
        logger.info("  AUC-PR: %.4f", auc_pr)
