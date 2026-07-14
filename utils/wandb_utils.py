"""W&B helpers for training and evaluation: epoch/dataset logging and run teardown."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _import_wandb():
    """Import wandb on demand so eval/train can run without it when --use_wandb is off."""
    import wandb

    return wandb


def _require_wandb():
    try:
        return _import_wandb()
    except ImportError as exc:
        raise ImportError(
            "wandb is required when --use_wandb is set. Install with: pip install wandb"
        ) from exc


def _normalize_losses_for_name(losses: Any) -> list:
    """Parse losses into lowercase tokens (e.g. ['ce', 'uncertainty'])."""
    if losses is None:
        return ["ce"]
    if isinstance(losses, list):
        return [str(x).strip().lower() for x in losses if str(x).strip()]
    s = str(losses).strip().lower()
    if not s:
        return ["ce"]
    return [x.strip() for x in s.replace(" ", "").split(",") if x.strip()]


_UNCERTAINTY_LOSS_NAME_SHORT = {
    "gaussian_nll": "gnll",
    "gaussian_cosine": "gcos",
    "vmf": "vmf",
    "kappa_ms": "kms",
}

_EARLY_STOP_NAME_SHORT = {
    "ece_recall_01": "es_ece1",
    "ece_recall_05": "es_ece5",
    "ece_recall_10": "es_ece10",
}


def _uncertainty_loss_tag(uncertainty_loss: str) -> str:
    key = str(uncertainty_loss or "vmf").strip().lower().replace(" ", "_")
    return _UNCERTAINTY_LOSS_NAME_SHORT.get(key, key)


def _default_wandb_run_name(args: Dict[str, Any], job_type: str) -> str:
    """Build default run name from job_type, var_head, losses and optional flags."""
    loss_tokens = _normalize_losses_for_name(args.get("losses"))

    parts = [job_type]

    # var_head (only if uncertainty mode is enabled)
    if args.get("model_mode") == "uncertainty":
        head_type = str(args.get("var_head_type", "descriptor")).lower()
        lin = str(args.get("var_head_linear", "d")).lower()
        if head_type == "gem":
            parts.append("gem-lin1")
        elif head_type == "agg":
            parts.append(f"agg-lin{lin}")
        else:
            parts.append(f"noagg-lin{lin}")

    # Loss tags: "ce" if cross-entropy is active; uncertainty loss type if uncertainty is active
    if "ce" in loss_tokens:
        parts.append("ce")
    if "uncertainty" in loss_tokens:
        parts.append(_uncertainty_loss_tag(args.get("uncertainty_loss", "gaussian_nll")))

    # Early stopping: omit default (recall only); tag non-default / multi-metric.
    es_list = args.get("early_stop_metrics")
    if not es_list:
        es_list = ["recall"]
    if len(es_list) == 1 and es_list[0] == "recall":
        pass
    elif len(es_list) == 1:
        m = es_list[0]
        parts.append(_EARLY_STOP_NAME_SHORT.get(m, f"es_{m}"))
    else:
        parts.append(f"es_x{len(es_list)}")

    # if init_var flag on
    if args.get("var_init"):
        parts.append("init_var")

    if args.get("load_classifiers"):
        parts.append("load_clf")
    if args.get("freeze_model"):
        parts.append("freeze_model")
    if args.get("resume_train"):
        parts.append("resume_train")

    # GNLL mean scaling: only for gaussian_nll; omit default (sqrt_dim).
    if (
        "uncertainty" in loss_tokens
        and _uncertainty_loss_tag(args.get("uncertainty_loss", "gaussian_nll")) == "gnll"
    ):
        gnll_mode = str(args.get("gnll_mu_scale_mode", "sqrt_dim")).lower()
        if gnll_mode == "custom":
            gnll_val = args.get("gnll_mu_scale_value", 1.0)
            parts.append(f"gnll_mu_custom{gnll_val}")
        elif gnll_mode == "none":
            parts.append("gnll_mu_none")

    # Add BN freeze tag only when the flag is enabled.
    if args.get("freeze_batchnorm"):
        parts.append("frzbn")
    if args.get("phased_early_stop"):
        parts.append("phased_es")
    return "_".join(parts)


def init_wandb(args: Dict[str, Any], job_type: str = "train") -> bool:
    """Initialize Weights & Biases run if use_wandb is enabled. Returns True if initialized."""
    if not args.get("use_wandb"):
        return False
    wandb = _require_wandb()
    run_name = args.get("wandb_run_name") or _default_wandb_run_name(args, job_type)
    if isinstance(run_name, Path):
        run_name = run_name.name
    wandb.init(
        project=args.get("wandb_project", "UncertaintyAwareVPR"),
        name=run_name,
        config=dict(args),
        job_type=job_type,
    )
    return True


def log_wandb(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to W&B if a run is active. No-op otherwise."""
    try:
        wandb = _import_wandb()
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def update_wandb_summary(metrics: Dict[str, Any]) -> None:
    """Write values directly to wandb.run.summary without creating history/panels.

    Use for one-shot scalars (e.g. eval-mode metrics) that should appear in the
    runs table and the run's Summary sidebar, but should NOT produce a
    single-point line chart in the workspace. No-op if no run is active.
    """
    try:
        wandb = _import_wandb()
    except ImportError:
        return
    if wandb.run is None or not metrics:
        return
    for k, v in metrics.items():
        wandb.run.summary[k] = v


def save_wandb_logs(log_dir: Optional[str]) -> None:
    """Upload debug.log and info.log from log_dir to the current W&B run. Call before wandb.finish()."""
    try:
        wandb = _import_wandb()
    except ImportError:
        return
    if wandb.run is None or not log_dir:
        return
    log_path = Path(log_dir)
    for name in ("debug.log", "info.log"):
        p = log_path / name
        if p.exists():
            wandb.save(str(p), base_path=str(log_path), policy="end")


def _recall_values(cfg: Dict[str, Any]) -> List[int]:
    return cfg.get("recall_values", [1, 5, 10, 20])


def _recall_key(k: int) -> str:
    """Format recall_01, recall_05, ... so all recalls group together in W&B."""
    return f"recall_{k:02d}"


def _map_key(k: int) -> str:
    """Format map_01, map_05, ... so all maps group together in W&B."""
    return f"map_{k:02d}"


def _add_recall_metrics(
    metrics: Dict[str, Any], prefix: str, recalls: np.ndarray, recall_values: List[int]
) -> None:
    """Add recall@k metrics to metrics dict (order: R@1, R@5, R@10, ...). prefix e.g. 'val/' or 'eval/sf_xl/'."""
    for k in sorted(recall_values):
        i = recall_values.index(k)
        if i < len(recalls):
            metrics[f"{prefix}{_recall_key(k)}"] = float(recalls[i])


def _add_map_metrics(
    metrics: Dict[str, Any],
    prefix: str,
    map_at_k: Optional[List[float]],
    recall_values: List[int],
) -> None:
    """Add mAP@k metrics to metrics dict (order: mAP@1, mAP@5, ...). prefix e.g. 'val/' or 'eval/sf_xl/'."""
    if map_at_k is None:
        return
    for k in sorted(recall_values):
        i = recall_values.index(k)
        if i < len(map_at_k):
            metrics[f"{prefix}{_map_key(k)}"] = float(map_at_k[i])


def _merge_images_into_metrics(
    metrics: Dict[str, Any], images: Optional[Dict[str, Path]]
) -> None:
    """Add wandb.Image entries for existing paths. Modifies metrics in place."""
    if not images:
        return
    import wandb as _wandb
    for key, img_path in images.items():
        if isinstance(img_path, list):
            valid_images = [Path(p) for p in img_path if Path(p).exists()]
            if valid_images:
                metrics[key] = [_wandb.Image(str(p)) for p in valid_images]
        else:
            if Path(img_path).exists():
                metrics[key] = _wandb.Image(str(img_path))


def _train_epoch_images_for_sections(images: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map eval images to train panel sections.
    Special cases are mapped to val/ or ece/ for consistent dashboard sections.
    All others are kept with their original prefix (usually Eval_{name}/).
    """
    if not images:
        return {}
    import wandb as _wandb
    out = {}
    for key, path in images.items():
        # Handle list of images (e.g. predictions)
        if isinstance(path, list):
            valid_images = [Path(p) for p in path if Path(p).exists()]
            if valid_images:
                # If we have a generic 'predictions' key from training setup, map it to val/
                # Otherwise keep original prefix if it contains dataset info.
                map_key = "val/predictions" if "predictions" in key and "/" not in key else key
                out[map_key] = [_wandb.Image(str(p)) for p in valid_images]
            continue
            
        p = Path(path).resolve()
        if not p.exists():
            continue
        
        # Standard mappings for the primary validation set plots
        if "variance_distribution" in key and "/" not in key:
            out["val/variance_distribution"] = _wandb.Image(str(p))
        elif "ece_plot" in key and "/" not in key:
            out["ece/ece_plot"] = _wandb.Image(str(p))
        else:
            # Keep original prefix (e.g. Eval_sf_xl/ece_pa)
            out[key] = _wandb.Image(str(p))
    return out


def log_train_epoch(
    cfg: Dict[str, Any],
    epoch_num: int,
    recalls: np.ndarray,
    map_at_k: Optional[List[float]],
    best_val_recall1: float,
    active_losses: List[str],
    epoch_variances: Optional[List[float]],
    epoch_losses: List[float],
    epoch_losses_ce: Optional[List[float]],
    epoch_losses_gnll: Optional[List[float]],
    mean_query_variance: Optional[float],
    std_query_variance: Optional[float],
    min_query_variance: Optional[float],
    max_query_variance: Optional[float],
    eval_wandb_metrics: Dict[str, Any],
    eval_wandb_images: Optional[Dict[str, Path]],
) -> None:
    """Build epoch metrics (scalars + images) and log to W&B. No-op if use_wandb is False.

    Panel ordering (dict insertion order determines initial W&B layout):
      train/  — losses (ce, uncertainty, total), variance stats (min, max, mean, std)
      val/    — losses, recalls, variance stats, variance distribution plot
      ece/    — recalls, ece plot
    """
    if not cfg.get("use_wandb"):
        return
    rv = _recall_values(cfg)
    metrics: Dict[str, Any] = {}
    metrics["epoch"] = epoch_num

    # ── train/ ── losses (individual then total), variance statistics
    if "ce" in active_losses and epoch_losses_ce:
        metrics["train/loss_ce"] = float(np.mean(epoch_losses_ce))
    if "uncertainty" in active_losses and epoch_losses_gnll:
        metrics["train/loss_uncertainty"] = float(np.mean(epoch_losses_gnll))
    if active_losses:
        metrics["train/loss"] = float(np.mean(epoch_losses))
    if "uncertainty" in active_losses and epoch_variances:
        metrics["train/variance_min"] = float(np.min(epoch_variances))
        metrics["train/variance_max"] = float(np.max(epoch_variances))
        metrics["train/variance_mean"] = float(np.mean(epoch_variances))
        metrics["train/variance_std"] = float(np.std(epoch_variances))

    # ── val/ ── losses, recalls, variance statistics, plots
    if "val/loss" in eval_wandb_metrics:
        metrics["val/loss"] = float(eval_wandb_metrics["val/loss"])
    metrics["val/best_recall_01"] = float(best_val_recall1)
    _add_recall_metrics(metrics, "val/", recalls, rv)
    _add_map_metrics(metrics, "val/", map_at_k, rv)
    if min_query_variance is not None:
        metrics["val/variance_min"] = float(min_query_variance)
    if max_query_variance is not None:
        metrics["val/variance_max"] = float(max_query_variance)
    if mean_query_variance is not None:
        metrics["val/variance_mean"] = float(mean_query_variance)
    if std_query_variance is not None:
        metrics["val/variance_std"] = float(std_query_variance)
    epoch_images = _train_epoch_images_for_sections(eval_wandb_images)
    for key, img in epoch_images.items():
        metrics[key] = img

    # ── Forward all other metrics ──
    # Forward anything that starts with 'ece/', 'val/', 'Eval_', or 'uncertainty/'
    # that hasn't been explicitly handled yet.
    _handled = set(metrics.keys()) | {"val/loss_total", "val/loss"}
    for k, v in eval_wandb_metrics.items():
        if k not in _handled:
            metrics[k] = v

    log_wandb(metrics, step=epoch_num)


def _collect_eval_media_for_dataset(
    dataset_name: str,
    eval_wandb_images: Dict[str, Any],
) -> Dict[str, Any]:
    """Collect W&B media metrics for one dataset eval."""
    import wandb as _wandb

    prefix = f"Eval_{dataset_name}"
    media_metrics: Dict[str, Any] = {}

    var_dist_key = f"Eval_{dataset_name}/variance_distribution"
    if var_dist_key in eval_wandb_images:
        p = Path(eval_wandb_images[var_dist_key])
        if p.exists():
            media_metrics[f"{prefix}/variance_distribution"] = _wandb.Image(str(p))

    preds_key = f"Eval_{dataset_name}/predictions"
    if preds_key in eval_wandb_images:
        img_list = eval_wandb_images[preds_key]
        if isinstance(img_list, list):
            valid = [Path(p) for p in img_list if Path(p).exists()]
            if valid:
                media_metrics[f"{prefix}/predictions"] = [_wandb.Image(str(p)) for p in valid]

    return media_metrics


def log_eval_results(
    cfg: Dict[str, Any],
    all_panel_data: List[Dict[str, Any]],
    all_wandb_images: List[Optional[Dict[str, Any]]],
    combined_outputs: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Log all eval W&B outputs after every dataset has been evaluated.

    Console logging stays in ``eval_dataset``; this handles run summary scalars,
    per-dataset media, and cross-dataset summary tables/panels only.
    """
    if not cfg.get("use_wandb") or not all_panel_data:
        return

    from eval_metrics.eval_wandb import (
        build_eval_panel_payload,
        build_eval_summary_html_panels,
        panel_to_wandb_metrics,
    )

    scalar_metrics: Dict[str, Any] = {}
    media_metrics: Dict[str, Any] = {}

    for panel, images in zip(all_panel_data, all_wandb_images):
        scalar_metrics.update(panel_to_wandb_metrics(panel))
        if images:
            media_metrics.update(_collect_eval_media_for_dataset(panel["dataset_name"], images))

    for combined in combined_outputs or []:
        scalar_metrics.update(panel_to_wandb_metrics(combined["panel"]))

    update_wandb_summary(scalar_metrics)

    panel_payload = build_eval_panel_payload(cfg, all_panel_data, combined_outputs)
    media_metrics.update(build_eval_summary_html_panels(cfg, all_panel_data, combined_outputs))
    log_payload = {**panel_payload, **media_metrics}
    if log_payload:
        try:
            wandb = _import_wandb()
        except ImportError:
            wandb = None
        if wandb is not None and wandb.run is not None:
            for key, value in panel_payload.items():
                if key.endswith("/table"):
                    wandb.run.summary[key] = value
        log_wandb(log_payload, step=0)


def log_eval_summary_panels(
    cfg: Dict[str, Any],
    all_panel_data: List[Dict[str, Any]],
    combined_outputs: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Log eval-summary chart panels (deprecated wrapper; prefer log_eval_results)."""
    log_eval_results(cfg, all_panel_data, [None] * len(all_panel_data), combined_outputs)


def log_eval_dataset(
    cfg: Dict[str, Any],
    dataset_name: str,
    recalls: np.ndarray,
    map_at_k: Optional[List[float]],
    mean_variance: Optional[float],
    std_variance: Optional[float],
    min_variance: Optional[float],
    max_variance: Optional[float],
    eval_wandb_metrics: Dict[str, Any],
    eval_wandb_images: Optional[Dict[str, Path]] = None,
) -> None:
    """Log evaluation metrics and images for one dataset to W&B.

    Eval mode is one-shot per dataset, so scalar metrics are written to
    `wandb.run.summary` (visible in the runs table + summary sidebar) and NOT
    through `wandb.log`, which would otherwise create useless single-point
    line charts that clutter the workspace.

    Only media (images and HTML tables) is routed through `wandb.log` so it
    still shows up as media panels in the workspace.

    Panel ordering per dataset (dict insertion order determines initial W&B layout):
      Eval_{name}/  — recalls, ECE values, ECE plot, variance distribution,
                      variance statistics, predictions
    """
    if not cfg.get("use_wandb"):
        return
    import wandb as _wandb

    prefix = f"Eval_{dataset_name}"
    rv = _recall_values(cfg)
    # Scalars → run.summary only (no chart). Media → wandb.log (panel).
    scalar_metrics: Dict[str, Any] = {}
    media_metrics: Dict[str, Any] = {}

    def _html_table(headers, rows):
        style = (
            "style='border-collapse:collapse;width:100%;font-family:monospace;font-size:16px;'"
        )
        th_style = "style='border:1px solid #555;padding:8px;background:#2a2a2a;color:#eee;text-align:center;'"
        td_style = "style='border:1px solid #555;padding:8px;text-align:center;'"
        td_label = "style='border:1px solid #555;padding:8px;text-align:left;font-weight:bold;'"
        html = f"<table {style}><tr>"
        for h in headers:
            html += f"<th {th_style}>{h}</th>"
        html += "</tr>"
        for row in rows:
            html += "<tr>"
            for i, cell in enumerate(row):
                s = td_label if i == 0 else td_style
                if isinstance(cell, float):
                    cell = f"{cell:.2f}"
                html += f"<td {s}>{cell}</td>"
            html += "</tr>"
        html += "</table>"
        return _wandb.Html(html)

    # ── 1. Recalls ──
    ret_headers = ["Metric"] + [f"@{k}" for k in sorted(rv)]
    ret_rows = []
    recall_row = ["Recall"]
    for k in sorted(rv):
        i = rv.index(k)
        val = float(recalls[i]) if i < len(recalls) else 0.0
        recall_row.append(val)
        scalar_metrics[f"{prefix}/{_recall_key(k)}"] = val
    ret_rows.append(recall_row)
    if map_at_k is not None:
        map_row = ["mAP"]
        for k in sorted(rv):
            i = rv.index(k)
            val = float(map_at_k[i]) if i < len(map_at_k) else 0.0
            map_row.append(val)
            scalar_metrics[f"{prefix}/{_map_key(k)}"] = val
        ret_rows.append(map_row)
    # HTML table disabled (option C): numbers are already in scalar columns of
    # the runs table; iframe panels were making the W&B workspace sluggish.
    # media_metrics[f"{prefix}/retrieval_metrics"] = _html_table(ret_headers, ret_rows)

    # ── 2. ECE values ──
    ece_rows = []
    has_ece_recall = any(f"Eval_{dataset_name}/ece_recall_" in k for k in eval_wandb_metrics)
    has_ece_map = any(f"Eval_{dataset_name}/ece_map_" in k for k in eval_wandb_metrics)
    if has_ece_recall:
        row = ["ECE Recall"]
        for k in sorted(rv):
            key = f"Eval_{dataset_name}/ece_recall_{k:02d}"
            val = float(eval_wandb_metrics.get(key, 0.0))
            row.append(val)
            scalar_metrics[key] = val
        ece_rows.append(row)
    if has_ece_map:
        row = ["ECE mAP"]
        for k in sorted(rv):
            key = f"Eval_{dataset_name}/ece_map_{k:02d}"
            val = float(eval_wandb_metrics.get(key, 0.0))
            row.append(val)
            scalar_metrics[key] = val
        ece_rows.append(row)
    ece_ap_key = f"Eval_{dataset_name}/ece_ap"
    if ece_ap_key in eval_wandb_metrics:
        val = float(eval_wandb_metrics[ece_ap_key])
        ece_rows.append(["ECE AP", val])
        scalar_metrics[ece_ap_key] = val
    if ece_rows:
        ece_headers = ["Metric"] + [f"@{k}" for k in sorted(rv)]
        # HTML table disabled (option C): see note above.
        # media_metrics[f"{prefix}/ece_metrics"] = _html_table(ece_headers, ece_rows)

    # ── 3. ECE plot ──
    if eval_wandb_images:
        ece_key = f"Eval_{dataset_name}/ece_plot"
        if ece_key in eval_wandb_images:
            p = Path(eval_wandb_images[ece_key])
            if p.exists():
                media_metrics[f"{prefix}/ece_plot"] = _wandb.Image(str(p))

    # ── 4. Variance distribution plot ──
    if eval_wandb_images:
        var_dist_key = f"Eval_{dataset_name}/variance_distribution"
        if var_dist_key in eval_wandb_images:
            p = Path(eval_wandb_images[var_dist_key])
            if p.exists():
                media_metrics[f"{prefix}/variance_distribution"] = _wandb.Image(str(p))

    # ── 5. Variance statistics ──
    var_headers = []
    var_row = []
    if min_variance is not None:
        var_headers.append("Min")
        var_row.append(float(min_variance))
        scalar_metrics[f"{prefix}/variance_min"] = float(min_variance)
    if max_variance is not None:
        var_headers.append("Max")
        var_row.append(float(max_variance))
        scalar_metrics[f"{prefix}/variance_max"] = float(max_variance)
    if mean_variance is not None:
        var_headers.append("Mean")
        var_row.append(float(mean_variance))
        scalar_metrics[f"{prefix}/variance_mean"] = float(mean_variance)
    if std_variance is not None:
        var_headers.append("Std")
        var_row.append(float(std_variance))
        scalar_metrics[f"{prefix}/variance_std"] = float(std_variance)
    if var_headers:
        # HTML table disabled (option C): see note above.
        # media_metrics[f"{prefix}/variance_statistics"] = _html_table(var_headers, [var_row])
        pass
    # ── 6. Predictions visualization ──
    if eval_wandb_images:
        preds_key = f"Eval_{dataset_name}/predictions"
        if preds_key in eval_wandb_images:
            img_list = eval_wandb_images[preds_key]
            if isinstance(img_list, list):
                valid = [Path(p) for p in img_list if Path(p).exists()]
                if valid:
                    media_metrics[f"{prefix}/predictions"] = [_wandb.Image(str(p)) for p in valid]

    # ── 7. Generic logging for any other images ──
    if eval_wandb_images:
        for key, value in eval_wandb_images.items():
            if key in media_metrics or "predictions" in key or not key.startswith(prefix):
                continue
            if isinstance(value, (str, Path)):
                p = Path(value)
                if p.exists():
                    media_metrics[key] = _wandb.Image(str(p))

    # Forward any remaining scalar metrics not yet handled (assume scalar unless wandb media type)
    _handled = set(scalar_metrics.keys()) | set(media_metrics.keys())
    for k, v in eval_wandb_metrics.items():
        if k in _handled:
            continue
        if isinstance(v, (_wandb.Image, _wandb.Html)) or (isinstance(v, list) and v and isinstance(v[0], _wandb.Image)):
            media_metrics[k] = v
        else:
            scalar_metrics[k] = v

    # Scalars → run.summary (runs table + summary sidebar, no workspace panel).
    update_wandb_summary(scalar_metrics)
    # Media → wandb.log so images/HTML tables still appear as media panels.
    if media_metrics:
        log_wandb(media_metrics)


def finish_run(cfg: Dict[str, Any]) -> None:
    """Upload log files and finish W&B run. No-op if use_wandb is False. Use for both train and eval."""
    if not cfg.get("use_wandb"):
        return
    save_wandb_logs(cfg.get("log_dir"))
    _require_wandb().finish()


def finish_train_run(cfg: Dict[str, Any]) -> None:
    """Alias for finish_run (train entrypoint)."""
    finish_run(cfg)
