"""Build W&B metric dicts and collect plot paths from per-dataset eval outputs."""
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval_metrics.uncertainty import compute_score_statistics

logger = logging.getLogger(__name__)

# Uncertainty ECE variants used for calibration tables and bins-distribution panels.
UNCERTAINTY_ECE_VARIANTS = (
    ("kappa", "kappa"),
    ("joint_kappa", "joint_kappa"),
    ("pairwise_joint_kappa", "pairwise"),
)

# Baseline ECE bin plots (``ece_{name}_bins.png``, ``ece_pairwise_l2_bins.png``).
BASELINE_ECE_VARIANTS = (
    ("l2", "l2"),
    ("l2_pairwise", "pairwise_l2"),
    ("pa", "pa"),
    ("sue", "sue"),
    ("sue_log", "sue_log"),
)

BASELINE_BINS_VARIANTS = BASELINE_ECE_VARIANTS

# Rows for ``calibration_ece`` summary (learned + baseline ECE_R@k).
CALIBRATION_ECE_VARIANTS = UNCERTAINTY_ECE_VARIANTS + BASELINE_ECE_VARIANTS

# Fixed 7-slot ECE curves panel per dataset (requested dashboard layout).
ECE_CURVES_PANEL_VARIANTS = (
    "kappa",
    "joint_kappa",
    "pairwise_joint_kappa",
    "l2",
    "pairwise_l2",
    "pa",
    "sue",
    "sue_log",
)

# Training val loop: learned uncertainty curves only (baselines are not computed).
UNCERTAINTY_ECE_CURVES_VARIANTS = tuple(k for k, _ in UNCERTAINTY_ECE_VARIANTS)

# W&B summary panel prefix and columns (values from panel["auc_pr"]).
AUC_PR_PANEL_KEY = "AUC-PR"

AUC_PR_SUMMARY_COLUMNS = (
    ("kappa", "kappa"),
    ("joint_kappa", "joint_kappa"),
    ("pairwise_joint_kappa", "pairwise_kappa"),
    ("l2", "l2"),
    ("l2_pairwise", "pairwise_l2"),
    ("pa", "pa"),
    ("sue", "sue"),
    ("sue_log", "sue_log"),
)

# Score tensors summarized inside ``uncertainty_stats`` (after descriptor row).
SCORE_STAT_VARIANTS = (
    "kappa",
    "joint_kappa",
    "pairwise_joint_kappa",
    "l2",
    "l2_pairwise",
    "pa",
    "sue",
    "sue_log",
)

SCORE_STAT_LABELS = {
    "pairwise_joint_kappa": "pairwise_kappa",
    "l2_pairwise": "pairwise_l2",
}


def _ece_plot_stem(variant: str) -> str:
    if variant == "l2_pairwise":
        return "ece_pairwise_l2"
    return f"ece_{variant}"


def _recall_value_dict(arr, rv: List[int]) -> Dict[int, float]:
    if arr is None:
        return {}
    return {int(k): float(arr[rv.index(k)]) for k in rv if rv.index(k) < len(arr)}


def _uncertainty_variant_keys(panel: Dict[str, Any]) -> set:
    return {k for k, _ in UNCERTAINTY_ECE_VARIANTS if k in panel.get("ece_recall", {})}


def _build_panel_data(
    dataset_name: str,
    args: Dict[str, Any],
    recalls,
    map_at_k,
    uncertainty_ece_results: Optional[Dict[str, Any]],
    uncertainty_raw_scores: Optional[Dict[str, Any]],
    uncertainty_aucpr: Optional[Dict[str, Any]],
    baseline_results: Optional[Dict[str, Any]],
    uncertainty_stats: Optional[Dict[str, float]],
    base_dataset_name: Optional[str] = None,
    query_folder_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build structured eval data for W&B panels and summary tables."""
    panel: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "base_dataset_name": base_dataset_name or dataset_name,
        "query_folder_name": query_folder_name,
        "recalls": _recall_value_dict(recalls, args.get("recall_values", [1, 5, 10, 20])),
        "map_at_k": _recall_value_dict(map_at_k, args.get("recall_values", [1, 5, 10, 20])),
        "ece_recall": {},
        "ece_map": {},
        "ece_ap": {},
        "auc_pr": {},
        "variance_stat": dict(uncertainty_stats or {}),
        "raw_scores": {},
    }

    for variant, ece_result in (uncertainty_ece_results or {}).items():
        if variant == "raw_scores" or not isinstance(ece_result, dict):
            continue
        if "ece_recall" in ece_result:
            panel["ece_recall"][variant] = {
                int(k): float(v) for k, v in ece_result["ece_recall"].items()
            }
            scores = (uncertainty_raw_scores or {}).get(variant)
            if scores is not None:
                panel["raw_scores"][variant] = scores
        if "ece_map" in ece_result:
            panel["ece_map"][variant] = {
                int(k): float(v) for k, v in ece_result["ece_map"].items()
            }
        if "ece_ap" in ece_result:
            panel["ece_ap"][variant] = float(ece_result["ece_ap"])

    if baseline_results:
        baseline_raw = baseline_results.get("raw_scores") or {}
        for variant in ("l2", "pa", "sue", "sue_log"):
            ece = baseline_results.get(f"ece_{variant}")
            if isinstance(ece, dict):
                panel["ece_recall"][variant] = {int(k): float(val) for k, val in ece.items()}
                if baseline_raw.get(variant) is not None:
                    panel["raw_scores"][variant] = baseline_raw[variant]
        pair = baseline_results.get("ece_l2_pairwise")
        if isinstance(pair, dict) and "ece_recall" in pair:
            panel["ece_recall"]["l2_pairwise"] = {
                int(k): float(val) for k, val in pair["ece_recall"].items()
            }
            if baseline_raw.get("l2_pairwise") is not None:
                panel["raw_scores"]["l2_pairwise"] = baseline_raw["l2_pairwise"]

    if uncertainty_aucpr:
        for variant in ("kappa", "joint_kappa", "pairwise_joint_kappa"):
            v = uncertainty_aucpr.get(variant)
            if isinstance(v, (int, float, np.floating)):
                panel["auc_pr"][variant] = float(v)
    if baseline_results:
        for variant in ("l2", "l2_pairwise", "pa", "sue", "sue_log"):
            v = baseline_results.get(variant)
            if isinstance(v, (int, float, np.floating)):
                panel["auc_pr"][variant] = float(v)

    logger.debug(
        "%s W&B ece_recall variants: %s",
        dataset_name,
        sorted(panel["ece_recall"].keys()),
    )
    return panel


def panel_to_wandb_metrics(
    panel: Dict[str, Any],
    wandb_step: Optional[int] = None,
) -> Dict[str, float]:
    """Flatten panel data into W&B scalar keys."""
    dataset_name = panel["dataset_name"]
    prefix = f"Eval_{dataset_name}/"
    wandb_metrics: Dict[str, float] = {}

    for k, v in panel.get("recalls", {}).items():
        wandb_metrics[f"{prefix}recall_{k:02d}"] = float(v)
    for k, v in panel.get("map_at_k", {}).items():
        wandb_metrics[f"{prefix}map_{k:02d}"] = float(v)

    uncertainty_keys = _uncertainty_variant_keys(panel)
    for variant in panel.get("ece_recall", {}):
        for k, v in panel["ece_recall"][variant].items():
            key = f"{prefix}ece_{variant}_recall_{k:02d}"
            wandb_metrics[key] = float(v)
            if wandb_step is not None and variant in uncertainty_keys:
                wandb_metrics[f"uncertainty/ece_{variant}/recall_{k:02d}"] = float(v)

    for variant, ece in panel.get("ece_map", {}).items():
        if variant not in uncertainty_keys:
            continue
        for k, v in ece.items():
            key = f"{prefix}ece_{variant}_map_{k:02d}"
            wandb_metrics[key] = float(v)
            if wandb_step is not None:
                wandb_metrics[f"uncertainty/ece_{variant}/map_{k:02d}"] = float(v)

    for variant, v in panel.get("ece_ap", {}).items():
        if variant not in uncertainty_keys:
            continue
        key = f"{prefix}ece_{variant}_ap"
        wandb_metrics[key] = float(v)
        if wandb_step is not None:
            wandb_metrics[f"uncertainty/ece_{variant}/ap"] = float(v)

    for variant, v in panel.get("auc_pr", {}).items():
        wandb_metrics[f"{prefix}auc_pr_{variant}"] = float(v)
        if wandb_step is not None and variant in uncertainty_keys:
            wandb_metrics[f"uncertainty/auc_pr_{variant}"] = float(v)

    for variant, v in panel.get("auc_pr_norm_per_dataset", {}).items():
        wandb_metrics[f"{prefix}auc_pr_{variant}_norm_per_dataset"] = float(v)
    for variant, v in panel.get("auc_pr_norm_global", {}).items():
        wandb_metrics[f"{prefix}auc_pr_{variant}_norm_global"] = float(v)

    for name, v in panel.get("variance_stat", {}).items():
        wandb_metrics[f"{prefix}uncertainty_{name}"] = float(v)
        if wandb_step is not None:
            wandb_metrics[f"uncertainty/stats_{name}"] = float(v)

    for variant in SCORE_STAT_VARIANTS:
        raw = panel.get("raw_scores", {}).get(variant)
        if raw is None:
            continue
        for name, v in compute_score_statistics(raw).items():
            wandb_metrics[f"{prefix}{variant}_{name}"] = float(v)

    rb = panel.get("random_baseline")
    if isinstance(rb, (int, float, np.floating)):
        wandb_metrics[f"{prefix}auc_pr_random_baseline"] = float(rb)

    return wandb_metrics


def collect_wandb_images(
    dataset_name: str,
    dataset_output_dir: Path,
    args: Dict[str, Any],
    save_plots: bool,
) -> Optional[Dict[str, Any]]:
    """Collect variance-distribution and prediction plot paths for W&B logging."""
    if not args.get("use_wandb") or not dataset_output_dir.exists():
        return None

    prefix = f"Eval_{dataset_name}/"
    wandb_images: Dict[str, Any] = {}

    var_path = dataset_output_dir / "variance_distribution.png"
    if var_path.exists():
        wandb_images[f"{prefix}variance_distribution"] = var_path.resolve()

    preds_dir = dataset_output_dir / "preds"
    if save_plots and preds_dir.exists():
        image_list = sorted(preds_dir.glob("*.jpg"))
        if image_list:
            wandb_images[f"{prefix}predictions"] = image_list

    return wandb_images or None


def _bins_png_path(
    output_dir: Path,
    variant_key: str,
    *,
    plot_prefix: Optional[str],
    baseline: bool,
) -> Path:
    """Resolve ECE bin-count plot path (matches ``compute_ece`` / combined ``plot_prefix`` names)."""
    if plot_prefix:
        if baseline:
            if variant_key == "l2_pairwise":
                return output_dir / f"{plot_prefix}_pairwise_l2_bins.png"
            return output_dir / f"{plot_prefix}_{variant_key}_bins.png"
        if variant_key == "kappa":
            return output_dir / f"{plot_prefix}_bins.png"
        return output_dir / f"{plot_prefix}_{variant_key}_bins.png"
    return output_dir / f"{_ece_plot_stem(variant_key)}_bins.png"


def _curve_png_path(
    output_dir: Path,
    variant_key: str,
    *,
    plot_prefix: Optional[str],
) -> Path:
    """Resolve ECE curve plot path (``plot_ece`` outputs, not bins)."""
    if plot_prefix:
        if variant_key == "kappa":
            return output_dir / f"{plot_prefix}.png"
        if variant_key == "pairwise_joint_kappa":
            return output_dir / f"{plot_prefix}_pairwise_joint_kappa.png"
        return output_dir / f"{plot_prefix}_{variant_key}.png"
    if variant_key == "pairwise_joint_kappa":
        return output_dir / "ece_pairwise_joint_kappa.png"
    return output_dir / f"ece_{variant_key}.png"


def collect_ece_curve_images(
    output_dir: Optional[Path],
    *,
    plot_prefix: Optional[str] = None,
    variants: Sequence[str] = ECE_CURVES_PANEL_VARIANTS,
) -> Dict[str, Path]:
    """Collect ``plot_ece`` PNGs for the fixed ECE-curves panel variants."""
    if output_dir is None or not output_dir.exists():
        return {}

    images: Dict[str, Path] = {}
    for variant in variants:
        path = _curve_png_path(output_dir, variant, plot_prefix=plot_prefix)
        if path.exists():
            images[variant] = path.resolve()
    return images


def ece_curves_wandb_payload(
    dataset_name: str,
    output_dir: Optional[Path],
    *,
    variants: Sequence[str] = ECE_CURVES_PANEL_VARIANTS,
    show_missing: bool = True,
) -> Dict[str, Any]:
    """W&B media keys for the tiled ECE-curves panel (per dataset output dir)."""
    curve_images = collect_ece_curve_images(output_dir, variants=variants)
    if not curve_images:
        return {}
    return _ece_curves_log_keys(
        dataset_name, curve_images, panel_variants=variants, show_missing=show_missing
    )


def _primary_ece_variant(panel: Dict[str, Any]) -> Optional[str]:
    keys = _uncertainty_variant_keys(panel)
    if "kappa" in keys:
        return "kappa"
    return next(iter(sorted(keys)), None)


def build_training_val_ece_wandb_payload(
    panel: Dict[str, Any],
    wandb_images: Optional[Dict[str, Any]],
    dataset_output_dir: Optional[Path],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """W&B ``val/`` + ``ece/`` + ``ece_curves/`` + ``bins_distribution/`` panels (train loop)."""
    import wandb as _wandb

    payload: Dict[str, Any] = {}
    dataset_name = panel.get("dataset_name", "val")

    for k, v in panel.get("recalls", {}).items():
        payload[f"val/recall_{int(k):02d}"] = float(v)

    stats = panel.get("variance_stat") or {}

    def _pick(*keys: str) -> Optional[float]:
        for key in keys:
            val = stats.get(key)
            if isinstance(val, (int, float, np.floating)):
                return float(val)
        return None

    vmin = _pick("q_min", "min")
    vmax = _pick("q_max", "max")
    vmean = _pick("q_mean", "mean")
    vstd = _pick("q_std", "std")
    vmed = _pick("q_median", "median")
    if vmin is not None:
        payload["val/variances_min"] = vmin
    if vmax is not None:
        payload["val/variances_max"] = vmax
    if vmean is not None:
        payload["val/variances_mean"] = vmean
    if vstd is not None:
        payload["val/variances_std"] = vstd
    if vmed is not None:
        payload["val/variances_median"] = vmed

    variant = _primary_ece_variant(panel)
    if variant:
        ece_by_k = panel.get("ece_recall", {}).get(variant, {})
        recall_k = cfg.get("recall_values", [1, 5, 10, 20])
        for k in recall_k:
            if k in ece_by_k:
                payload[f"ece/ece_r{k}"] = float(ece_by_k[k])

    payload.update(
        ece_curves_wandb_payload(
            dataset_name,
            dataset_output_dir,
            variants=UNCERTAINTY_ECE_CURVES_VARIANTS,
            show_missing=False,
        )
    )
    bins_images = collect_bins_distribution_images(dataset_output_dir)
    payload.update(_bins_distribution_log_keys(dataset_name, bins_images))

    if wandb_images:
        prefix = f"Eval_{dataset_name}/"
        var_key = f"{prefix}variance_distribution"
        if var_key in wandb_images:
            p = Path(wandb_images[var_key])
            if p.exists():
                payload["val/variance_distribution"] = _wandb.Image(str(p))
        preds_key = f"{prefix}predictions"
        if preds_key in wandb_images:
            img_list = wandb_images[preds_key]
            if isinstance(img_list, list):
                valid = [Path(p) for p in img_list if Path(p).exists()]
                if valid:
                    payload["val/predictions"] = [_wandb.Image(str(p)) for p in valid]

    return payload


def collect_bins_distribution_images(
    output_dir: Optional[Path],
    *,
    plot_prefix: Optional[str] = None,
) -> Dict[str, Path]:
    """Collect ``*_bins.png`` for uncertainty ECE and baseline ECE variants."""
    if output_dir is None or not output_dir.exists():
        return {}

    images: Dict[str, Path] = {}
    for variant_key, variant_label in UNCERTAINTY_ECE_VARIANTS:
        path = _bins_png_path(
            output_dir, variant_key, plot_prefix=plot_prefix, baseline=False
        )
        if path.exists():
            images[variant_label] = path.resolve()

    for variant_key, variant_label in BASELINE_BINS_VARIANTS:
        path = _bins_png_path(output_dir, variant_key, plot_prefix=plot_prefix, baseline=True)
        if path.exists():
            images[variant_label] = path.resolve()

    return images


def collect_combined_bins_distribution_images(
    output_dir: Path,
    output_subdir: str,
    *,
    plot_prefix: str,
) -> Dict[str, Path]:
    """Collect combined ECE bin plots and optional kappa distribution histogram."""
    images = collect_bins_distribution_images(output_dir, plot_prefix=plot_prefix)
    dist_path = output_dir / f"{output_subdir}_kappa_distribution.png"
    if dist_path.exists():
        images["query_uncertainty_distribution"] = dist_path.resolve()
    return images


def _bins_distribution_log_keys(name: str, bins_images: Dict[str, Any]) -> Dict[str, Any]:
    """Single tiled key under ``bins_distribution/<dataset>``."""
    import wandb as _wandb

    payload: Dict[str, Any] = {}
    prefix = "bins_distribution"
    ordered = [caption for caption, img in sorted(bins_images.items()) if img and Path(img).exists()]
    if ordered:
        payload[f"{prefix}/{name}_gallery"] = [
            _wandb.Image(str(Path(bins_images[caption])), caption=caption) for caption in ordered
        ]
    tile = _render_image_tile(
        image_map=bins_images,
        ordered_captions=ordered,
        output_path=(Path("/tmp") / f"bins_distribution_{name}_all_metrics.png"),
        title=f"bins_distribution/{name}",
        show_missing=False,
    )
    if tile and tile.exists():
        payload[f"{prefix}/{name}"] = _wandb.Image(str(tile), caption="all_metrics")
    return payload


def _ece_curves_log_keys(
    name: str,
    curve_images: Dict[str, Any],
    *,
    panel_variants: Sequence[str] = ECE_CURVES_PANEL_VARIANTS,
    show_missing: bool = True,
) -> Dict[str, Any]:
    """Single tiled key under ``ece_curves/<dataset>``."""
    import wandb as _wandb

    payload: Dict[str, Any] = {}
    prefix = "ece_curves"
    ordered = [
        caption
        for caption in panel_variants
        if curve_images.get(caption) and Path(curve_images[caption]).exists()
    ]
    if ordered:
        payload[f"{prefix}/{name}_gallery"] = [
            _wandb.Image(str(Path(curve_images[caption])), caption=caption) for caption in ordered
        ]
    tile = _render_image_tile(
        image_map=curve_images,
        ordered_captions=list(panel_variants),
        output_path=(Path("/tmp") / f"ece_curves_{name}_all_metrics.png"),
        title=f"ece_curves/{name}",
        show_missing=show_missing,
    )
    if tile and tile.exists():
        payload[f"{prefix}/{name}"] = _wandb.Image(str(tile), caption="all_metrics")
    else:
        payload[f"{prefix}/{name}"] = _wandb.Html(
            "<div style='padding:12px;font-family:monospace;'>No ECE curve plots available</div>"
        )
    return payload


def _render_image_tile(
    *,
    image_map: Dict[str, Any],
    ordered_captions: Sequence[str],
    output_path: Path,
    title: str,
    show_missing: bool,
) -> Optional[Path]:
    """Render a multi-plot image tile from metric images."""
    if not ordered_captions:
        return None

    existing = [c for c in ordered_captions if image_map.get(c) and Path(image_map[c]).exists()]
    if not existing and not show_missing:
        return None

    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return None

    cols = min(3, max(1, math.ceil(math.sqrt(len(ordered_captions)))))
    rows = math.ceil(len(ordered_captions) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows))
    axes_arr = np.array(axes).reshape(-1)

    for i, caption in enumerate(ordered_captions):
        ax = axes_arr[i]
        img = image_map.get(caption)
        if img and Path(img).exists():
            ax.imshow(mpimg.imread(str(img)))
            ax.set_title(caption, fontsize=10)
            ax.axis("off")
        else:
            ax.set_title(caption, fontsize=10)
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

    for j in range(len(ordered_captions), len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.resolve()


def _is_numeric_cell(cell: Any) -> bool:
    if cell is None:
        return False
    if isinstance(cell, (int, float, np.floating)):
        return not (isinstance(cell, float) and math.isnan(cell))
    return False


def _best_positions(
    indexed_values: List[Tuple[int, float]],
    *,
    higher_better: bool,
) -> List[int]:
    """Return indices tied for the best value in ``indexed_values``."""
    if not indexed_values:
        return []
    if higher_better:
        best = max(v for _, v in indexed_values)
        return [i for i, v in indexed_values if v == best]
    best = min(v for _, v in indexed_values)
    return [i for i, v in indexed_values if v == best]


def _bold_mask_variant_rows(
    rows: List[List[Any]],
    *,
    variant_col: int,
    numeric_start: int,
    main_labels: frozenset,
    pairwise_labels: frozenset,
    higher_better: bool,
) -> set:
    """Per dataset row-group: bold best numeric cell in each comparison group."""
    bold: set = set()
    by_dataset: Dict[Any, List[Tuple[int, List[Any]]]] = {}
    for row_idx, row in enumerate(rows):
        by_dataset.setdefault(row[0], []).append((row_idx, row))

    for entries in by_dataset.values():
        for col_idx in range(numeric_start, len(entries[0][1])):
            main_vals = [
                (row_idx, float(row[col_idx]))
                for row_idx, row in entries
                if row[variant_col] in main_labels and _is_numeric_cell(row[col_idx])
            ]
            pair_vals = [
                (row_idx, float(row[col_idx]))
                for row_idx, row in entries
                if row[variant_col] in pairwise_labels and _is_numeric_cell(row[col_idx])
            ]
            for row_idx in _best_positions(main_vals, higher_better=higher_better):
                bold.add((row_idx, col_idx))
            for row_idx in _best_positions(pair_vals, higher_better=higher_better):
                bold.add((row_idx, col_idx))
    return bold


def _bold_mask_auc_pr_columns(cols: List[str], rows: List[List[Any]]) -> set:
    """Per dataset row: bold best column within each method group."""
    bold: set = set()
    col_idx = {name: i for i, name in enumerate(cols)}
    main_cols = [col_idx[c] for c in _AUC_PR_MAIN_COLUMNS if c in col_idx]
    pair_cols = [col_idx[c] for c in _AUC_PR_PAIRWISE_COLUMNS if c in col_idx]

    for row_idx, row in enumerate(rows):
        main_vals = [
            (col_idx, float(row[col_idx]))
            for col_idx in main_cols
            if col_idx < len(row) and _is_numeric_cell(row[col_idx])
        ]
        pair_vals = [
            (col_idx, float(row[col_idx]))
            for col_idx in pair_cols
            if col_idx < len(row) and _is_numeric_cell(row[col_idx])
        ]
        for col_idx in _best_positions(main_vals, higher_better=True):
            bold.add((row_idx, col_idx))
        for col_idx in _best_positions(pair_vals, higher_better=True):
            bold.add((row_idx, col_idx))
    return bold


def _summary_table_bold_cells(
    table_key: str,
    cols: List[str],
    rows: List[List[Any]],
) -> set:
    if table_key not in _CALIBRATION_COMPARISON_TABLE_KEYS:
        return set()
    if table_key == AUC_PR_PANEL_KEY:
        return _bold_mask_auc_pr_columns(cols, rows)
    # calibration_ece + uncertainty_stats: lower is better (ECE / score magnitude).
    return _bold_mask_variant_rows(
        rows,
        variant_col=1,
        numeric_start=2,
        main_labels=_CALIBRATION_MAIN_VARIANT_LABELS,
        pairwise_labels=_CALIBRATION_PAIRWISE_VARIANT_LABELS,
        higher_better=False,
    )


def _html_table(
    headers: List[str],
    rows: List[List[Any]],
    *,
    label_columns: int = 1,
    bold_cells: Optional[set] = None,
) -> Any:
    """Compact HTML table (all rows visible; avoids W&B Table row pagination)."""
    import wandb as _wandb

    def _cell_text(cell: Any) -> str:
        if cell is None:
            return "—"
        if isinstance(cell, (float, np.floating)):
            return f"{float(cell):.3f}"
        return str(cell)

    style = "style='border-collapse:collapse;width:100%;font-family:monospace;font-size:14px;'"
    th_style = "style='border:1px solid #555;padding:6px;background:#2a2a2a;color:#eee;'"
    td_style = "style='border:1px solid #555;padding:6px;text-align:center;'"
    td_label = "style='border:1px solid #555;padding:6px;text-align:left;font-weight:bold;'"
    td_bold = "style='border:1px solid #555;padding:6px;text-align:center;font-weight:bold;'"
    html = [f"<table {style}><tr>"]
    for h in headers:
        html.append(f"<th {th_style}>{h}</th>")
    html.append("</tr>")
    for row_idx, row in enumerate(rows):
        html.append("<tr>")
        for col_idx, cell in enumerate(row):
            if col_idx < label_columns:
                s = td_label
            elif bold_cells and (row_idx, col_idx) in bold_cells:
                s = td_bold
            else:
                s = td_style
            html.append(f"<td {s}>{_cell_text(cell)}</td>")
        html.append("</tr>")
    html.append("</table>")
    return _wandb.Html("".join(html))


# Variant labels in ``calibration_ece`` / ``uncertainty_stats`` rows (column 1).
_CALIBRATION_MAIN_VARIANT_LABELS = frozenset(
    {"kappa", "joint_kappa", "l2", "pa", "sue_log"}
)
_CALIBRATION_PAIRWISE_VARIANT_LABELS = frozenset({"pairwise_l2", "pairwise"})

# ``AUC-PR`` table column headers (one row per dataset).
_AUC_PR_MAIN_COLUMNS = frozenset({"kappa", "joint_kappa", "l2", "pa", "sue_log"})
_AUC_PR_PAIRWISE_COLUMNS = frozenset({"pairwise_l2", "pairwise_kappa"})

# Tables that compare calibration methods (bold best per group in summary HTML).
_CALIBRATION_COMPARISON_TABLE_KEYS = frozenset(
    {"calibration_ece", AUC_PR_PANEL_KEY, "uncertainty_stats"}
)

# label_columns: leading columns rendered left-aligned (dataset / variant names).
_SUMMARY_TABLE_LABEL_COLS = {
    "retrieval": 1,
    "calibration_ece": 2,
    AUC_PR_PANEL_KEY: 1,
    "descriptor_stats": 1,
    "uncertainty_stats": 2,
}


def _score_stat_label(variant: str) -> str:
    return SCORE_STAT_LABELS.get(variant, variant)


def _build_descriptor_stats_table(
    all_panel_data: List[Dict[str, Any]],
    combined_panels: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[List[str], List[List[Any]]]]:
    """DB+query learned uncertainty (κ / variance), one row per base dataset.

    When multiple query folders are evaluated for the same dataset, aggregate the
    descriptor stats by mean so the table stays one-row-per-dataset.
    """
    stat_keys = sorted({k for d in all_panel_data for k in d.get("variance_stat", {})})
    if not stat_keys:
        return None
    cols = ["dataset"] + stat_keys
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for d in all_panel_data:
        vs = d.get("variance_stat")
        if not vs:
            continue
        base_name = d.get("base_dataset_name") or d.get("dataset_name")
        grouped.setdefault(str(base_name), []).append(vs)

    rows: List[List[Any]] = []
    for dataset in sorted(grouped):
        stats_list = grouped[dataset]
        row_vals: List[Any] = []
        for k in stat_keys:
            vals = [float(s[k]) for s in stats_list if k in s and s[k] is not None]
            row_vals.append(float(np.mean(vals)) if vals else None)
        rows.append([dataset] + row_vals)

    # Keep descriptor_stats as per-dataset only (no combined_* rows).
    if not rows:
        return None
    return cols, rows


def _build_uncertainty_stats_table(
    all_panel_data: List[Dict[str, Any]],
    combined_panels: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[List[str], List[List[Any]]]]:
    """Per-metric score stats used for ECE / AUC-PR (kappa, baselines, …)."""
    entries: List[Tuple[str, str, Dict[str, float]]] = []

    for d in all_panel_data:
        dataset = d["dataset_name"]
        for variant in SCORE_STAT_VARIANTS:
            raw = d.get("raw_scores", {}).get(variant)
            if raw is None:
                continue
            stats = compute_score_statistics(raw)
            if stats:
                entries.append((dataset, _score_stat_label(variant), stats))

    for d in combined_panels or []:
        dataset = d.get("dataset_name")
        for variant in SCORE_STAT_VARIANTS:
            raw = d.get("raw_scores", {}).get(variant)
            if raw is None:
                continue
            stats = compute_score_statistics(raw)
            if stats:
                entries.append((dataset, _score_stat_label(variant), stats))

    if not entries:
        return None

    stat_keys = sorted({k for _, _, stats in entries for k in stats})
    cols = ["dataset", "metric"] + stat_keys
    rows = [
        [dataset, metric] + [stats.get(k) for k in stat_keys]
        for dataset, metric, stats in entries
    ]
    return cols, rows


def _build_eval_summary_tables(
    cfg: Dict[str, Any],
    all_panel_data: List[Dict[str, Any]],
    combined_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Tuple[List[str], List[List[Any]]]]:
    """Shared columns/rows for wandb.Table and HTML summary panels."""
    rv_sorted = sorted(cfg.get("recall_values", [1, 5, 10, 20]))
    out: Dict[str, Tuple[List[str], List[List[Any]]]] = {}

    combined_panels: List[Dict[str, Any]] = [
        combined.get("panel", {})
        for combined in (combined_outputs or [])
        if combined.get("panel")
    ]
    table_panels: List[Dict[str, Any]] = list(all_panel_data) + combined_panels

    cols = ["dataset"] + [f"R@{k}" for k in rv_sorted] + [f"mAP@{k}" for k in rv_sorted]
    rows = [
        [d["dataset_name"]]
        + [d.get("recalls", {}).get(k) for k in rv_sorted]
        + [d.get("map_at_k", {}).get(k) for k in rv_sorted]
        for d in table_panels
    ]
    if rows:
        out["retrieval"] = (cols, rows)

    cols = ["dataset", "variant"] + [f"ECE_R@{k}" for k in rv_sorted]
    rows = []
    for d in table_panels:
        for variant_key, variant_label in CALIBRATION_ECE_VARIANTS:
            ece = d.get("ece_recall", {}).get(variant_key)
            if not ece:
                continue
            rows.append(
                [d["dataset_name"], variant_label] + [ece.get(k) for k in rv_sorted]
            )
    if rows:
        out["calibration_ece"] = (cols, rows)

    fp_keys = [key for key, _ in AUC_PR_SUMMARY_COLUMNS]
    fp_labels = [label for _, label in AUC_PR_SUMMARY_COLUMNS]
    cols = ["dataset"] + fp_labels
    rows = [
        [d["dataset_name"]] + [d.get("auc_pr", {}).get(k) for k in fp_keys]
        for d in table_panels
    ]
    if rows:
        out[AUC_PR_PANEL_KEY] = (cols, rows)

    descriptor_stats = _build_descriptor_stats_table(all_panel_data, combined_panels)
    if descriptor_stats is not None:
        out["descriptor_stats"] = descriptor_stats

    uncertainty_stats = _build_uncertainty_stats_table(all_panel_data, combined_panels)
    if uncertainty_stats is not None:
        out["uncertainty_stats"] = uncertainty_stats

    return out


def build_eval_summary_html_panels(
    cfg: Dict[str, Any],
    all_panel_data: List[Dict[str, Any]],
    combined_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """HTML summaries for all cross-dataset tables (no row pagination)."""
    panels: Dict[str, Any] = {}
    for key, (cols, rows) in _build_eval_summary_tables(
        cfg, all_panel_data, combined_outputs
    ).items():
        bold_cells = _summary_table_bold_cells(key, cols, rows)
        panels[f"{key}/summary_html"] = _html_table(
            cols,
            rows,
            label_columns=_SUMMARY_TABLE_LABEL_COLS.get(key, 1),
            bold_cells=bold_cells,
        )
    return panels


def build_eval_panel_payload(
    cfg: Dict[str, Any],
    all_panel_data: List[Dict[str, Any]],
    combined_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build batched W&B eval panels (summary tables + bins-distribution images)."""
    import wandb as _wandb

    tables: Dict[str, Any] = {}
    bins: Dict[str, Any] = {}
    ece_curves: Dict[str, Any] = {}

    for key, (cols, rows) in _build_eval_summary_tables(
        cfg, all_panel_data, combined_outputs
    ).items():
        tables[f"{key}/table"] = _wandb.Table(columns=cols, data=rows)

    eval_root = Path(cfg["log_dir"]) / "eval"
    for d in all_panel_data:
        bins_images = collect_bins_distribution_images(eval_root / d["dataset_name"])
        curve_images = collect_ece_curve_images(eval_root / d["dataset_name"])
        bins.update(_bins_distribution_log_keys(d["dataset_name"], bins_images))
        ece_curves.update(_ece_curves_log_keys(d["dataset_name"], curve_images))

    for combined in combined_outputs or []:
        bins_images = collect_combined_bins_distribution_images(
            combined["bins_output_dir"],
            combined["name"],
            plot_prefix=combined["plot_prefix"],
        )
        curve_images = collect_ece_curve_images(
            combined["bins_output_dir"], plot_prefix=combined["plot_prefix"]
        )
        bins.update(_bins_distribution_log_keys(combined["name"], bins_images))
        ece_curves.update(_ece_curves_log_keys(combined["name"], curve_images))

    return {**tables, **bins, **ece_curves}


def build_eval_outputs(
    dataset_name: str,
    args: Dict[str, Any],
    recalls,
    map_at_k,
    uncertainty_ece_results,
    uncertainty_raw_scores,
    uncertainty_aucpr,
    baseline_results,
    uncertainty_stats,
    dataset_output_dir: Optional[Path] = None,
    save_plots: bool = False,
    base_dataset_name: Optional[str] = None,
    query_folder_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Build W&B panel data and image paths for one dataset eval."""
    panel_data = _build_panel_data(
        dataset_name,
        args,
        recalls,
        map_at_k,
        uncertainty_ece_results,
        uncertainty_raw_scores,
        uncertainty_aucpr,
        baseline_results,
        uncertainty_stats,
        base_dataset_name=base_dataset_name,
        query_folder_name=query_folder_name,
    )

    wandb_images = None
    if dataset_output_dir is not None:
        wandb_images = collect_wandb_images(dataset_name, dataset_output_dir, args, save_plots)

    return panel_data, wandb_images
