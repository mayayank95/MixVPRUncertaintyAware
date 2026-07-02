"""Combined multi-dataset ECE and failure-prediction metrics."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eval_metrics.eval_ece_sh import compute_ece, compute_ece_pairwise, vmf_like_uncertainty_loss
from eval_metrics.failure_prediction import combined_failure_prediction_pr_auc
from eval_metrics.log_utils import log_combined_panel
from eval_metrics.uncertainty import compute_score_statistics

logger = logging.getLogger(__name__)


def _combined_shards_from_panels(all_panel_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build combined-ECE shard dicts from per-dataset panel rows."""
    shards: List[Dict[str, Any]] = []
    for panel in all_panel_data:
        pool = panel.get("pool")
        if not pool:
            continue
        shards.append({
            "name": panel["dataset_name"],
            **pool,
            "raw_scores": panel.get("raw_scores", {}),
            "base_dataset_name": panel.get("base_dataset_name", ""),
            "query_folder_name": panel.get("query_folder_name"),
        })
    return shards


def _shard_base_name(shard: Dict[str, Any]) -> str:
    return (shard.get("base_dataset_name") or shard.get("name") or "").lower()


def _is_sf_xl(shard: Dict[str, Any]) -> bool:
    bn = _shard_base_name(shard)
    return "sf_xl" in bn or "sf-xl" in bn


def _is_pitts30(shard: Dict[str, Any]) -> bool:
    return "pitts30" in _shard_base_name(shard)


def _is_msls_query(shard: Dict[str, Any]) -> bool:
    return "msls" in _shard_base_name(shard) and shard.get("query_folder_name") == "query"


def _is_amstertime(shard: Dict[str, Any]) -> bool:
    return "amstertime" in _shard_base_name(shard)


def _matches_core_four(shard: Dict[str, Any]) -> bool:
    """all sf-xl shards, msls-val query, pitts30k."""
    return _is_sf_xl(shard) or _is_pitts30(shard) or _is_msls_query(shard)


def _matches_core_five(shard: Dict[str, Any]) -> bool:
    return _matches_core_four(shard) or _is_amstertime(shard)


def _merge_ece_variant(panel: Dict[str, Any], variant: str, ece_result: Dict[str, Any]) -> None:
    if "ece_recall" in ece_result:
        panel.setdefault("ece_recall", {})[variant] = {
            int(k): float(v) for k, v in ece_result["ece_recall"].items()
        }
    if "ece_map" in ece_result:
        panel.setdefault("ece_map", {})[variant] = {
            int(k): float(v) for k, v in ece_result["ece_map"].items()
        }
    if "ece_ap" in ece_result:
        panel.setdefault("ece_ap", {})[variant] = float(ece_result["ece_ap"])


def _new_combined_panel(output_subdir: str) -> Dict[str, Any]:
    return {
        "dataset_name": output_subdir,
        "recalls": {},
        "map_at_k": {},
        "ece_recall": {},
        "ece_map": {},
        "ece_ap": {},
        "auc_pr": {},
        "auc_pr_norm_per_dataset": {},
        "auc_pr_norm_global": {},
        "raw_scores": {},
        "variance_stat": {},
    }


def _set_auc_pr_norms(
    panel: Dict[str, Any],
    variant: str,
    per_dataset: float,
    norm_global: float,
) -> None:
    panel.setdefault("auc_pr_norm_per_dataset", {})[variant] = float(per_dataset)
    panel.setdefault("auc_pr_norm_global", {})[variant] = float(norm_global)
    panel.setdefault("auc_pr", {})[variant] = float(per_dataset)


def compute_and_log_combined(
    cfg,
    agg_data_list,
    output_subdir,
    plot_prefix,
    wandb_prefix,
) -> Optional[Dict[str, Any]]:
    """Pool shards with DB index offsets; report AUC-PR with per-dataset vs global score normalization."""
    combined_variances = np.vstack([d["q_var"] for d in agg_data_list])
    combined_distances = np.vstack([d["distances"] for d in agg_data_list])
    ece_metrics = cfg.get("ece_metrics") or ["recall"]
    uncertainty_loss = cfg.get("uncertainty_loss", "gaussian_nll")

    combined_predictions_list = []
    combined_positives = []
    combined_matched = []
    kappa_sh_matched: List[np.ndarray] = []
    kappa_sh_conf: List[np.ndarray] = []
    db_offset = 0
    for d in agg_data_list:
        preds = d["predictions"]
        positives = d["positives_per_query"]
        max_db_idx = int(preds.max()) + 1 if len(preds) > 0 else 0
        nq = preds.shape[0]
        matched_vec = np.array(
            [float(np.any(np.isin(preds[i, :1], positives[i]))) for i in range(nq)],
            dtype=np.float64,
        )
        mean_var_shard = np.mean(d["q_var"], axis=-1)
        kconf_shard = mean_var_shard if vmf_like_uncertainty_loss(uncertainty_loss) else -mean_var_shard
        kappa_sh_matched.append(matched_vec)
        kappa_sh_conf.append(kconf_shard)

        combined_predictions_list.append(preds + db_offset)
        for i, pos_list in enumerate(positives):
            combined_positives.append(np.array(pos_list) + db_offset)
            is_hit = np.any(np.isin(preds[i, :1], pos_list))
            combined_matched.append(float(is_hit))

        db_offset += max_db_idx

    combined_predictions = np.vstack(combined_predictions_list)
    combined_matched = np.array(combined_matched)

    combined_output_dir = Path(cfg["log_dir"]) / "eval" / output_subdir
    combined_output_dir.mkdir(parents=True, exist_ok=True)

    combined_panel = _new_combined_panel(output_subdir)

    combined_ece = compute_ece(
        combined_predictions,
        combined_positives,
        combined_variances,
        n_values=cfg["recall_values"],
        output_dir=combined_output_dir,
        metrics=ece_metrics,
        distances=combined_distances,
        uncertainty_loss=uncertainty_loss,
        zoom_threshold=cfg.get("ece_zoom_threshold", 0.001),
        plot_filename=f"{plot_prefix}.png",
        bin_mode=cfg.get("ece_bin_mode", "zoom"),
        vmf_kappa_floor=cfg.get("ece_vmf_kappa_floor", False),
        percentile_two_sided=bool(cfg.get("ece_two_sided", False)),
    )
    _merge_ece_variant(combined_panel, "kappa", combined_ece)

    total_q = len(combined_positives)
    if total_q > 0:
        orig_rec = np.zeros(len(cfg["recall_values"]))
        for qi, preds in enumerate(combined_predictions):
            for i, n in enumerate(cfg["recall_values"]):
                if np.any(np.isin(preds[:n], combined_positives[qi])):
                    orig_rec[i:] += 1
                    break
        orig_rec = orig_rec / total_q * 100
        combined_panel["recalls"] = {
            int(k_val): float(v) for k_val, v in zip(cfg["recall_values"], orig_rec)
        }

    combined_mean_var = np.mean(combined_variances, axis=-1)
    combined_panel["raw_scores"]["kappa"] = combined_variances
    # Combined views pool query-side uncertainty only; keep q_* stats explicit.
    combined_panel["variance_stat"] = compute_score_statistics(combined_variances)

    if not cfg.get("skip_auc_pr"):
        kappa_pd = combined_failure_prediction_pr_auc(
            "per_dataset", kappa_sh_matched, kappa_sh_conf
        )
        kappa_g = combined_failure_prediction_pr_auc(
            "global", kappa_sh_matched, kappa_sh_conf
        )
        _set_auc_pr_norms(combined_panel, "kappa", kappa_pd, kappa_g)

    for b_name in ["l2", "pa", "sue", "sue_log"]:
        if agg_data_list and all(
            d.get("raw_scores")
            and b_name in d["raw_scores"]
            and d["raw_scores"][b_name] is not None
            for d in agg_data_list
        ):
            base_predictions_list = []
            base_positives = []
            base_distances = []
            base_db_offset = 0
            for d in agg_data_list:
                preds = d["predictions"]
                positives = d["positives_per_query"]
                dists = d["distances"]
                max_db_idx = int(preds.max()) + 1 if len(preds) > 0 else 0

                base_predictions_list.append(preds + base_db_offset)
                base_distances.append(dists)
                for pos_list in positives:
                    base_positives.append(np.array(pos_list) + base_db_offset)
                base_db_offset += max_db_idx

            base_predictions = np.vstack(base_predictions_list)
            base_distances = np.vstack(base_distances)

            base_sh_matched: List[np.ndarray] = []
            base_sh_bconf: List[np.ndarray] = []
            for d in agg_data_list:
                preds = d["predictions"]
                positives = d["positives_per_query"]
                nqb = preds.shape[0]
                mv_b = np.array(
                    [float(np.any(np.isin(preds[i, :1], positives[i]))) for i in range(nqb)],
                    dtype=np.float64,
                )
                raw_b = np.asarray(d["raw_scores"][b_name], dtype=np.float64).reshape(-1)
                base_sh_matched.append(mv_b)
                base_sh_bconf.append(-raw_b)

            b_scores = np.concatenate([d["raw_scores"][b_name] for d in agg_data_list])
            combined_panel["raw_scores"][b_name] = b_scores
            b_ece = compute_ece(
                base_predictions,
                base_positives,
                b_scores[:, None],
                n_values=cfg["recall_values"],
                output_dir=combined_output_dir,
                metrics=ece_metrics,
                distances=base_distances,
                uncertainty_loss="gaussian_nll",
                zoom_threshold=cfg.get("ece_zoom_threshold", 0.001),
                plot_filename=f"{plot_prefix}_{b_name}.png",
                bin_mode=cfg.get("ece_bin_mode", "zoom"),
                percentile_two_sided=False,
            )
            _merge_ece_variant(combined_panel, b_name, b_ece)

            if not cfg.get("skip_auc_pr"):
                b_pd = combined_failure_prediction_pr_auc(
                    "per_dataset", base_sh_matched, base_sh_bconf
                )
                b_g = combined_failure_prediction_pr_auc(
                    "global", base_sh_matched, base_sh_bconf
                )
                _set_auc_pr_norms(combined_panel, b_name, b_pd, b_g)

    if all(
        d.get("raw_scores") and d["raw_scores"].get("l2_pairwise") is not None
        for d in agg_data_list
    ):
        l2_pair_scores_combined = np.vstack([
            d["raw_scores"]["l2_pairwise"] for d in agg_data_list
        ])
        combined_panel["raw_scores"]["l2_pairwise"] = l2_pair_scores_combined
        l2_pw_ece = compute_ece_pairwise(
            combined_predictions,
            combined_positives,
            l2_pair_scores_combined,
            n_values=cfg["recall_values"],
            output_dir=combined_output_dir,
            plot_filename=f"{plot_prefix}_pairwise_l2.png",
            zoom_threshold=cfg.get("ece_zoom_threshold", 0.001),
            bin_mode=cfg.get("ece_bin_mode", "zoom"),
            uncertainty_loss="gaussian_nll",
            percentile_two_sided=False,
        )
        _merge_ece_variant(combined_panel, "l2_pairwise", l2_pw_ece)

    if all(
        d.get("raw_scores")
        and d["raw_scores"].get("joint_kappa") is not None
        for d in agg_data_list
    ):
        jk_scores = np.concatenate([d["raw_scores"]["joint_kappa"] for d in agg_data_list])
        combined_panel["raw_scores"]["joint_kappa"] = jk_scores
        jk_ece = compute_ece(
            combined_predictions,
            combined_positives,
            jk_scores[:, None],
            n_values=cfg["recall_values"],
            output_dir=combined_output_dir,
            metrics=ece_metrics,
            distances=combined_distances,
            uncertainty_loss="vmf",
            zoom_threshold=cfg.get("ece_zoom_threshold", 0.001),
            plot_filename=f"{plot_prefix}_joint_kappa.png",
            bin_mode=cfg.get("ece_bin_mode", "zoom"),
            vmf_kappa_floor=cfg.get("ece_vmf_kappa_floor", False),
            percentile_two_sided=bool(cfg.get("ece_two_sided", False)),
        )
        _merge_ece_variant(combined_panel, "joint_kappa", jk_ece)

        if not cfg.get("skip_auc_pr"):
            jk_sh_matched: List[np.ndarray] = []
            jk_sh_scores: List[np.ndarray] = []
            for d in agg_data_list:
                preds = d["predictions"]
                positives = d["positives_per_query"]
                jk_row = np.asarray(d["raw_scores"]["joint_kappa"], dtype=np.float64).reshape(-1)
                nqj = preds.shape[0]
                mv_j = np.array(
                    [float(np.any(np.isin(preds[i, :1], positives[i]))) for i in range(nqj)],
                    dtype=np.float64,
                )
                jk_sh_matched.append(mv_j)
                jk_sh_scores.append(jk_row)
            jk_pd = combined_failure_prediction_pr_auc(
                "per_dataset", jk_sh_matched, jk_sh_scores
            )
            jk_g = combined_failure_prediction_pr_auc(
                "global", jk_sh_matched, jk_sh_scores
            )
            _set_auc_pr_norms(combined_panel, "joint_kappa", jk_pd, jk_g)

    if all(
        d.get("raw_scores")
        and d["raw_scores"].get("pairwise_joint_kappa") is not None
        for d in agg_data_list
    ):
        jk_pair_scores_combined = np.vstack([
            d["raw_scores"]["pairwise_joint_kappa"] for d in agg_data_list
        ])
        combined_panel["raw_scores"]["pairwise_joint_kappa"] = jk_pair_scores_combined
        jk_pw_ece = compute_ece_pairwise(
            combined_predictions,
            combined_positives,
            jk_pair_scores_combined,
            n_values=cfg["recall_values"],
            output_dir=combined_output_dir,
            plot_filename=f"{plot_prefix}_pairwise_joint_kappa.png",
            zoom_threshold=cfg.get("ece_zoom_threshold", 0.001),
            bin_mode=cfg.get("ece_bin_mode", "zoom"),
            uncertainty_loss="vmf",
            vmf_kappa_floor=cfg.get("ece_vmf_kappa_floor", False),
            percentile_two_sided=bool(cfg.get("ece_two_sided", False)),
        )
        _merge_ece_variant(combined_panel, "pairwise_joint_kappa", jk_pw_ece)

    log_combined_panel(output_subdir, combined_panel)

    if len(combined_matched):
        pos_rate = float(np.mean(combined_matched))
        logger.info(
            "%s: top-1 positive rate (combined) = %.4f (random classifier ≈ this area under PR)",
            output_subdir,
            pos_rate,
        )

    try:
        import matplotlib.pyplot as plt
        with plt.style.context("default"):
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            ax = axs[0]
            for d in agg_data_list:
                ds_mean_var = np.mean(d["q_var"], axis=-1)
                ds_label = d.get("name", "unknown")
                ax.hist(
                    ds_mean_var,
                    bins=50,
                    alpha=0.5,
                    label=f"{ds_label} (n={len(ds_mean_var)})",
                    edgecolor="black",
                    linewidth=0.3,
                )
            ax.set_xlabel("Mean kappa / variance per query")
            ax.set_ylabel("Frequency")
            ax.set_title("Per-dataset query uncertainty distribution")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax = axs[1]
            ax.hist(
                combined_mean_var,
                bins=50,
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
                linewidth=0.3,
            )
            ax.axvline(
                np.median(combined_mean_var),
                color="tomato",
                linestyle="--",
                linewidth=1.5,
                label=f"Median = {np.median(combined_mean_var):.4f}",
            )
            ax.axvline(
                np.mean(combined_mean_var),
                color="orange",
                linestyle="-",
                linewidth=1.5,
                label=f"Mean = {np.mean(combined_mean_var):.4f}",
            )
            ax.set_xlabel("Mean kappa / variance per query")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Combined query uncertainty (n={len(combined_mean_var)})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            dist_path = combined_output_dir / f"{output_subdir}_kappa_distribution.png"
            plt.savefig(dist_path, dpi=150)
            plt.close(fig)
        logger.info("Combined kappa distribution plot saved to %s", dist_path)
    except ImportError:
        logger.warning("matplotlib not installed, skipping combined kappa distribution plot.")
    except Exception as e:
        logger.warning("Error saving combined kappa distribution plot: %s", e)

    total_queries = len(combined_positives)
    n_datasets = len(agg_data_list)
    logger.info(
        "%s: combined ECE computed over %d queries from %d datasets.",
        output_subdir,
        total_queries,
        n_datasets,
    )

    return {
        "name": output_subdir,
        "wandb_prefix": wandb_prefix,
        "panel": combined_panel,
        "bins_output_dir": combined_output_dir,
        "plot_prefix": plot_prefix,
    }


def _maybe_run_combined(
    cfg,
    shards: List[Dict[str, Any]],
    *,
    output_subdir: str,
    plot_prefix: str,
    wandb_prefix: str,
    combined_outputs: List[Dict[str, Any]],
    description: str,
    min_shards: int = 2,
    exact_shards: Optional[int] = None,
    required_predicate=None,
) -> None:
    if exact_shards is not None:
        if len(shards) != exact_shards:
            return
    elif len(shards) < min_shards:
        return
    if required_predicate is not None and not required_predicate(shards):
        return
    logger.info("=" * 30 + "\nComputing %s (%s)...", output_subdir, description)
    out = compute_and_log_combined(cfg, shards, output_subdir, plot_prefix, wandb_prefix)
    if out:
        combined_outputs.append(out)


def _filter_shards_all(agg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return agg_list


def _filter_shards_no_amstertime(agg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [d for d in agg_list if not _is_amstertime(d)]


def _filter_shards_core_five(agg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [d for d in agg_list if _matches_core_five(d)]


def _filter_shards_core_four(agg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [d for d in agg_list if _matches_core_four(d)]


def _has_core_four_requirements(shards: List[Dict[str, Any]]) -> bool:
    """Require at least one sf-xl shard, msls(query), and pitts30k shard."""
    return (
        any(_is_sf_xl(s) for s in shards)
        and any(_is_msls_query(s) for s in shards)
        and any(_is_pitts30(s) for s in shards)
    )


def _has_core_five_requirements(shards: List[Dict[str, Any]]) -> bool:
    """Core four requirements plus at least one amstertime shard."""
    return _has_core_four_requirements(shards) and any(_is_amstertime(s) for s in shards)


# Each view pools the same shard set for kappa, baselines (l2/pa/sue/sue_log), and vMF extras.
_COMBINED_VIEWS = (
    {
        "output_subdir": "combined_all",
        "plot_prefix": "ece_combined_all",
        "wandb_prefix": "Eval_combined_all",
        "description": "all evaluated shards",
        "filter_fn": _filter_shards_all,
        "exact_shards": None,
    },
    {
        "output_subdir": "combined_all_no_amst",
        "plot_prefix": "ece_combined_all_no_amst",
        "wandb_prefix": "Eval_combined_all_no_amst",
        "description": "all evaluated shards without amstertime",
        "filter_fn": _filter_shards_no_amstertime,
        "exact_shards": None,
    },
    {
        "output_subdir": "combined_6",
        "plot_prefix": "ece_combined_6",
        "wandb_prefix": "Eval_combined_6",
        "description": "all sf-xl shards, msls-val query, pitts30k, amstertime",
        "filter_fn": _filter_shards_core_five,
        "exact_shards": None,
        "required_predicate": _has_core_five_requirements,
    },
    {
        "output_subdir": "combined_5",
        "plot_prefix": "ece_combined_5",
        "wandb_prefix": "Eval_combined_5",
        "description": "all sf-xl shards, msls-val query, pitts30k",
        "filter_fn": _filter_shards_core_four,
        "exact_shards": None,
        "required_predicate": _has_core_four_requirements,
    },
)


def run_combined_ece_variants(cfg, all_panel_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run four combined views; every metric uses the same shards within each view.

    Views:
      1. combined_all — all evaluated shards
      2. combined_all_no_amst — same, minus amstertime
      3. combined_6 — all sf-xl shards + msls-val(query) + pitts30k + amstertime
      4. combined_5 — all sf-xl shards + msls-val(query) + pitts30k
    """
    combined_outputs: List[Dict[str, Any]] = []
    agg_list = _combined_shards_from_panels(all_panel_data)
    if len(agg_list) <= 1:
        logger.info("Only 1 dataset evaluated; skipping combined ECE.")
        return combined_outputs
    if cfg["model_mode"] != "uncertainty":
        return combined_outputs

    for view in _COMBINED_VIEWS:
        shards = view["filter_fn"](agg_list)
        _maybe_run_combined(
            cfg,
            shards,
            output_subdir=view["output_subdir"],
            plot_prefix=view["plot_prefix"],
            wandb_prefix=view["wandb_prefix"],
            combined_outputs=combined_outputs,
            description=view["description"],
            exact_shards=view["exact_shards"],
            required_predicate=view.get("required_predicate"),
        )

    return combined_outputs
