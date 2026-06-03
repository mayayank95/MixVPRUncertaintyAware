"""Per-dataset evaluation: extraction, retrieval, uncertainty metrics, W&B payloads."""
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import torch

from data.test_dataset import TestDataset
from eval_metrics import visualizations
from eval_metrics.eval_ece_sh import compute_ece, compute_ece_pairwise
from eval_metrics.eval_wandb import build_eval_outputs
from eval_metrics.extraction import extract_descriptors
from eval_metrics.joint_kappa import compute_joint_kappa
from eval_metrics.log_utils import log_metric_summary
from eval_metrics.retrieval import log_retrieval_results, run_retrieval
from eval_metrics.uncertainty import compute_auc_pr, compute_uncertainty_statistics
from eval_metrics.baselines import compute_baselines

logger = logging.getLogger(__name__)


class EvalDatasetResult(NamedTuple):
    recalls: np.ndarray
    recalls_str: str
    map_at_k: Optional[List[float]]
    panel_data: Dict[str, Any]
    wandb_images: Optional[Dict[str, Any]]
    db_desc: np.ndarray
    db_var: np.ndarray


def _compute_uncertainty_block(
    args: Dict[str, Any],
    test_ds: TestDataset,
    predictions,
    distances,
    positives_per_query,
    q_var,
    db_var,
    q_desc,
    db_desc,
    dataset_output_dir: Path,
    wandb_step: Optional[int],
):
    """Compute uncertainty ECE, eval-only AUC-PR, and variance statistics."""
    uncertainty_aucpr = None
    uncertainty_stats = None
    uncertainty_ece_results: Dict[str, Any] = {}
    uncertainty_raw_scores: Dict[str, Any] = {}
    save_plots = args.get("save_plots", False) or args.get("use_wandb", False)

    stats_output_dir = None
    if save_plots:
        if dataset_output_dir.exists() and not dataset_output_dir.is_dir():
            logger.warning(
                "Output path %s exists as a file; skipping variance distribution plot.",
                dataset_output_dir,
            )
        else:
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            stats_output_dir = dataset_output_dir

    uncertainty_stats = compute_uncertainty_statistics(
        np.vstack([db_var, q_var]),
        stats_output_dir,
        num_database=len(db_var),
    )

    if args["use_labels"] and positives_per_query is not None:
        if args["dry_run"]:
            q_var = q_var[: max(1, args.get("num_queries_to_save", 3))]

        ece_metrics = args.get("ece_metrics") or ["recall"]
        if save_plots:
            dataset_output_dir.mkdir(parents=True, exist_ok=True)

        is_vmf = args.get("uncertainty_loss", "gaussian_nll").lower() == "vmf"
        ece_variants = [("kappa", q_var, "ece_kappa.png")]

        if is_vmf:
            joint_kappa = compute_joint_kappa(
                predictions,
                q_var,
                db_var,
                q_desc,
                db_desc,
            )
            ece_joint_kappa_pairwise = None
            ece_variants.append(("joint_kappa", joint_kappa[:, 0][:, None], "ece_joint_kappa.png"))

        for name, variances, plot_filename in ece_variants:
            ece = compute_ece(
                predictions,
                positives_per_query,
                variances,
                n_values=args["recall_values"],
                output_dir=dataset_output_dir if save_plots else None,
                metrics=ece_metrics,
                distances=distances,
                uncertainty_loss=args.get("uncertainty_loss", "gaussian_nll"),
                **({"plot_filename": plot_filename} if plot_filename else {}),
                zoom_threshold=args.get("ece_zoom_threshold", 0.001),
                bin_mode=args.get("ece_bin_mode", "zoom"),
                vmf_kappa_floor=args.get("ece_vmf_kappa_floor", False),
                percentile_two_sided=bool(args.get("ece_two_sided", False)),
            )
            uncertainty_ece_results[name] = ece
            uncertainty_raw_scores[name] = variances

        if is_vmf:
            ece_joint_kappa_pairwise = compute_ece_pairwise(
                predictions,
                positives_per_query,
                joint_kappa,
                n_values=args["recall_values"],
                output_dir=dataset_output_dir if save_plots else None,
                plot_filename="ece_pairwise_joint_kappa.png",
                zoom_threshold=args.get("ece_zoom_threshold", 0.001),
                bin_mode=args.get("ece_bin_mode", "zoom"),
                uncertainty_loss="vmf",
                vmf_kappa_floor=args.get("ece_vmf_kappa_floor", False),
                percentile_two_sided=bool(args.get("ece_two_sided", False)),
                metrics=ece_metrics,
            )
            uncertainty_ece_results["pairwise_joint_kappa"] = ece_joint_kappa_pairwise
            uncertainty_raw_scores["pairwise_joint_kappa"] = joint_kappa

        kappa_per_query = np.mean(q_var, axis=1)
        run_auc_pr = wandb_step is None and not args.get("skip_auc_pr")
        if run_auc_pr:
            logger.debug("Computing detailed uncertainty AUC-PR...")
            uncertainty_aucpr = {}
            auc_pr_variants = [
                (
                    "kappa",
                    kappa_per_query,
                    args.get("uncertainty_loss", "gaussian_nll").lower() == "vmf",
                )
            ]
            if is_vmf:
                auc_pr_variants.append(("joint_kappa", joint_kappa[:, 0], True))
                auc_pr_variants.append(
                    ("pairwise_joint_kappa", np.max(joint_kappa, axis=1), True)
                )

            for name, scores, higher_is_confidence in auc_pr_variants:
                uncertainty_aucpr[name] = compute_auc_pr(
                    predictions,
                    positives_per_query,
                    scores,
                    higher_is_confidence=higher_is_confidence,
                    label=name,
                )
                uncertainty_aucpr[f"{name}_min"] = float(np.min(scores))
                uncertainty_aucpr[f"{name}_max"] = float(np.max(scores))
                uncertainty_aucpr[f"{name}_mean"] = float(np.mean(scores))

        metric_summaries = [
            ("kappa", uncertainty_ece_results.get("kappa"), kappa_per_query, "kappa"),
        ]
        if is_vmf:
            metric_summaries.extend([
                ("joint_kappa", uncertainty_ece_results.get("joint_kappa"), joint_kappa[:, 0], "joint_kappa"),
                (
                    "pairwise_joint_kappa",
                    uncertainty_ece_results.get("pairwise_joint_kappa"),
                    joint_kappa,
                    None,
                ),
            ])

        for metric_name, ece_result, scores, auc_key in metric_summaries:
            log_metric_summary(
                metric_name,
                ece_result,
                scores,
                uncertainty_aucpr.get(auc_key) if uncertainty_aucpr and auc_key else None,
                ece_only=(metric_name == "pairwise_joint_kappa"),
            )

    return uncertainty_aucpr, uncertainty_ece_results, uncertainty_raw_scores, uncertainty_stats


def eval_dataset(
    args,
    model,
    device,
    dataset_name,
    eval_ds_path,
    queries_folder_name="queries",
    wandb_step=None,
    log_dataset_info=True,
    base_dataset_name=None,
    cached_db_desc=None,
    cached_db_var=None,
    use_descriptor_cache=True,
) -> EvalDatasetResult:
    """
    Evaluate the model on a single dataset.
    Extracts features once, then computes recalls and uncertainty metrics.
    """
    model = model.eval()
    dataset_output_dir = Path(args["log_dir"]) / "eval" / dataset_name
    if not args["dry_run"] or args.get("use_wandb"):
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

    test_ds = TestDataset(
        f"{eval_ds_path}/database",
        f"{eval_ds_path}/{queries_folder_name}",
        positive_dist_threshold=args["positive_dist_threshold"],
        image_size=args.get("image_size"),
        use_labels=args["use_labels"],
        resize_test_imgs=args.get("resize_test_imgs", False),
    )
    if log_dataset_info:
        logger.info(f"Testing on {test_ds}")

    db_desc, q_desc, db_var, q_var = extract_descriptors(
        args,
        model,
        device,
        test_ds,
        dataset_name,
        base_dataset_name=base_dataset_name,
        cached_db_desc=cached_db_desc,
        cached_db_var=cached_db_var,
        use_descriptor_cache=use_descriptor_cache,
    )

    predictions, distances, recalls, recalls_str, positives_per_query, map_at_k = run_retrieval(
        args,
        test_ds,
        db_desc,
        q_desc,
        dataset_output_dir,
        save_recalls=use_descriptor_cache,
    )

    uncertainty_aucpr = baseline_results = uncertainty_stats = None
    uncertainty_ece_results: Dict[str, Any] = {}
    uncertainty_raw_scores: Dict[str, Any] = {}
    save_plots = args.get("save_plots", False) or args.get("use_wandb", False)

    if args["use_labels"]:
        log_retrieval_results(dataset_name, recalls_str, recalls, map_at_k, args["recall_values"])

    if args["model_mode"] == "uncertainty":
        uncertainty_aucpr, uncertainty_ece_results, uncertainty_raw_scores, uncertainty_stats = (
            _compute_uncertainty_block(
                args,
                test_ds,
                predictions,
                distances,
                positives_per_query,
                q_var,
                db_var,
                q_desc,
                db_desc,
                dataset_output_dir,
                wandb_step,
            )
        )
        if (
            args["use_labels"]
            and positives_per_query is not None
            and wandb_step is None
            and not args.get("skip_baselines")
        ):
            logger.info("Computing baselines...")
            baseline_results = compute_baselines(
                predictions,
                positives_per_query,
                distances,
                database_utms=getattr(test_ds, "database_utms", None),
                n_values=args.get("recall_values"),
                output_dir=dataset_output_dir if save_plots else None,
                zoom_threshold=args.get("ece_zoom_threshold", 0.001),
                bin_mode=args.get("ece_bin_mode", "zoom"),
                skip_auc_pr=args.get("skip_auc_pr", False),
            )

    if save_plots and args.get("num_preds_to_save", 0) != 0:
        preds_to_save = predictions[:, : args["num_preds_to_save"]]
        pred_distances = distances[:, : args["num_preds_to_save"]]
        query_variances = np.mean(q_var, axis=1)
        db_var_plot = db_var if args["model_mode"] == "uncertainty" else None
        visualizations.save_preds(
            preds_to_save,
            test_ds,
            str(dataset_output_dir),
            args["save_only_wrong_preds"],
            args["use_labels"],
            args["num_queries_to_save"],
            distances=pred_distances,
            query_variances=query_variances,
            db_variances=db_var_plot,
            q_desc=q_desc,
            db_desc=db_desc,
        )

    panel_data, wandb_images = build_eval_outputs(
        dataset_name,
        args,
        recalls,
        map_at_k,
        uncertainty_ece_results,
        uncertainty_raw_scores,
        uncertainty_aucpr,
        baseline_results,
        uncertainty_stats,
        dataset_output_dir=dataset_output_dir,
        save_plots=save_plots,
        base_dataset_name=base_dataset_name or dataset_name,
        query_folder_name=queries_folder_name,
    )

    if (
        args["model_mode"] == "uncertainty"
        and args["use_labels"]
        and positives_per_query is not None
        and not args["dry_run"]
    ):
        panel_data["pool"] = {
            "predictions": predictions,
            "positives_per_query": positives_per_query,
            "q_var": q_var,
            "distances": distances,
        }

    return EvalDatasetResult(
        recalls=recalls,
        recalls_str=recalls_str,
        map_at_k=map_at_k,
        panel_data=panel_data,
        wandb_images=wandb_images,
        db_desc=db_desc,
        db_var=db_var,
    )
