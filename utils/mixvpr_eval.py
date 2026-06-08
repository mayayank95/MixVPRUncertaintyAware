"""MixVPR validation: shared eval.py metrics for pre/post train and Lightning val epochs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

from data.mixvpr_val_dataset import VAL_SET_MAP, load_val_dataset_paths
from eval_metrics.dataset_eval import EvalDatasetResult, eval_dataset, evaluate_from_descriptors
from eval_metrics.eval_wandb import (
    build_training_val_ece_wandb_payload,
    ece_curves_wandb_payload,
    panel_to_wandb_metrics,
)
from utils import wandb_utils
from utils.mixvpr_train_wandb import MixVPRValEvalBundle
from validation import MIXVPR_VAL_RECALL_K_VALUES, print_recall_pretty_table

if TYPE_CHECKING:
    import pytorch_lightning as pl

logger = logging.getLogger(__name__)


def _prepare_mixvpr_eval_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy PrettyTable recall on stdout (K=1,5,10,15,20,50,100)."""
    eval_cfg = dict(cfg)
    eval_cfg["use_labels"] = True
    eval_cfg["mixvpr_recall_pretty_table"] = True
    eval_cfg["mixvpr_recall_k_values"] = MIXVPR_VAL_RECALL_K_VALUES
    eval_cfg["recall_values"] = list(MIXVPR_VAL_RECALL_K_VALUES)
    return eval_cfg


def print_mixvpr_val_recalls(
    val_set_name: str,
    recalls,
    k_values: Optional[List[int]] = None,
) -> None:
    """Legacy MixVPR stdout: PrettyTable ``Performances on {dataset}``."""
    import sys

    k_values = list(k_values or MIXVPR_VAL_RECALL_K_VALUES)
    print_recall_pretty_table(recalls, k_values, val_set_name)
    sys.stdout.flush()


def _val_set_names(cfg: Dict[str, Any]) -> List[str]:
    names = cfg.get("mixvpr_val_sets") or []
    if isinstance(names, str):
        return [s.strip() for s in names.split(",") if s.strip()]
    return list(names)


def mixvpr_val_num_references(val_set_name: str, val_dataset) -> int:
    if "pitts" in val_set_name:
        return int(val_dataset.dbStruct.numDb)
    if "msls" in val_set_name:
        return int(val_dataset.num_references)
    raise NotImplementedError(f"Unknown validation set: {val_set_name}")


def panel_to_train_val_metrics(val_set_name: str, panel: Dict[str, Any]) -> Dict[str, float]:
    """Lightning / checkpoint keys: ``pitts30k_val/R1``, ``.../ece_kappa_recall_01``, etc."""
    metrics: Dict[str, float] = {}
    for k, v in panel.get("recalls", {}).items():
        metrics[f"{val_set_name}/R{int(k)}"] = float(v)
    kappa_ece = panel.get("ece_recall", {}).get("kappa", {})
    for k, v in kappa_ece.items():
        metrics[f"{val_set_name}/ece_kappa_recall_{int(k):02d}"] = float(v)
    kappa_map = panel.get("ece_map", {}).get("kappa", {})
    for k, v in kappa_map.items():
        metrics[f"{val_set_name}/ece_kappa_map_{int(k):02d}"] = float(v)
    if "kappa" in panel.get("ece_ap", {}):
        metrics[f"{val_set_name}/ece_kappa_ap"] = float(panel["ece_ap"]["kappa"])
    for stat_k, stat_v in (panel.get("variance_stat") or {}).items():
        if isinstance(stat_v, (int, float)):
            metrics[f"{val_set_name}/{stat_k}"] = float(stat_v)
    return metrics


def log_mixvpr_eval_result(
    result: EvalDatasetResult,
    val_set_name: str,
    cfg: Dict[str, Any],
    *,
    pl_module: Optional["pl.LightningModule"] = None,
    wandb_step: Optional[int] = None,
    dataset_output_dir: Optional[Path] = None,
    log_wandb: bool = True,
    training_wandb_layout: bool = False,
) -> Dict[str, float]:
    """Log scalars to Lightning and W&B; return flat train-style metric dict."""
    panel = result.panel_data or {}
    train_metrics = panel_to_train_val_metrics(val_set_name, panel)

    if pl_module is not None:
        for k, v in train_metrics.items():
            pl_module.log(k, v, prog_bar=False, logger=True, on_epoch=True)

    if cfg.get("use_wandb") and panel and log_wandb:
        step = wandb_step if wandb_step is not None else 0
        out_dir = dataset_output_dir
        if out_dir is None and cfg.get("log_dir"):
            out_dir = Path(cfg["log_dir"]) / "eval" / panel.get("dataset_name", val_set_name)

        if training_wandb_layout:
            payload = build_training_val_ece_wandb_payload(
                panel, result.wandb_images, out_dir, cfg
            )
        else:
            payload = dict(panel_to_wandb_metrics(panel, wandb_step=wandb_step))
            dataset_name = panel.get("dataset_name", val_set_name)
            if result.wandb_images:
                payload.update(
                    wandb_utils._collect_eval_media_for_dataset(dataset_name, result.wandb_images)
                )
            ece_media = ece_curves_wandb_payload(dataset_name, out_dir)
            if ece_media:
                payload.update(ece_media)

        if payload:
            wandb_utils.log_wandb(payload, step=step)

    return train_metrics


def run_mixvpr_validation_eval(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    *,
    wandb_step: Optional[int] = None,
    use_descriptor_cache: bool = False,
) -> Dict[str, float]:
    """
    Full ``eval_dataset`` pass for each ``mixvpr_val_sets`` entry (extract + shared metrics).
    """
    eval_cfg = _prepare_mixvpr_eval_cfg(cfg)
    if not eval_cfg.get("log_dir"):
        raise ValueError("log_dir is required for eval outputs (set in build_config).")

    val_paths = load_val_dataset_paths(eval_cfg.get("config"))
    metrics: Dict[str, float] = {}

    for val_set_name in _val_set_names(eval_cfg):
        key = val_set_name.lower()
        if key not in VAL_SET_MAP:
            logger.warning("Skipping unknown mixvpr val set %r", val_set_name)
            continue

        entry_name, split_key = VAL_SET_MAP[key]
        eval_ds_path = val_paths[entry_name][split_key]
        if not eval_ds_path.is_dir():
            logger.warning("[%s] path not found: %s", val_set_name, eval_ds_path)
            continue

        logger.info(
            "\n>>> eval_dataset: %s (%s / %s) -> %s",
            val_set_name,
            entry_name,
            split_key,
            eval_ds_path,
        )

        results = eval_dataset(
            eval_cfg,
            model,
            device,
            val_set_name,
            eval_ds_path,
            wandb_step=wandb_step,
            log_dataset_info=True,
            base_dataset_name=entry_name,
            use_descriptor_cache=use_descriptor_cache,
        )
        print_mixvpr_val_recalls(val_set_name, results.recalls, eval_cfg.get("recall_values"))
        metrics.update(
            log_mixvpr_eval_result(
                results,
                val_set_name,
                eval_cfg,
                wandb_step=wandb_step,
                training_wandb_layout=bool(eval_cfg.get("use_wandb")),
            )
        )

    return metrics


def run_mixvpr_lightning_val_eval(
    pl_module: "pl.LightningModule",
    cfg: Dict[str, Any],
) -> List[MixVPRValEvalBundle]:
    """
    After ``validation_step`` buffers are full: one ``evaluate_from_descriptors`` per val set.
    W&B logging is deferred to ``MixVPRTrainWandbCallback`` (train/val/ece panels).
    """
    dm = pl_module.trainer.datamodule
    feats_by_dl = getattr(pl_module, "_val_feats_by_dl", None) or {}
    vars_by_dl = getattr(pl_module, "_val_vars_by_dl", None) or {}
    bundles: List[MixVPRValEvalBundle] = []
    pl_module._mixvpr_val_eval_bundles = bundles

    if not getattr(dm, "val_datasets", None):
        return bundles

    eval_cfg = _prepare_mixvpr_eval_cfg(cfg)
    step = pl_module.trainer.current_epoch
    log_dir = eval_cfg.get("log_dir")

    for i, (val_set_name, val_dataset) in enumerate(
        zip(dm.val_set_names, dm.val_datasets)
    ):
        feat_batches = feats_by_dl.get(i, [])
        var_batches = vars_by_dl.get(i, [])
        if not feat_batches or not var_batches:
            continue

        feats = torch.concat(feat_batches, dim=0).float().numpy()
        variances = torch.concat(var_batches, dim=0).float().numpy()
        num_references = mixvpr_val_num_references(val_set_name, val_dataset)

        db_desc = feats[:num_references]
        q_desc = feats[num_references:]
        db_var = variances[:num_references]
        q_var = variances[num_references:]

        out_dir = None
        if log_dir:
            out_dir = Path(log_dir) / "wandb_inspect" / f"epoch_{step:03d}" / val_set_name

        key = val_set_name.lower()
        base_name = VAL_SET_MAP[key][0] if key in VAL_SET_MAP else val_set_name

        result = evaluate_from_descriptors(
            eval_cfg,
            val_dataset._ds,
            val_set_name,
            db_desc,
            q_desc,
            db_var,
            q_var,
            wandb_step=step,
            dataset_output_dir=out_dir,
            base_dataset_name=base_name,
            save_recalls=False,
        )
        print_mixvpr_val_recalls(val_set_name, result.recalls, eval_cfg.get("recall_values"))
        log_mixvpr_eval_result(
            result,
            val_set_name,
            eval_cfg,
            pl_module=pl_module,
            wandb_step=step,
            dataset_output_dir=out_dir,
            log_wandb=False,
        )
        bundles.append(MixVPRValEvalBundle(val_set_name, result, out_dir))
    print("\n\n")
    return bundles
