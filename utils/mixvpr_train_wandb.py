"""W&B panels for MixVPR Lightning training (train / val / ece), updated each epoch."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch

from eval_metrics.eval_wandb import build_training_val_ece_wandb_payload
from eval_metrics.uncertainty import compute_variance_summary
from utils import wandb_utils

if TYPE_CHECKING:
    from eval_metrics.dataset_eval import EvalDatasetResult

logger = logging.getLogger(__name__)


class MixVPRValEvalBundle:
    """One validation set result from ``run_mixvpr_lightning_val_eval``."""

    __slots__ = ("val_set_name", "result", "output_dir")

    def __init__(
        self,
        val_set_name: str,
        result: "EvalDatasetResult",
        output_dir: Optional[Path],
    ):
        self.val_set_name = val_set_name
        self.result = result
        self.output_dir = output_dir


def build_train_panel_metrics(
    active_losses: List[str],
    epoch_losses: List[float],
    epoch_losses_basic: List[float],
    epoch_losses_uncertainty: List[float],
    epoch_variances: List[float],
    epoch_losses_ms: Optional[List[float]] = None,
    epoch_losses_r_kappa: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Scalars for the ``train/`` W&B panel (mirrors ``val/`` variance stats, no plot)."""
    metrics: Dict[str, Any] = {}
    n_losses = len([x for x in active_losses if x in ("basic", "uncertainty")])

    if epoch_losses:
        metrics["train/loss"] = float(np.mean(epoch_losses))
    if n_losses > 1 and epoch_losses:
        metrics["train/loss_total"] = float(np.mean(epoch_losses))
    if epoch_losses_basic:
        metrics["train/loss_basic"] = float(np.mean(epoch_losses_basic))
    if epoch_losses_uncertainty:
        metrics["train/loss_uncertainty"] = float(np.mean(epoch_losses_uncertainty))
    if epoch_losses_ms:
        metrics["train/loss_ms"] = float(np.mean(epoch_losses_ms))
    if epoch_losses_r_kappa:
        metrics["train/loss_r_kappa"] = float(np.mean(epoch_losses_r_kappa))
    if epoch_losses_ms and epoch_losses_r_kappa:
        metrics["train/loss_total"] = float(np.mean(epoch_losses))

    if epoch_variances:
        vs = compute_variance_summary(np.asarray(epoch_variances, dtype=np.float64))
        metrics["train/variances_min"] = vs["min"]
        metrics["train/variances_max"] = vs["max"]
        metrics["train/variances_mean"] = vs["mean"]
        metrics["train/variances_std"] = vs["std"]
        metrics["train/variances_median"] = vs["median"]
    return metrics


def log_mixvpr_training_epoch_wandb(
    cfg: Dict[str, Any],
    step: int,
    train_metrics: Dict[str, Any],
    val_eval: Optional[MixVPRValEvalBundle],
    *,
    val_variance_median: Optional[float] = None,
) -> None:
    """Log train + val + ece panels at ``step`` (one ``wandb.log`` per epoch)."""
    if not cfg.get("use_wandb"):
        return

    payload: Dict[str, Any] = {"epoch": step}
    payload.update(train_metrics)

    if val_eval is not None and val_eval.result.panel_data:
        panel = val_eval.result.panel_data
        payload.update(
            build_training_val_ece_wandb_payload(
                panel,
                val_eval.result.wandb_images,
                val_eval.output_dir,
                cfg,
            )
        )
        if val_variance_median is not None and "val/variances_median" not in payload:
            payload["val/variances_median"] = float(val_variance_median)

    if len(payload) > 1:
        wandb_utils.log_wandb(payload, step=step)


def log_mixvpr_train_panel_wandb(cfg: Dict[str, Any], step: int, train_metrics: Dict[str, Any]) -> None:
    """Log ``train/`` scalars at end of each training epoch."""
    if not cfg.get("use_wandb") or not train_metrics:
        return
    wandb_utils.log_wandb({"epoch": step, **train_metrics}, step=step)


def log_mixvpr_val_ece_panel_wandb(
    cfg: Dict[str, Any],
    step: int,
    val_eval: Optional[MixVPRValEvalBundle],
) -> None:
    """Log ``val/`` + ``ece/`` + ``ece_curves/`` + ``bins_distribution/`` after validation eval."""
    if not cfg.get("use_wandb") or val_eval is None or not val_eval.result.panel_data:
        return

    panel = val_eval.result.panel_data
    payload: Dict[str, Any] = {"epoch": step}
    payload.update(
        build_training_val_ece_wandb_payload(
            panel,
            val_eval.result.wandb_images,
            val_eval.output_dir,
            cfg,
        )
    )
    stats = panel.get("variance_stat") or {}
    qmed = stats.get("q_median", stats.get("median"))
    if isinstance(qmed, (int, float)) and "val/variances_median" not in payload:
        payload["val/variances_median"] = float(qmed)

    if len(payload) > 1:
        wandb_utils.log_wandb(payload, step=step)


class MixVPRTrainWandbCallback(pl.Callback):
    """Epoch-level W&B: train/, val/, ece/, ece_curves/, bins_distribution/ panels."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.active_losses = cfg.get("losses") or []
        if isinstance(self.active_losses, str):
            self.active_losses = [s.strip() for s in self.active_losses.split(",") if s.strip()]

        self.epoch_losses: List[float] = []
        self.epoch_losses_basic: List[float] = []
        self.epoch_losses_uncertainty: List[float] = []
        self.epoch_losses_ms: List[float] = []
        self.epoch_losses_r_kappa: List[float] = []
        self.epoch_variances: List[float] = []
        self._cached_train_metrics: Dict[str, Any] = {}

    def _enabled(self) -> bool:
        return bool(self.cfg.get("use_wandb"))

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.epoch_losses = []
        self.epoch_losses_basic = []
        self.epoch_losses_uncertainty = []
        self.epoch_losses_ms = []
        self.epoch_losses_r_kappa = []
        self.epoch_variances = []
        self._cached_train_metrics = {}

    def record_train_batch(
        self,
        pl_module: pl.LightningModule,
        loss: torch.Tensor,
        variances: Optional[torch.Tensor],
        descriptors: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if not self._enabled():
            return
        basic_v, unc_v, ms_v, r_kappa_v = self._batch_loss_parts(
            pl_module, descriptors, labels, variances
        )
        self._append_loss_batch(
            float(loss.detach().item()),
            basic_v,
            unc_v,
            ms_v,
            r_kappa_v,
            variances,
        )

    def _batch_loss_parts(
        self,
        pl_module: pl.LightningModule,
        descriptors: torch.Tensor,
        labels: torch.Tensor,
        variances: Optional[torch.Tensor],
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        basic_v = None
        unc_v = None
        ms_v = None
        r_kappa_v = None

        if "basic" in self.active_losses and pl_module.loss_basic is not None:
            if pl_module.miner is not None:
                miner_out = pl_module.miner(descriptors, labels)
                l_basic = pl_module.loss_basic(descriptors, labels, miner_out)
            else:
                l_basic = pl_module.loss_basic(descriptors, labels)
            if isinstance(l_basic, tuple):
                l_basic = l_basic[0]
            basic_v = float(l_basic.item())

        if "uncertainty" in self.active_losses and variances is not None:
            if getattr(pl_module, "_use_kappa_ms", False):
                indices_tuple = (
                    pl_module.miner(descriptors, labels)
                    if pl_module.miner is not None
                    else None
                )
                _, ms, reg = pl_module._kappa_ms_loss(
                    descriptors, labels, variances, indices_tuple, return_parts=True
                )
                ms_v = float(ms.item())
                r_kappa_v = float(reg.item())
            else:
                unc_v = float(pl_module._uncertainty_loss(descriptors, labels, variances).item())

        return basic_v, unc_v, ms_v, r_kappa_v

    def _append_loss_batch(
        self,
        total_loss: float,
        basic_v: Optional[float],
        unc_v: Optional[float],
        ms_v: Optional[float],
        r_kappa_v: Optional[float],
        variances: Optional[torch.Tensor],
    ) -> None:
        self.epoch_losses.append(total_loss)
        if basic_v is not None:
            self.epoch_losses_basic.append(basic_v)
        if unc_v is not None:
            self.epoch_losses_uncertainty.append(unc_v)
        if ms_v is not None:
            self.epoch_losses_ms.append(ms_v)
        if r_kappa_v is not None:
            self.epoch_losses_r_kappa.append(r_kappa_v)
        if variances is not None:
            per_sample = variances.detach().float().mean(dim=-1).cpu().numpy()
            self.epoch_variances.extend(per_sample.reshape(-1).tolist())

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._enabled():
            return
        self._cached_train_metrics = build_train_panel_metrics(
            self.active_losses,
            self.epoch_losses,
            self.epoch_losses_basic,
            self.epoch_losses_uncertainty,
            self.epoch_variances,
            self.epoch_losses_ms,
            self.epoch_losses_r_kappa,
        )
        log_mixvpr_train_panel_wandb(self.cfg, trainer.current_epoch, self._cached_train_metrics)

    def log_val_ece_wandb(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Call from ``VPRModel.on_validation_epoch_end`` after ``run_mixvpr_lightning_val_eval``."""
        if not self._enabled():
            return

        step = trainer.current_epoch
        bundles: List[MixVPRValEvalBundle] = getattr(pl_module, "_mixvpr_val_eval_bundles", []) or []
        primary = bundles[0] if bundles else None

        if primary and primary.result.panel_data:
            recalls = primary.result.recalls
            r1 = float(recalls[0]) if len(recalls) else 0.0
            logger.info(
                "W&B val epoch %s: %s R@1=%.4f",
                step,
                primary.val_set_name,
                r1,
            )

        log_mixvpr_val_ece_panel_wandb(self.cfg, step, primary)


def attach_wandb_callback(cfg: Dict[str, Any], callbacks: List[pl.Callback]) -> Optional[MixVPRTrainWandbCallback]:
    """Init W&B and register callback. Returns callback when enabled."""
    if not cfg.get("use_wandb"):
        return None
    wandb_utils.init_wandb(cfg, job_type="train")
    cb = MixVPRTrainWandbCallback(cfg)
    callbacks.append(cb)
    return cb
