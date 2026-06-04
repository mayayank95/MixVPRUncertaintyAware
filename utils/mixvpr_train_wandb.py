"""W&B inspection for MixVPR Lightning training (losses + variance stats)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch

from eval_metrics.uncertainty import compute_variance_summary, plot_variance_distribution
from utils import wandb_utils

logger = logging.getLogger(__name__)


class MixVPRTrainWandbCallback(pl.Callback):
    """
    Epoch-level W&B logging for train.py (inspired by train_exp.py + wandb_utils.log_train_epoch).

    Tracks per epoch:
      - train/loss, train/loss_uncertainty, train/loss_basic (when applicable)
      - train/variance_{min,max,mean,std}
      - val/recall@k from Lightning callback_metrics
      - val/variance_* + variance distribution plot (when model outputs variances)
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.active_losses = cfg.get("losses") or []
        if isinstance(self.active_losses, str):
            self.active_losses = [s.strip() for s in self.active_losses.split(",") if s.strip()]

        self.epoch_losses: List[float] = []
        self.epoch_losses_basic: List[float] = []
        self.epoch_losses_uncertainty: List[float] = []
        self.epoch_variances: List[float] = []

        self.val_variance_vectors: List[np.ndarray] = []
        self.best_val_r1 = 0.0

    def _enabled(self) -> bool:
        return bool(self.cfg.get("use_wandb"))

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.epoch_losses = []
        self.epoch_losses_basic = []
        self.epoch_losses_uncertainty = []
        self.epoch_variances = []

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
        self.epoch_losses.append(float(loss.detach().item()))

        if "basic" in self.active_losses and pl_module.loss_basic is not None:
            with torch.no_grad():
                if pl_module.miner is not None:
                    miner_out = pl_module.miner(descriptors, labels)
                    l_basic = pl_module.loss_basic(descriptors, labels, miner_out)
                else:
                    l_basic = pl_module.loss_basic(descriptors, labels)
                if isinstance(l_basic, tuple):
                    l_basic = l_basic[0]
                self.epoch_losses_basic.append(float(l_basic.item()))

        if "uncertainty" in self.active_losses and variances is not None:
            with torch.no_grad():
                l_unc = pl_module._uncertainty_loss(descriptors, labels, variances)
                self.epoch_losses_uncertainty.append(float(l_unc.item()))
            per_sample = variances.detach().float().mean(dim=-1).cpu().numpy()
            self.epoch_variances.extend(per_sample.reshape(-1).tolist())

    def record_val_batch(self, variances: Optional[torch.Tensor]) -> None:
        if not self._enabled() or variances is None:
            return
        self.val_variance_vectors.append(
            variances.detach().float().cpu().numpy()
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._enabled() or not self.epoch_losses:
            return
        step = trainer.current_epoch
        metrics: Dict[str, Any] = {"epoch": step}
        metrics["train/loss"] = float(np.mean(self.epoch_losses))
        if self.epoch_losses_basic:
            metrics["train/loss_basic"] = float(np.mean(self.epoch_losses_basic))
        if self.epoch_losses_uncertainty:
            metrics["train/loss_uncertainty"] = float(np.mean(self.epoch_losses_uncertainty))
        if self.epoch_variances:
            vs = compute_variance_summary(np.asarray(self.epoch_variances, dtype=np.float64))
            metrics["train/variance_mean"] = vs["mean"]
            metrics["train/variance_std"] = vs["std"]
            metrics["train/variance_min"] = vs["min"]
            metrics["train/variance_max"] = vs["max"]
            metrics["train/variance_median"] = vs["median"]
        wandb_utils.log_wandb(metrics, step=step)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.val_variance_vectors = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._enabled():
            return

        step = trainer.current_epoch
        cm = {
            k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
            for k, v in trainer.callback_metrics.items()
        }

        recalls, val_set = _extract_val_recalls(cm, self.cfg.get("mixvpr_ckpt_monitor", "pitts30k_val/R1"))
        if recalls is not None and len(recalls) > 0:
            self.best_val_r1 = max(self.best_val_r1, float(recalls[0]))

        eval_wandb_metrics: Dict[str, Any] = {}
        for k, v in cm.items():
            if "/R" in k or k.startswith("val/"):
                eval_wandb_metrics[k] = v

        mean_q_var = std_q_var = min_q_var = max_q_var = None
        wandb_images: Optional[Dict[str, Path]] = None
        if self.val_variance_vectors:
            all_var = np.concatenate(self.val_variance_vectors, axis=0)
            vs = compute_variance_summary(all_var.reshape(all_var.shape[0], -1).mean(axis=1))
            mean_q_var = vs["mean"]
            std_q_var = vs["std"]
            min_q_var = vs["min"]
            max_q_var = vs["max"]
            log_dir = self.cfg.get("log_dir")
            if log_dir:
                out_dir = Path(log_dir) / "wandb_inspect" / f"epoch_{step:03d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                plot_variance_distribution(all_var, out_dir)
                plot_path = out_dir / "variance_distribution.png"
                if plot_path.exists():
                    wandb_images = {"variance_distribution": plot_path.resolve()}

        recalls_arr = recalls if recalls is not None else np.array([0.0])
        wandb_utils.log_train_epoch(
            self.cfg,
            step,
            recalls_arr,
            map_at_k=None,
            best_val_recall1=self.best_val_r1,
            active_losses=self.active_losses,
            epoch_variances=self.epoch_variances if self.epoch_variances else None,
            epoch_losses=self.epoch_losses,
            epoch_losses_ce=self.epoch_losses_basic if self.epoch_losses_basic else None,
            epoch_losses_gnll=self.epoch_losses_uncertainty
            if self.epoch_losses_uncertainty
            else None,
            mean_query_variance=mean_q_var,
            std_query_variance=std_q_var,
            min_query_variance=min_q_var,
            max_query_variance=max_q_var,
            eval_wandb_metrics=eval_wandb_metrics,
            eval_wandb_images=wandb_images,
        )
        if val_set:
            logger.info(
                "W&B val epoch %s: %s R@1=%.4f variance_mean=%s",
                step,
                val_set,
                float(recalls_arr[0]) if len(recalls_arr) else 0.0,
                f"{mean_q_var:.4f}" if mean_q_var is not None else "n/a",
            )


def _extract_val_recalls(
    callback_metrics: Dict[str, float], ckpt_monitor: str
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Build recall@k array from Lightning logs (pitts30k_val/R1, ...)."""
    val_set = ckpt_monitor.split("/")[0] if "/" in ckpt_monitor else None
    if val_set and f"{val_set}/R1" in callback_metrics:
        k_values = [1, 5, 10, 15, 20, 50, 100]
        recalls = []
        for k in k_values:
            key = f"{val_set}/R{k}"
            if key not in callback_metrics:
                break
            recalls.append(callback_metrics[key])
        if recalls:
            return np.asarray(recalls, dtype=np.float64), val_set

    for key in sorted(callback_metrics):
        if key.endswith("/R1"):
            val_set = key.rsplit("/", 1)[0]
            k_values = [1, 5, 10, 15, 20, 50, 100]
            recalls = []
            for k in k_values:
                rk = f"{val_set}/R{k}"
                if rk not in callback_metrics:
                    break
                recalls.append(callback_metrics[rk])
            if recalls:
                return np.asarray(recalls, dtype=np.float64), val_set
    return None, None


def attach_wandb_callback(cfg: Dict[str, Any], callbacks: List[pl.Callback]) -> Optional[MixVPRTrainWandbCallback]:
    """Init W&B and register callback. Returns callback when enabled."""
    if not cfg.get("use_wandb"):
        return None
    wandb_utils.init_wandb(cfg, job_type="train")
    cb = MixVPRTrainWandbCallback(cfg)
    callbacks.append(cb)
    return cb
