"""Per-metric ECE early stopping and frozen checkpoints for MixVPR Lightning training."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.early_stop_utils import initial_best_for_metric, is_improvement

logger = logging.getLogger(__name__)

MetricSpec = Tuple[str, str, str, str]  # (canonical_id, monitor, mode, filename_tag)


class FrozenModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that stops saving after the metric is locked (patience exhausted)."""

    def __init__(self, *args, metric_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_id = metric_id
        self.frozen = False

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.frozen:
            return
        super().on_validation_end(trainer, pl_module)


class MultiMetricEarlyStop(pl.Callback):
    """Stop when every tracked metric exhausts patience; freeze each metric's checkpoint then."""

    def __init__(
        self,
        metric_specs: List[MetricSpec],
        patience: int,
        ckpt_callbacks: Dict[str, FrozenModelCheckpoint],
    ):
        super().__init__()
        self.metric_specs = metric_specs
        self.patience = patience
        self.ckpt_callbacks = ckpt_callbacks
        self.not_improved: Dict[str, int] = {m[0]: 0 for m in metric_specs}
        self.best_values: Dict[str, float] = {
            m[0]: initial_best_for_metric(m[0]) for m in metric_specs
        }
        self.frozen: Dict[str, bool] = {m[0]: False for m in metric_specs}

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        for metric_id, monitor, mode, _tag in self.metric_specs:
            if self.frozen[metric_id]:
                continue

            current = trainer.callback_metrics.get(monitor)
            if current is None:
                continue
            current_f = float(current)

            if is_improvement(metric_id, current_f, self.best_values[metric_id]):
                self.best_values[metric_id] = current_f
                self.not_improved[metric_id] = 0
            else:
                self.not_improved[metric_id] += 1
                if self.not_improved[metric_id] >= self.patience:
                    self.frozen[metric_id] = True
                    ckpt_cb = self.ckpt_callbacks.get(metric_id)
                    if ckpt_cb is not None:
                        ckpt_cb.frozen = True
                    logger.info(
                        "Metric %r locked after %s epochs without improvement (patience=%s).",
                        metric_id,
                        self.not_improved[metric_id],
                        self.patience,
                    )

        if all(self.frozen.values()):
            logger.info(
                "Early stopping: all %s metric(s) exhausted patience=%s.",
                len(self.metric_specs),
                self.patience,
            )
            trainer.should_stop = True


class PhasedMultiMetricEarlyStop(pl.Callback):
    """Phase 1: recall only; when recall plateaus, Phase 2: ECE metrics with fresh patience."""

    def __init__(
        self,
        recall_spec: MetricSpec,
        ece_specs: List[MetricSpec],
        patience: int,
        ckpt_callbacks: Dict[str, FrozenModelCheckpoint],
    ):
        super().__init__()
        if not ece_specs:
            raise ValueError("PhasedMultiMetricEarlyStop requires at least one ECE metric.")
        self.recall_id, self.recall_monitor, _, _ = recall_spec
        self.ece_specs = ece_specs
        self.patience = patience
        self.ckpt_callbacks = ckpt_callbacks
        self.ece_phase_started = False
        self.recall_not_improved = 0
        self.recall_best = initial_best_for_metric("recall")
        self.recall_frozen = False
        self.ece_not_improved: Dict[str, int] = {m[0]: 0 for m in ece_specs}
        self.ece_best: Dict[str, float] = {
            m[0]: initial_best_for_metric(m[0]) for m in ece_specs
        }
        self.ece_frozen: Dict[str, bool] = {m[0]: False for m in ece_specs}

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if not self.ece_phase_started:
            current = trainer.callback_metrics.get(self.recall_monitor)
            if current is None:
                return
            current_f = float(current)
            if is_improvement("recall", current_f, self.recall_best):
                self.recall_best = current_f
                self.recall_not_improved = 0
            else:
                self.recall_not_improved += 1
                if self.recall_not_improved >= self.patience:
                    self.recall_frozen = True
                    recall_ckpt = self.ckpt_callbacks.get(self.recall_id)
                    if recall_ckpt is not None:
                        recall_ckpt.frozen = True
                    self.ece_phase_started = True
                    for metric_id, monitor, _, _ in self.ece_specs:
                        ece_cur = trainer.callback_metrics.get(monitor)
                        if ece_cur is not None:
                            self.ece_best[metric_id] = float(ece_cur)
                        self.ece_not_improved[metric_id] = 0
                        ece_ckpt = self.ckpt_callbacks.get(metric_id)
                        if ece_ckpt is not None:
                            ece_ckpt.frozen = False
                    logger.info(
                        "Phased early stop: recall locked after %s epochs without improvement "
                        "(patience=%s, best R@1=%.4f). Activating Phase 2 ECE metrics: %s.",
                        self.recall_not_improved,
                        self.patience,
                        self.recall_best,
                        [m[0] for m in self.ece_specs],
                    )
            return

        for metric_id, monitor, mode, _tag in self.ece_specs:
            if self.ece_frozen[metric_id]:
                continue
            current = trainer.callback_metrics.get(monitor)
            if current is None:
                continue
            current_f = float(current)
            if is_improvement(metric_id, current_f, self.ece_best[metric_id]):
                self.ece_best[metric_id] = current_f
                self.ece_not_improved[metric_id] = 0
            else:
                self.ece_not_improved[metric_id] += 1
                if self.ece_not_improved[metric_id] >= self.patience:
                    self.ece_frozen[metric_id] = True
                    ckpt_cb = self.ckpt_callbacks.get(metric_id)
                    if ckpt_cb is not None:
                        ckpt_cb.frozen = True
                    logger.info(
                        "Metric %r locked after %s epochs without improvement (patience=%s).",
                        metric_id,
                        self.ece_not_improved[metric_id],
                        self.patience,
                    )

        if all(self.ece_frozen.values()):
            logger.info(
                "Early stopping: all %s Phase-2 metric(s) exhausted patience=%s.",
                len(self.ece_specs),
                self.patience,
            )
            trainer.should_stop = True
