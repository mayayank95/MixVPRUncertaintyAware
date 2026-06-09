import logging
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.optim import lr_scheduler

import numpy as np

from configs.parser import build_config
from data.GSVCitiesDataloader import GSVCitiesDataModule
from losses.losses import get_loss, get_miner
from losses.vmf_place_loss import place_centroid_targets
from models.model_mode import _encode_inputs, build_model_mode
from utils.runtime import init_model
from utils.mixvpr_train_wandb import attach_wandb_callback
# from utils.mixvpr_val_ece import attach_val_ece_callback  # old: second FAISS + ECE callback
from utils.mixvpr_eval import run_mixvpr_lightning_val_eval, run_mixvpr_validation_eval
from utils import wandb_utils
from utils.early_stop_utils import resolve_lightning_ckpt_monitor
# from validation import get_validation_recalls  # old: Lightning-only recall (PrettyTable)

logger = logging.getLogger(__name__)


class VPRModel(pl.LightningModule):
    """Lightning wrapper around MixVPR encoder from get_model (same path as eval)."""

    def __init__(self, cfg, core: torch.nn.Module):
        super().__init__()
        self.freeze_base = bool(cfg.get("freeze_model"))
        self.freeze_batchnorm = bool(cfg.get("freeze_batchnorm"))

        self.encoder_arch = cfg.get("mixvpr_encoder_arch", "resnet50")
        self.lr = cfg["lr"]
        self.head_lr = float(cfg.get("head_lr", 1e-3))
        self.optimizer = cfg["mixvpr_optimizer"]
        self.weight_decay = cfg["mixvpr_weight_decay"]
        self.momentum = cfg["mixvpr_momentum"]
        self.warmpup_steps = cfg["mixvpr_warmup_steps"]
        self.milestones = list(cfg["mixvpr_milestones"])
        self.lr_mult = cfg["mixvpr_lr_mult"]
        self.faiss_gpu = bool(cfg.get("mixvpr_faiss_gpu"))

        loss_name = cfg["mixvpr_loss_name"]
        uncertainty_loss = cfg["uncertainty_loss"]
        active_losses = cfg["losses"]
        uncertainty_lambda = cfg["uncertainty_lambda"]

        miner_name = cfg.get("mixvpr_miner_name") or ""
        if "basic" not in active_losses:
            miner_name = ""
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = cfg["mixvpr_miner_margin"]

        self.save_hyperparameters(ignore=["core"])

        self.uncertainty_lambda = float(uncertainty_lambda)
        self.descriptors_dimension = int(cfg["descriptors_dimension"])
        self.loss_basic, self.loss_uncertainty = get_loss(
            loss_name, active_losses, uncertainty_loss, self.descriptors_dimension
        )
        self.miner = get_miner(miner_name, self.miner_margin) if miner_name else None
        self.batch_acc = []
        self._epoch_variances = []

        for name, child in core.named_children():
            self.add_module(name, child)
        self._var_from_feature_map = core._var_from_feature_map
        self._uses_mixvpr = core._uses_mixvpr
        self._eval_cfg = cfg
        self._wandb_cb = None

    def _set_train_mode(self) -> None:
        self.train()
        if self.freeze_batchnorm:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

    def on_fit_start(self) -> None:
        self._set_train_mode()

    def on_train_epoch_start(self):
        self._set_train_mode()
        if self.freeze_batchnorm:
            logger.info("BatchNorm2d layers frozen (freeze_batchnorm)")

    def forward(self, x):
        feat, desc = _encode_inputs(self, x)
        mu = self.final_l2(desc)
        if self._var_from_feature_map:
            variance = self.var_head(feat)
        else:
            variance = self.var_head(desc)
        return mu, variance + 1e-6

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                "No trainable parameters. With --freeze_model, use model_mode=uncertainty "
                "so var_head (and final_l2) remain trainable."
            )
        opt_lr = self.head_lr if self.freeze_base else self.lr
        if self.optimizer.lower() == "sgd":
            optim = torch.optim.SGD(
                trainable,
                lr=opt_lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif self.optimizer.lower() in ("adamw", "adam"):
            optim = torch.optim.AdamW(
                trainable, lr=opt_lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(
                f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"'
            )
        scheduler = lr_scheduler.MultiStepLR(
            optim, milestones=self.milestones, gamma=self.lr_mult
        )
        return [optim], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        del epoch, batch_idx
        base_lr = self.head_lr if self.freeze_base else self.lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * base_lr
        optimizer.step(closure=optimizer_closure)

    def _uncertainty_loss(self, descriptors, labels, variances):
        targets = place_centroid_targets(descriptors, labels)
        kappa = variances.mean(dim=-1, keepdim=True)
        return self.uncertainty_lambda * self.loss_uncertainty(descriptors, kappa, targets)

    def loss_function(self, descriptors, labels, variances=None):
        loss = torch.tensor(0.0, device=descriptors.device, dtype=descriptors.dtype)
        batch_acc = 0.0

        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            if self.loss_basic is not None:
                loss = loss + self.loss_basic(descriptors, labels, miner_outputs)
            if self.loss_uncertainty is not None and variances is not None:
                loss = loss + self._uncertainty_loss(descriptors, labels, variances)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            if self.loss_basic is not None:
                basic_out = self.loss_basic(descriptors, labels)
                if isinstance(basic_out, tuple):
                    loss, batch_acc = basic_out
                else:
                    loss = loss + basic_out
            if self.loss_uncertainty is not None and variances is not None:
                loss = loss + self._uncertainty_loss(descriptors, labels, variances)

        if not self.freeze_base and (self.miner is not None or self.loss_basic is not None):
            self.batch_acc.append(batch_acc)
        return loss

    def training_step(self, batch, batch_idx):
        places, labels = batch
        bs, n, ch, h, w = places.shape
        images = places.view(bs * n, ch, h, w)
        labels = labels.view(-1)

        descriptors, variances = self(images)
        loss = self.loss_function(descriptors, labels, variances)

        if variances is not None:
            per_sample = variances.detach().float().mean(dim=-1).cpu()
            self._epoch_variances.extend(per_sample.tolist())

        if self._wandb_cb is not None:
            self._wandb_cb.record_train_batch(self, loss, variances, descriptors, labels)

        self.log("loss", loss.item(), logger=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        if self.freeze_base and self._epoch_variances:
            median = float(np.median(self._epoch_variances))
            self.log(
                "variances_median",
                median,
                prog_bar=True,
                logger=True,
                on_epoch=True,
            )
        elif not self.freeze_base and self.batch_acc:
            self.log(
                "b_acc",
                sum(self.batch_acc) / len(self.batch_acc),
                prog_bar=True,
                logger=True,
                on_epoch=True,
            )
        self._epoch_variances = []
        self.batch_acc = []

    def on_validation_epoch_start(self):
        self._val_feats_by_dl = {}
        self._val_vars_by_dl = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        descriptors, variances = self(images)
        out = descriptors.detach().cpu()
        dl_idx = 0 if dataloader_idx is None else int(dataloader_idx)
        self._val_feats_by_dl.setdefault(dl_idx, []).append(out)
        self._val_vars_by_dl.setdefault(dl_idx, []).append(variances.detach().cpu())
        return out

    def on_validation_epoch_end(self):
        """Same metrics as eval.py: one FAISS pass + PrettyTable recall on stdout."""
        run_mixvpr_lightning_val_eval(self, self._eval_cfg)
        if self._wandb_cb is not None:
            self._wandb_cb.log_val_ece_wandb(self.trainer, self)

        # --- old Lightning validation (get_validation_recalls + MixVPRValEceCallback) ---
        # dm = self.trainer.datamodule
        # val_step_outputs = [
        #     self._val_feats_by_dl.get(i, []) for i in range(len(dm.val_datasets))
        # ]
        # if len(dm.val_datasets) == 1:
        #     val_step_outputs = [val_step_outputs]
        # for i, (val_set_name, val_dataset) in enumerate(
        #     zip(dm.val_set_names, dm.val_datasets)
        # ):
        #     feats = torch.concat(val_step_outputs[i], dim=0)
        #     if "pitts" in val_set_name:
        #         num_references = val_dataset.dbStruct.numDb
        #         positives = val_dataset.getPositives()
        #     elif "msls" in val_set_name:
        #         num_references = val_dataset.num_references
        #         positives = val_dataset.pIdx
        #     else:
        #         raise NotImplementedError(f"validation_epoch_end for {val_set_name}")
        #     r_list = feats[:num_references]
        #     q_list = feats[num_references:]
        #     pitts_dict = get_validation_recalls(
        #         r_list=r_list,
        #         q_list=q_list,
        #         k_values=[1, 5, 10, 15, 20, 50, 100],
        #         gt=positives,
        #         print_results=True,
        #         dataset_name=val_set_name,
        #         faiss_gpu=self.faiss_gpu,
        #     )
        #     self.log(f"{val_set_name}/R1", pitts_dict[1], prog_bar=False, logger=True)
        #     self.log(f"{val_set_name}/R5", pitts_dict[5], prog_bar=False, logger=True)
        #     self.log(f"{val_set_name}/R10", pitts_dict[10], prog_bar=False, logger=True)
        # print("\n\n")


# def _collect_recall_metrics(trainer: pl.Trainer) -> dict:
#     """Used with trainer.validate; pre/post train now use run_mixvpr_validation_eval."""
#     out = {}
#     for key, value in trainer.callback_metrics.items():
#         name = str(key)
#         if name.endswith("/R1") or name.endswith("/R5") or name.endswith("/R10"):
#             out[name] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
#     return out


def _eval_model_from_state(pl_module: VPRModel) -> torch.nn.Module:
    """Rebuild eval ``Uncertainty`` from Lightning ``state_dict`` (same flat keys)."""
    eval_model = build_model_mode(pl_module._eval_cfg)
    eval_model.load_state_dict(pl_module.state_dict(), strict=False)
    return eval_model


def _print_recall_comparison(before: dict, after: dict) -> None:
    print("\n" + "=" * 60)
    print("Frozen sanity check: validation recall comparison")
    print("=" * 60)
    all_keys = sorted(set(before) | set(after))
    for key in all_keys:
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None:
            print(f"  {key}: before={b} after={a}")
            continue
        delta = a - b
        mark = "OK" if abs(delta) < 1e-5 else "DIFF"
        print(f"  [{mark}] {key}: before={b:.4f} after={a:.4f} delta={delta:+.6f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    cfg, _entries = build_config()
    logger.info(" ".join(sys.argv))
    if cfg.get("log_dir"):
        logger.info("The outputs are being saved in %s", cfg["log_dir"])

    seed = cfg["seed"] if cfg["seed"] != -1 else 190223
    pl.seed_everything(seed=seed, workers=True)

    device, core = init_model(cfg)

    img_side = int(cfg["image_size"])
    datamodule = GSVCitiesDataModule(
        batch_size=cfg["batch_size"],
        img_per_place=cfg["img_per_place"],
        min_img_per_place=cfg["min_img_per_place"],
        shuffle_all=bool(cfg["mixvpr_shuffle_all"]),
        random_sample_from_each_place=bool(cfg["mixvpr_random_sample_from_each_place"]),
        image_size=(img_side, img_side),
        num_workers=cfg["num_workers"],
        show_data_stats=True,
        val_set_names=list(cfg["mixvpr_val_sets"]),
        datasets_config=cfg["config"],
        positive_dist_threshold=cfg["positive_dist_threshold"],
    )

    callbacks = [
        TQDMProgressBar(refresh_rate=max(1, int(cfg["log_every_n_steps"]))),
    ]
    wandb_cb = attach_wandb_callback(cfg, callbacks)

    model = VPRModel(cfg, core=core)
    if wandb_cb is not None:
        model._wandb_cb = wandb_cb
    # attach_val_ece_callback(cfg, callbacks)  # old: duplicate FAISS for ECE only

    if not cfg.get("no_checkpoint"):
        ckpt_monitor, ckpt_mode, ckpt_tag = resolve_lightning_ckpt_monitor(cfg)
        logger.info(
            "ModelCheckpoint monitor=%r mode=%s (from early_stop_metrics=%s)",
            ckpt_monitor,
            ckpt_mode,
            cfg.get("early_stop_metrics"),
        )
        callbacks.append(
            ModelCheckpoint(
                monitor=ckpt_monitor,
                filename=(
                    f"{model.encoder_arch}"
                    + f"_epoch({{epoch:02d}})_step({{step:04d}})_"
                    + f"{ckpt_tag}[{{{ckpt_monitor}:.4f}}]"
                ),
                auto_insert_metric_name=False,
                save_weights_only=True,
                save_top_k=cfg["mixvpr_ckpt_save_top_k"],
                mode=ckpt_mode,
            )
        )

    use_gpu = cfg["device"] == "cuda"
    default_root_dir = f"./LOGS/{model.encoder_arch}"
    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1 if use_gpu else "auto",
        default_root_dir=default_root_dir,
        num_sanity_val_steps=cfg["num_sanity_val_steps"],
        precision=cfg["precision"],
        max_epochs=cfg["epochs_num"],
        check_val_every_n_epoch=cfg["check_val_every_n_epoch"],
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1 if cfg["reload_dataloaders"] else 0,
        log_every_n_steps=cfg["log_every_n_steps"],
    )

    if cfg.get("validate_before_fit"):
        print("\n>>> Validation BEFORE training\n")
        # trainer.validate(model=model, datamodule=datamodule)
        # recalls_before = _collect_recall_metrics(trainer)
        recalls_before = run_mixvpr_validation_eval(
            cfg, core, device, wandb_step=0, use_descriptor_cache=False
        )

    model._set_train_mode()
    trainer.fit(model=model, datamodule=datamodule)

    if cfg.get("validate_before_fit") and recalls_before is not None:
        print("\n>>> Validation AFTER training\n")
        # Lightning may leave weights on CPU after fit; eval_dataset moves model back to device.
        recalls_after = run_mixvpr_validation_eval(
            cfg,
            _eval_model_from_state(model),
            device,
            wandb_step=cfg.get("epochs_num", 0),
            use_descriptor_cache=False,
        )
        _print_recall_comparison(recalls_before, recalls_after)

    wandb_utils.finish_train_run(cfg)
