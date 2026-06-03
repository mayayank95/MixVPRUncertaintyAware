import logging
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler
# from torch.optim import lr_scheduler, optimizer

from configs.parser import build_config
from data.GSVCitiesDataloader import GSVCitiesDataModule
from losses.losses import get_loss, get_miner
from utils.runtime import init_model
from validation import get_validation_recalls

logger = logging.getLogger(__name__)


class VPRModel(pl.LightningModule):
    """Lightning wrapper around MixVPR encoder from get_model (same path as eval)."""

    def __init__(self, cfg, core: torch.nn.Module):
        super().__init__()
        self.freeze_all = bool(cfg.get("freeze_model"))
        self.freeze_batchnorm = bool(cfg.get("freeze_batchnorm"))
        if self.freeze_all:
            self.automatic_optimization = False

        self.encoder_arch = cfg.get("mixvpr_encoder_arch", "resnet50")
        self.lr = cfg["lr"]
        self.optimizer = cfg["mixvpr_optimizer"]
        self.weight_decay = cfg["mixvpr_weight_decay"]
        self.momentum = cfg["mixvpr_momentum"]
        self.warmpup_steps = cfg["mixvpr_warmup_steps"]
        self.milestones = list(cfg["mixvpr_milestones"])
        self.lr_mult = cfg["mixvpr_lr_mult"]

        loss_name = cfg["mixvpr_loss_name"]
        miner_name = cfg.get("mixvpr_miner_name") or ""
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = cfg["mixvpr_miner_margin"]
        self.faiss_gpu = bool(cfg.get("mixvpr_faiss_gpu"))

        self.save_hyperparameters(ignore=["core"])

        self.loss_fn = get_loss(loss_name)
        self.miner = get_miner(miner_name, self.miner_margin) if miner_name else None
        self.batch_acc = []

        # Same descriptor path as eval.py (resize + backbone + aggregator + L2Norm).
        self.core = core

    def on_train_epoch_start(self):
        self.train()
        if self.freeze_batchnorm:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
            logger.info("BatchNorm2d layers frozen (freeze_batchnorm)")

    def forward(self, x):
        descriptors, _ = self.core(x)
        return descriptors

    # configure the optimizer
    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            # Frozen sanity run: optimizer is unused (no grads); satisfy Lightning API.
            trainable = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]
        if self.optimizer.lower() == "sgd":
            optim = torch.optim.SGD(
                trainable,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif self.optimizer.lower() == "adamw":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == "adam":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(
                f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"'
            )
        scheduler = lr_scheduler.MultiStepLR(
            optim, milestones=self.milestones, gamma=self.lr_mult
        )
        return [optim], [scheduler]

    # configure the optizer step, takes into account the warmup stage
    # Lightning 2.x passes optimizer_closure as the 4th arg (no optimizer_idx / on_tpu kwargs).
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        del epoch, batch_idx
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)

    # The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

        else:  # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                # somes losses do the online mining inside (they don't need a miner objet),
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class,
                # and return a tuple containing the loss value and the batch_accuracy
                # (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log(
            "b_acc", sum(self.batch_acc) / len(self.batch_acc), prog_bar=True, logger=True
        )
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        bs, n, ch, h, w = places.shape

        # reshape places and labels
        images = places.view(bs * n, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images)  # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels)  # Call the loss_function we defined above

        self.log("loss", loss.item(), logger=True)
        if self.freeze_all:
            return None
        return {"loss": loss}

    # Lightning 2.x: replaces training_epoch_end (removed in PL2).
    def on_train_epoch_end(self):
        self.batch_acc = []

    # Lightning 2.x: validation_epoch_end removed; accumulate step outputs for on_validation_epoch_end.
    def on_validation_epoch_start(self):
        self._val_feats_by_dl = {}

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        # places, _ = batch
        # calculate descriptors
        descriptors = self(images)
        # descriptors = self(places)
        out = descriptors.detach().cpu()
        # Lightning 2.x: collect descriptors here (see on_validation_epoch_end).
        dl_idx = 0 if dataloader_idx is None else int(dataloader_idx)
        self._val_feats_by_dl.setdefault(dl_idx, []).append(out)
        return out

    # Lightning 2.x: replaces validation_epoch_end (removed in PL2).
    def on_validation_epoch_end(self):
        """Descriptors ordered refs then queries per val set (MSLS / Pitts)."""
        dm = self.trainer.datamodule
        val_step_outputs = [
            self._val_feats_by_dl.get(i, []) for i in range(len(dm.val_datasets))
        ]
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets) == 1:  # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]

        for i, (val_set_name, val_dataset) in enumerate(
            zip(dm.val_set_names, dm.val_datasets)
        ):
            feats = torch.concat(val_step_outputs[i], dim=0)

            if "pitts" in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                # num_queries = len(val_dataset) - num_references
                positives = val_dataset.getPositives()
            elif "msls" in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                # num_queries = len(val_dataset) - num_references
                positives = val_dataset.pIdx
            else:
                print(f"Please implement validation_epoch_end for {val_set_name}")
                raise NotImplementedError(f"validation_epoch_end for {val_set_name}")

            r_list = feats[:num_references]
            q_list = feats[num_references:]
            pitts_dict = get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
            )
            # del r_list, q_list, feats, num_references, positives

            self.log(f"{val_set_name}/R1", pitts_dict[1], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R5", pitts_dict[5], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R10", pitts_dict[10], prog_bar=False, logger=True)
        print("\n\n")


def _collect_recall_metrics(trainer: pl.Trainer) -> dict:
    out = {}
    for key, value in trainer.callback_metrics.items():
        name = str(key)
        if name.endswith("/R1") or name.endswith("/R5") or name.endswith("/R10"):
            out[name] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    return out


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

    # Lightning 2.x: pl.seed_everything (replaces pl.utilities.seed.seed_everything).
    seed = cfg["seed"] if cfg["seed"] != -1 else 190223
    pl.seed_everything(seed=seed, workers=True)

    _device, core = init_model(cfg)
    del _device

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

    model = VPRModel(cfg, core=core)

    callbacks = []
    if not cfg.get("no_checkpoint"):
        ckpt_monitor = cfg["mixvpr_ckpt_monitor"]
        callbacks.append(
            ModelCheckpoint(
                monitor=ckpt_monitor,
                filename=(
                    f"{model.encoder_arch}"
                    + f"_epoch({{epoch:02d}})_step({{step:04d}})_R1[{{{ckpt_monitor}:.4f}}]"
                ),
                auto_insert_metric_name=False,
                save_weights_only=True,
                save_top_k=cfg["mixvpr_ckpt_save_top_k"],
                mode="max",
            )
        )

    # ------------------
    # we instanciate a trainer
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
        # fast_dev_run=True  # uncomment for one-iteration smoke test
    )

    datamodule.setup("fit")
    recalls_before = None
    if cfg.get("validate_before_fit") or cfg.get("freeze_model"):
        print("\n>>> Validation BEFORE training\n")
        trainer.validate(model=model, datamodule=datamodule)
        recalls_before = _collect_recall_metrics(trainer)

    trainer.fit(model=model, datamodule=datamodule)

    if cfg.get("freeze_model") and recalls_before is not None:
        recalls_after = _collect_recall_metrics(trainer)
        _print_recall_comparison(recalls_before, recalls_after)
