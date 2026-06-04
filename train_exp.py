import logging
import multiprocessing
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

from utils.runtime import build_config_and_datasets, init_model
from data.test_dataset import TestDataset
from data.train_dataset import TrainDataset
from eval import eval_dataset
from eval_metrics.eval_wandb import panel_to_wandb_metrics
from losses import cosface_loss
from losses import uncertainty_utils
from utils import augmentations, commons, early_stop_utils, util, wandb_utils

logger = logging.getLogger(__name__)


def _load_and_freeze_classifiers(classifiers, checkpoint_path):
    """Load classifier weights from checkpoint and freeze them. Returns updated optimizers (None per classifier)."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or "classifiers_state_dict" not in checkpoint:
        logger.warning("No classifiers_state_dict found in checkpoint.")
        return None
    if len(checkpoint["classifiers_state_dict"]) != len(classifiers):
        logger.warning(f"Skipping classifiers load: checkpoint has {len(checkpoint['classifiers_state_dict'])}, config has {len(classifiers)}.")
        return None
    for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
        c.load_state_dict(sd)
    for c in classifiers:
        for p in c.parameters():
            p.requires_grad = False
    logger.info("Classifiers loaded and frozen.")
    return [None] * len(classifiers)


def train(args, model, device, dataset_name, datasets_dir):
    start_time = datetime.now()

    # ---- Losses & optimizer ----
    active_losses = args['losses']
    logger.info(f"Active losses: {active_losses}")

    if "ce" in active_losses:
        ce_criterion = torch.nn.CrossEntropyLoss()
    
    if "uncertainty" in active_losses and args['model_mode'] == "uncertainty":
        uncertainty_lambda = args.get('uncertainty_lambda', 1.0)
        logger.info(f"Using uncertainty loss: {args.get('uncertainty_loss', 'gaussian_nll')}")
        logger.info(
            "Variance head: "
            f"agg={bool(args.get('var_head_agg', False))}, "
            f"linear={args.get('var_head_linear', 'd')}, "
            f"activation={args.get('variance_activation') or 'softplus'}"
        )

    # LR for model: when freeze_model use head_lr (default 1e-3) so uncertainty head can learn; else use --lr
    model_lr = (args.get("head_lr") if args.get("head_lr") is not None else 1e-3) if args.get("freeze_model") else args["lr"]
    model_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=model_lr)
    opt_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.debug(f"Optimizer (model): lr={model_lr}, params ({len(opt_param_names)}): {opt_param_names}")
    if args.get("freeze_model"):
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"freeze_model: training {n_trainable} params (backbone/aggregation frozen)")

    train_set_folder = f"{datasets_dir[dataset_name]['train']}"
    val_set_folder = f"{datasets_dir[dataset_name]['validation']}"

    # ---- Datasets & classifiers ----
    groups = [TrainDataset(args, train_set_folder, M=args['M'], alpha=args['alpha'], N=args['N'], L=args['L'],
                    current_group=n, min_images_per_class=args['min_images_per_class']) for n in range(args['groups_num'])]

    # Each group has its own classifier, which depends on the number of classes in the group
    classifiers = [
        cosface_loss.MarginCosineProduct(
            args['descriptors_dimension'], len(group),
            s=args.get('s', 100.0), m=args.get('m', 0.4),
        )
        for group in groups
    ]
    classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args['classifiers_lr']) for classifier in classifiers]

    logger.debug(f"Using {len(groups)} groups")
    logger.debug(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
    logger.debug(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

    val_ds = TestDataset(f"{val_set_folder}/database", f"{val_set_folder}/queries", args['positive_dist_threshold'], args.get('image_size'), use_labels=True)
    logger.info(f"Validation set: {val_ds}")

    # GPU augmentations operate on batches (4D tensors) and must be applied in the
    # training loop after collation, unlike CPU augmentations which run per-image
    # inside TrainDataset.__getitem__.
    if args['augmentation_device'] == "cuda":
        gpu_augmentation = T.Compose([
                augmentations.DeviceAgnosticColorJitter(brightness=args['brightness'],
                                                        contrast=args['contrast'],
                                                        saturation=args['saturation'],
                                                        hue=args['hue']),
                augmentations.DeviceAgnosticRandomResizedCrop([args['image_size'], args['image_size']],
                                                            scale=[1-args['random_resized_crop'], 1]),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    early_stop_metrics = args["early_stop_metrics"]
    early_stop_best_values: dict = {m: early_stop_utils.initial_best_for_metric(m) for m in early_stop_metrics}
    early_stop_best_epochs: dict = {m: None for m in early_stop_metrics}
    not_improved_counts: dict = {m: 0 for m in early_stop_metrics}

    # Phased early stopping: Phase 1 = recall only, Phase 2 = ECE metrics only.
    phased = args.get("phased_early_stop", False) and "recall" in early_stop_metrics
    if phased:
        phase2_metrics = [m for m in early_stop_metrics if m.startswith("ece_")]
        active_phase_metrics = ["recall"]
        ece_phase_started = False
        logger.info(f"Phased early stop enabled: Phase 1 = recall, Phase 2 = {phase2_metrics}")
    else:
        active_phase_metrics = list(early_stop_metrics)
        ece_phase_started = True  # treat as already in final phase (no transition needed)

    # ---- Resume from checkpoint ----
    if args.get('resume_train') is not None:
        (
            model, model_optimizer, classifiers, classifiers_optimizers,
            best_val_recall1, start_epoch_num,
            ckpt_es_bests, ckpt_es_epochs, ckpt_patience_counts,
        ) = util.resume_train(device, args, args['log_dir'], model, model_optimizer, classifiers, classifiers_optimizers)
        epoch_num = start_epoch_num - 1
        for m in early_stop_metrics:
            if m in ckpt_es_bests:
                early_stop_best_values[m] = float(ckpt_es_bests[m])
            if m in ckpt_es_epochs and ckpt_es_epochs[m] is not None:
                early_stop_best_epochs[m] = int(ckpt_es_epochs[m])
            if m in ckpt_patience_counts:
                not_improved_counts[m] = int(ckpt_patience_counts[m])
        logger.info(
            f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.2f} from checkpoint {args['resume_train']}; "
            f"early_stop metrics: {early_stop_metrics}, patience counters: {not_improved_counts}"
        )
    elif args.get('load_classifiers'):
        best_val_recall1 = start_epoch_num = 0
        logger.info(f"Loading classifier weights from {args['load_classifiers']}")
        frozen_optims = _load_and_freeze_classifiers(classifiers, args['load_classifiers'])
        if frozen_optims is not None:
            classifiers_optimizers = frozen_optims
    else:
        best_val_recall1 = start_epoch_num = 0

    is_resuming = args.get('resume_train') or args.get('resume_model') or args.get('method') == 'cosplace_pretrained'
    if is_resuming and args.get("debug"):
        logger.info(f"Verifying resumed/pretrained model performance on '{val_set_folder}' folder (before any training)...")
        init = eval_dataset(
            args, model, device, dataset_name, val_set_folder, wandb_step=0, log_dataset_info=False,
            use_descriptor_cache=False,
        )
        _vs = init.panel_data.get("variance_stat", {})
        _mv = f"{_vs['q_mean']:.4f}" if _vs.get("q_mean") is not None else "N/A"
        _xv = f"{_vs['q_max']:.4f}" if _vs.get("q_max") is not None else "N/A"
        _map = f", mAP@1={init.map_at_k[0]:.2f}" if init.map_at_k is not None else ""
        init_wandb_metrics = (
            panel_to_wandb_metrics(init.panel_data, wandb_step=0)
            if init.panel_data else {}
        )
        extras = []
        for m in early_stop_metrics:
            v = early_stop_utils.get_metric_value(
                m,
                recalls=init.recalls,
                recall_values=args["recall_values"],
                eval_wandb_metrics=init_wandb_metrics,
                panel_data=init.panel_data,
                dataset_name=dataset_name,
            )
            if v is not None:
                early_stop_best_values[m] = float(v)
                extras.append(f"{m}={v:.4f}" if m != "recall" else f"{m}={v:.2f}")
        ece_part = f", init_early_stop=[{', '.join(extras)}]" if extras else ""
        if args.get("var_init"):
            logger.info(f"Initial val (after var_init, before training): R@1={init.recalls[0]:.2f}{_map}, mean_var={_mv}, max_var={_xv}{ece_part}")
        else:
            logger.info(f"Initial val (before training): R@1={init.recalls[0]:.2f}{_map}, mean_var={_mv}, max_var={_xv}{ece_part}")
    elif args.get('resume_train') or args.get('resume_model'):
        logger.debug("Skipping pre-training resume verification (enable --debug to run it).")

    # ---- Training loop ----
    logger.info("Start training ...")
    logger.info(f"There are {len(groups[0])} classes for the first group, " +
                f"each epoch has {args['iterations_per_epoch']} iterations " +
                f"with batch_size {args['batch_size']}, therefore the model sees each class (on average) " +
                f"{args['iterations_per_epoch'] * args['batch_size'] / len(groups[0]):.1f} times per epoch")

    early_stop_enabled = not args.get("disable_early_stop", False)
    patience = args.get("patience", 5)
    if early_stop_enabled and patience is None:
        patience = 5
    if early_stop_enabled:
        logger.info(f"Early stopping enabled with patience={patience}.")
    else:
        logger.info("Early stopping disabled: training will run until epochs_num is reached.")

    mean_variances_history = []
    best_model_epoch = None  # recall best epoch (legacy summary); per-metric in early_stop_best_epochs
    best_variance_mean = None
    best_variance_std = None
    best_variance_min = None
    best_variance_max = None
    for epoch_num in range(start_epoch_num, args['epochs_num']):
        
        # ---- Train ----
        epoch_start_time = datetime.now()
        current_group_num = epoch_num % args['groups_num']
        classifiers[current_group_num] = classifiers[current_group_num].to(device)
        if classifiers_optimizers[current_group_num] is not None:
            util.move_to_device(classifiers_optimizers[current_group_num], device)

        dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args['num_workers'],
                                                batch_size=args['batch_size'], shuffle=True,
                                                pin_memory=(device == "cuda"), drop_last=True)

        dataloader_iterator = iter(dataloader)
        model = model.train()
        if args.get("freeze_batchnorm"):
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()  # Freeze BN statistics
            logger.info("BatchNorm2d layers frozen (flag enabled)")
        epoch_losses = []
        epoch_losses_ce = []
        epoch_losses_gnll = []
        epoch_variances = []
        
        for iteration in tqdm(range(args['iterations_per_epoch']), ncols=100):
            images, targets, _ = next(dataloader_iterator)
            images, targets = images.to(device), targets.to(device)
            
            if args['augmentation_device'] == "cuda":
                images = gpu_augmentation(images)
            
            model_optimizer.zero_grad()
            if classifiers_optimizers[current_group_num] is not None:
                classifiers_optimizers[current_group_num].zero_grad()
            
            loss = torch.tensor(0.0, device=device)

            mu_norm, variance = model(images)
            if "ce" in active_losses:
                output = classifiers[current_group_num](mu_norm, targets)
                loss_ce = ce_criterion(output, targets)
                loss = loss + loss_ce
                epoch_losses_ce.append(loss_ce.item())

            # Uncertainty calculations (outside autocast for stability)
            if "uncertainty" in active_losses and args['model_mode'] == "uncertainty":
                # Normalize class weights to get the ground-truth "prototype" for each class
                weights = classifiers[current_group_num].weight
                norm_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
                target_vectors = norm_weights[targets]
                loss_uncertainty = uncertainty_utils.compute_uncertainty_loss(
                    mu_norm, target_vectors, variance,
                    loss_type=args.get('uncertainty_loss', 'gaussian_nll'),
                    lambda_=uncertainty_lambda,
                    gnll_mu_scale_mode=args.get("gnll_mu_scale_mode", "sqrt_dim"),
                    gnll_mu_scale_value=args.get("gnll_mu_scale_value", 1.0),
                )
                loss = loss + loss_uncertainty
                epoch_losses_gnll.append(loss_uncertainty.item())
            
            if args['model_mode'] == "uncertainty":
                epoch_variances.append(variance.mean().item())
                
            # ---- Backward ----
            loss.backward()
            if args.get("debug") and args.get("model_mode") == "uncertainty" and iteration == 0:
                root = getattr(model, "module", model)
                if hasattr(root, "var_head"):
                    norms = []
                    for n, p in root.var_head.named_parameters():
                        g = p.grad.norm().item() if p.grad is not None else 0.0
                        norms.append(f"{n}={g:.6f}")
                    logger.debug(f"[debug_var_head_grad] epoch {epoch_num} first batch var_head grad norms: {norms}")
            model_optimizer.step()
            if classifiers_optimizers[current_group_num] is not None:
                classifiers_optimizers[current_group_num].step()
            
            epoch_losses.append(loss.item())
            del loss
            if "ce" in active_losses:
                del output
            del images

            if args['dry_run']:
                logger.info("Dry run: breaking epoch loop after one iteration")
                break
        
        classifiers[current_group_num] = classifiers[current_group_num].cpu()
        if classifiers_optimizers[current_group_num] is not None:
            util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
        
        if args['model_mode'] == "uncertainty":
            mean_total_loss = np.mean(epoch_losses)
            mean_variance = np.mean(epoch_variances)
            mean_variances_history.append(mean_variance)
            
            log_msg = f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, loss_total = {mean_total_loss:.4f}"
            if "ce" in active_losses:
                log_msg += f", loss_ce = {np.mean(epoch_losses_ce):.4f}"
            if "uncertainty" in active_losses:
                log_msg += f", loss_uncertainty = {np.mean(epoch_losses_gnll):.4f}"
                log_msg += f", mean_variance = {mean_variance:.4f}"
            logger.info(log_msg)
        else:
            logger.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                        f"loss = {np.mean(epoch_losses):.4f}")
        
        # ---- Evaluate ----
        val = eval_dataset(
            args, model, device, dataset_name, val_set_folder, wandb_step=epoch_num, log_dataset_info=False,
            use_descriptor_cache=False,
        )
        # Track best recall@1 unconditionally for checkpoint metadata / W&B display.
        best_val_recall1 = max(val.recalls[0], best_val_recall1)

        val_wandb_metrics = (
            panel_to_wandb_metrics(val.panel_data, wandb_step=epoch_num)
            if val.panel_data else {}
        )
        improved: dict = {}
        rv = args["recall_values"]
        for m in early_stop_metrics:
            # Skip metrics not in the active phase
            if m not in active_phase_metrics:
                improved[m] = False
                continue
            if early_stop_enabled and not_improved_counts[m] >= patience:
                improved[m] = False
                continue
            cur = early_stop_utils.get_metric_value(
                m,
                recalls=val.recalls,
                recall_values=rv,
                eval_wandb_metrics=val_wandb_metrics,
                panel_data=val.panel_data,
                dataset_name=dataset_name,
            )
            if cur is None:
                improved[m] = False
                continue
            prev = early_stop_best_values[m]
            if early_stop_utils.is_improvement(m, cur, prev):
                improved[m] = True
                early_stop_best_values[m] = max(cur, prev) if m == "recall" else min(cur, prev)
                early_stop_best_epochs[m] = epoch_num
                not_improved_counts[m] = 0
            else:
                improved[m] = False
                not_improved_counts[m] += 1
                if early_stop_enabled and not_improved_counts[m] >= patience:
                    logger.info(f"Metric '{m}' exhausted patience={patience} at epoch {epoch_num}. Locked.")

        any_improved = any(improved.values())

        val_vs = val.panel_data.get("variance_stat", {})
        wandb_utils.log_train_epoch(
            args, epoch_num, val.recalls, val.map_at_k, best_val_recall1, active_losses,
            epoch_variances, epoch_losses, epoch_losses_ce, epoch_losses_gnll,
            val_vs.get("q_mean"), val_vs.get("q_std"), val_vs.get("q_min"), val_vs.get("q_max"),
            val_wandb_metrics, val.wandb_images,
        )

        if any_improved:
            best_model_epoch = epoch_num
            if epoch_variances:
                best_variance_mean = float(np.mean(epoch_variances))
                best_variance_std = float(np.std(epoch_variances))
                best_variance_min = float(np.min(epoch_variances))
                best_variance_max = float(np.max(epoch_variances))

        # Save checkpoint + per-metric best models (before early-stop break so this epoch is persisted).
        if not args['dry_run']:
            util.save_checkpoint({
                "epoch_num": epoch_num + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model_optimizer.state_dict(),
                "classifiers_state_dict": [c.state_dict() for c in classifiers],
                "optimizers_state_dict": [c.state_dict() if c is not None else {} for c in classifiers_optimizers],
                "best_val_recall1": best_val_recall1,
                "best_model_epoch": early_stop_best_epochs.get("recall"),
                "early_stop_metrics": early_stop_metrics,
                "early_stop_best_values": dict(early_stop_best_values),
                "early_stop_best_epochs": {k: v for k, v in early_stop_best_epochs.items() if v is not None},
                "not_improved_counts": dict(not_improved_counts),
            }, args['log_dir'], best_for_metrics=improved if any_improved else None)
            saved = [m for m, ok in improved.items() if ok]
            if saved:
                files = [early_stop_utils.best_model_filename(m, early_stop_metrics) for m in saved]
                logger.info(f"Saved best checkpoint(s): {list(zip(saved, files))}")

        # ---- Phased early stop: transition from recall → ECE ----
        if early_stop_enabled and phased and not ece_phase_started and not_improved_counts.get("recall", 0) >= patience:
            ece_phase_started = True
            active_phase_metrics = phase2_metrics
            # Seed ECE best values from current epoch and reset patience counters.
            for m in phase2_metrics:
                cur = early_stop_utils.get_metric_value(
                    m, recalls=val.recalls, recall_values=rv,
                    eval_wandb_metrics=val_wandb_metrics,
                    panel_data=val.panel_data,
                    dataset_name=dataset_name,
                )
                if cur is not None:
                    early_stop_best_values[m] = float(cur)
                not_improved_counts[m] = 0
            logger.info(
                f"Phased early stop: recall plateaued at epoch {epoch_num} "
                f"(best R@1={early_stop_best_values.get('recall', 0):.2f} at epoch {early_stop_best_epochs.get('recall')}). "
                f"Activating Phase 2 ECE metrics: {phase2_metrics} with fresh patience={patience}."
            )

        # Early stop: all *active* metrics must have exhausted their patience.
        if early_stop_enabled:
            exhausted = [m for m in active_phase_metrics if not_improved_counts[m] >= patience]
            if ece_phase_started and len(exhausted) == len(active_phase_metrics):
                logger.info(
                    f"Early stopping: all {len(active_phase_metrics)} active metric(s) exhausted patience={patience}. "
                    f"Last improvement epochs: {early_stop_best_epochs}"
                )
                break

        if args['dry_run']:
            break

    if mean_variances_history:
        logger.info(f"Mean variance evolution: start={mean_variances_history[0]:.4f}, end={mean_variances_history[-1]:.4f}")

    # Populate W&B Summary panel with best-model metadata (not just time-series history).
    if wandb_utils and args.get("use_wandb"):
        try:
            import wandb as _wandb
            if _wandb.run is not None:
                _wandb.run.summary["best_model_epoch"] = best_model_epoch
                _wandb.run.summary["best_val_recall_01"] = float(best_val_recall1)
                if best_variance_mean is not None:
                    _wandb.run.summary["best_variance_mean"] = float(best_variance_mean)
                    _wandb.run.summary["best_variance_std"] = float(best_variance_std)
                    _wandb.run.summary["best_variance_min"] = float(best_variance_min)
                    _wandb.run.summary["best_variance_max"] = float(best_variance_max)
                _wandb.run.summary["early_stop_metrics"] = list(early_stop_metrics)
                for m in early_stop_metrics:
                    bv = early_stop_best_values.get(m)
                    be = early_stop_best_epochs.get(m)
                    sm = m.replace("/", "_")
                    if bv is not None and bv != float("inf"):
                        _wandb.run.summary[f"best_value_{sm}"] = float(bv)
                    if be is not None:
                        _wandb.run.summary[f"best_epoch_{sm}"] = int(be)
        except Exception:
            # Summary is best-effort; do not fail training due to logging.
            pass

    logger.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")
    if best_model_epoch is not None:
        parts = [
            f"best_model_epoch={best_model_epoch}",
            f"best_val_recall_01={best_val_recall1:.4f}",
            f"early_stop_metrics={early_stop_metrics}",
        ]
        for m in early_stop_metrics:
            bv = early_stop_best_values.get(m)
            be = early_stop_best_epochs.get(m)
            if m == "recall":
                parts.append(f"best_epoch_{m}={be if be is not None else 'N/A'}")
            else:
                bv_str = f"{bv:.4f}" if bv is not None and bv != float("inf") else "N/A"
                parts.append(f"best_{m}={bv_str}")
                parts.append(f"best_epoch_{m}={be if be is not None else 'N/A'}")
        logger.info("Best model summary: " + ", ".join(parts))
    else:
        logger.info("Best model summary: no best_model_epoch set (no improvement vs initial metric).")
    logger.info("Experiment finished (without any errors)")

if __name__ == "__main__":
    # ---- Load config and datasets (shared helper) ----
    cfg, entries, datasets_dir = build_config_and_datasets()
    wandb_utils.init_wandb(cfg, job_type="train")

    # Training-specific setup
    commons.make_deterministic(cfg["seed"])
    # Handle the cuDNN Benchmark speed/reproducibility trade-off
    commons.setup_cudnn(cfg["cudnn_benchmark"])
    device, model = init_model(cfg)

    logger.debug(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")
    if torch.cuda.is_available():
        logger.debug(f"GPU type: {torch.cuda.get_device_name(0)}")

    # Optionally copy the resume model into the current log directory
    commons.copy_resume_model_to_log_dir(cfg, logger)

    for e in entries:
        logger.info(f"Training dataset: {e['name']}")
        train(cfg, model, device, e["name"], datasets_dir)
    logger.info("Training completed.")

    wandb_utils.finish_train_run(cfg)