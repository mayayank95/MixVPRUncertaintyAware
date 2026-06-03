import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Type

import torch
from losses.cosface_loss import MarginCosineProduct

from utils.early_stop_utils import best_model_filename





def move_to_device(optimizer: Type[torch.optim.Optimizer], device: str):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_checkpoint(
    state: dict,
    output_folder: str,
    ckpt_filename: str = "last_checkpoint.pth",
    best_for_metrics: Optional[Dict[str, bool]] = None,
):
    """Save training checkpoint. Optionally write per-metric best_model_*.pth when flags are True."""
    # TODO it would be better to move weights to cpu before saving
    checkpoint_path = f"{output_folder}/{ckpt_filename}"
    torch.save(state, checkpoint_path)
    if not best_for_metrics:
        return
    metrics_order = state.get("early_stop_metrics") or ["recall"]
    epochs = state.get("early_stop_best_epochs") or {}
    base_payload = {
        "model_state_dict": state["model_state_dict"],
        "classifiers_state_dict": state["classifiers_state_dict"],
    }
    for metric, did_improve in best_for_metrics.items():
        if not did_improve:
            continue
        fname = best_model_filename(metric, metrics_order)
        payload = dict(base_payload)
        ep = epochs.get(metric)
        if ep is not None:
            payload["best_model_epoch"] = int(ep)
        payload["early_stop_metric"] = metric
        torch.save(payload, f"{output_folder}/{fname}")


def resume_train(device: str, args: Dict, output_folder: str, model: torch.nn.Module,
                 model_optimizer: Type[torch.optim.Optimizer], classifiers: List[MarginCosineProduct],
                 classifiers_optimizers: List[Type[torch.optim.Optimizer]]):
    """Load full training state: model, optimizer, classifiers, and epoch counter."""
    logging.info(f"Loading checkpoint: {args['resume_train']}")
    checkpoint = torch.load(args['resume_train'], map_location='cpu', weights_only=False)

    start_epoch_num = checkpoint["epoch_num"]
    
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    
    model = model.to(device)
    model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    assert args['groups_num'] == len(classifiers) == len(classifiers_optimizers) == \
        len(checkpoint["classifiers_state_dict"]) == len(checkpoint["optimizers_state_dict"]), \
        (f"{args['groups_num']}, {len(classifiers)}, {len(classifiers_optimizers)}, "
         f"{len(checkpoint['classifiers_state_dict'])}, {len(checkpoint['optimizers_state_dict'])}")
    
    for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
        c = c.to(device)
        c.load_state_dict(sd)
    for c, sd in zip(classifiers_optimizers, checkpoint["optimizers_state_dict"]):
        if len(sd.get("state", {})) > 0:
            c.load_state_dict(sd)
    for c in classifiers:
        c = c.cpu()
    
    best_val_recall1 = checkpoint["best_val_recall1"]

    early_stop_best_values = checkpoint.get("early_stop_best_values")
    if early_stop_best_values is None:
        early_stop_best_values = {"recall": float(best_val_recall1)}
    early_stop_best_epochs = checkpoint.get("early_stop_best_epochs")
    if early_stop_best_epochs is None:
        bm = checkpoint.get("best_model_epoch")
        early_stop_best_epochs = {"recall": int(bm)} if bm is not None else {}
    not_improved_counts = checkpoint.get("not_improved_counts") or {}

    # Copy only best_model files that match the new run's early_stop_metrics.
    new_metrics = args.get("early_stop_metrics") or ["recall"]
    expected_files = {best_model_filename(m, new_metrics) for m in new_metrics}
    prev_dir = Path(args["resume_train"]).parent
    copied = []
    for fname in sorted(expected_files):
        src = prev_dir / fname
        if src.exists():
            shutil.copy(src, output_folder)
            copied.append(fname)
    if copied:
        logging.info("Copied best model file(s) from previous run: %s -> %s", copied, output_folder)

    return (
        model,
        model_optimizer,
        classifiers,
        classifiers_optimizers,
        best_val_recall1,
        start_epoch_num,
        early_stop_best_values,
        early_stop_best_epochs,
        not_improved_counts,
    )
