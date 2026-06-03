import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from configs.parser import build_config
from data.dataset_paths import resolve_dataset_paths
from models.get_model import get_model

logger = logging.getLogger(__name__)


def build_config_and_datasets() -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Path]]]:
    """Shared helper for entrypoints (train.py, eval.py, etc.).
    Parses CLI + config, prepares datasets, returns (cfg, entries, datasets_paths).
    """
    cfg, entries = build_config()
    logger.info(" ".join(sys.argv))
    logger.info(f"The outputs are being saved in {cfg['log_dir']}")
    datasets_paths = resolve_dataset_paths(cfg, entries)
    logger.info(f"Loaded {len(entries)} config entr{'y' if len(entries) == 1 else 'ies'}.")
    return cfg, entries, datasets_paths


def init_model(args: Dict[str, Any]):
    """Create model from config and move to device."""
    model = get_model(args)
    device = torch.device(args["device"])
    model = model.to(device)
    return device, model
