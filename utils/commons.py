
import torch
import random
import logging
import numpy as np
from pathlib import Path
import shutil
import hashlib


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def make_deterministic(seed: int = 0):
    """Make results deterministic. If seed == -1, do not make deterministic.
        Running your script in a deterministic way might slow it down.
        Note that for some packages (eg: sklearn's PCA) this function is not enough.
    """
    seed = int(seed)
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_cudnn(benchmark: bool = False):
    if benchmark:
        # If speed is requested:
        torch.backends.cudnn.benchmark = True 
        torch.backends.cudnn.deterministic = False
        logging.debug("cuDNN benchmark ENABLED: Training will be FASTER but not bit-by-bit reproducible.")
    else:
        # If reproducibility is requested (already set to False by make_deterministic):
        # This ensures exact results if the same seed is used
        logging.debug("cuDNN benchmark DISABLED: Training will be bit-by-bit DETERMINISTIC (Slower).")


def copy_resume_model_to_log_dir(cfg, logger: logging.Logger):
    """
    If cfg['resume_model'] is set and exists on disk, copy it into cfg['log_dir'].
    This is shared between training and evaluation entrypoints.
    Skip when dry_run (log_dir may not exist; copying would create a file at log_dir and break eval paths).
    """
    resume_model = cfg.get("resume_model")
    log_dir = cfg.get("log_dir")

    if not resume_model or not log_dir:
        return
    if cfg.get("dry_run"):
        return

    src = Path(resume_model)
    if not src.exists():
        logger.warning(f"resume_model path does not exist, skipping copy: {src}")
        return

    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        logger.warning(f"log_dir is not a directory, skipping copy: {log_dir}")
        return

    try:
        shutil.copy(src, log_dir)
        logger.debug(f"Copied resume model from {src} to {log_dir}")
    except Exception as e:
        logger.error(f"Failed to copy resume model from {src} to {log_dir}: {e}")

def get_cache_path(base_folder, sub_path, filename):
    """Try to get a path in the base_folder/cache, or fallback to local project/cache."""
    # 1. Try dataset-local cache
    try_path = Path(base_folder) / "cache" / sub_path / filename
    try:
        try_path.parent.mkdir(parents=True, exist_ok=True)
        # Check if we can actually write here
        test_file = try_path.parent / ".write_test"
        test_file.touch()
        test_file.unlink()
        return try_path
    except (PermissionError, OSError):
        # 2. Fallback to project-local cache
        project_root = Path(__file__).resolve().parent.parent
        path_hash = hashlib.md5(str(base_folder).encode()).hexdigest()[:8]
        fallback_dir = project_root / "cache" / path_hash / sub_path
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir / filename
