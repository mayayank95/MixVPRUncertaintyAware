import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def resolve_dataset_paths(cfg, entries) -> Dict[str, Dict[str, Path]]:
    """Validate dataset entries and resolve their train/validation/test paths."""
    data_root = Path(cfg["data_folder"]).expanduser()
    dataset_paths: Dict[str, Dict[str, Path]] = {}

    for e in entries:
        name = e.get("name")
        if not name:
            logger.error("Validation failed: Each entry must have a non-empty 'name'")
            raise ValueError("Each entry must have a non-empty 'name'")

        path = e.get("path")
        if not path:
            logger.error(f"Validation failed: Entry '{name}' missing required field: path")
            raise ValueError(f"Entry '{name}' missing required field: path")

        src = (data_root / path).resolve()
        if not src.exists():
            logger.error(f"[{name}] Source not found: {src}")
            raise FileNotFoundError(f"[{name}] Source not found: {src}")

        dataset_paths[name] = {
            "train": (src / "train").resolve(strict=False),
            "validation": (src / "val").resolve(strict=False),
            "test": (src / "test").resolve(strict=False),
        }

    return dataset_paths
