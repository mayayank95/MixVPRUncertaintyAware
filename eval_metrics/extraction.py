"""Descriptor and variance extraction for VPR evaluation."""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.test_dataset import TestDataset

logger = logging.getLogger(__name__)


def _descriptor_cache_dirs(
    args: Dict[str, Any],
    base_dataset_name: str,
    dataset_name: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Return stable on-disk cache locations for DB and query descriptors.

    Database descriptors live under the base dataset cache directory because
    they are shared by all query folders of that dataset. Query descriptors
    live under the per-query-folder subdirectory.
    """
    ckpt = args.get("resume_train") or args.get("resume_model")
    if not ckpt:
        return None, None
    ckpt_path = Path(ckpt)
    db_cache_dir = ckpt_path.parent / ckpt_path.stem / "descriptors_cache" / base_dataset_name
    query_cache_dir = db_cache_dir if dataset_name == base_dataset_name else db_cache_dir / dataset_name
    return db_cache_dir, query_cache_dir


def _try_load_database_from_disk(
    db_cache_dir: Optional[Path],
    test_ds: TestDataset,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (db_desc, db_var) or None."""
    if db_cache_dir is None:
        return None

    db_desc_path = db_cache_dir / "database_descriptors.npy"
    db_var_path = db_cache_dir / "database_variances.npy"

    if not db_desc_path.exists():
        return None

    db_desc = np.load(db_desc_path)
    if len(db_desc) != test_ds.num_database:
        logger.warning(
            "Saved database descriptors in %s size mismatch (DB: %s vs %s); re-extracting.",
            db_cache_dir,
            len(db_desc),
            test_ds.num_database,
        )
        return None

    if db_var_path.exists():
        db_var = np.load(db_var_path)
        if len(db_var) != test_ds.num_database:
            db_var = np.zeros_like(db_desc)
        else:
            logger.debug("Loaded saved database descriptors and variances from %s", db_cache_dir)
    else:
        db_var = np.zeros_like(db_desc)
        logger.debug("Loaded saved database descriptors from %s", db_cache_dir)

    return db_desc, db_var


def _try_load_queries_from_disk(
    query_cache_dir: Optional[Path],
    test_ds: TestDataset,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (q_desc, q_var) or None."""
    if query_cache_dir is None:
        return None

    q_desc_path = query_cache_dir / "queries_descriptors.npy"
    q_var_path = query_cache_dir / "queries_variances.npy"

    if not q_desc_path.exists():
        return None

    q_desc = np.load(q_desc_path)
    if len(q_desc) != test_ds.num_queries:
        logger.warning(
            "Saved query descriptors in %s size mismatch (Q: %s vs %s); re-extracting.",
            query_cache_dir,
            len(q_desc),
            test_ds.num_queries,
        )
        return None

    logger.debug("Loading saved query descriptors from %s", query_cache_dir)

    if q_var_path.exists():
        q_var = np.load(q_var_path)
        if len(q_var) != test_ds.num_queries:
            q_var = np.zeros_like(q_desc)
        else:
            logger.debug("Loaded saved query variances from %s", query_cache_dir)
    else:
        q_var = np.zeros_like(q_desc)

    return q_desc, q_var


def extract_descriptors(
    args: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    test_ds: TestDataset,
    dataset_name: str,
    base_dataset_name: Optional[str] = None,
    cached_db_desc: Optional[np.ndarray] = None,
    cached_db_var: Optional[np.ndarray] = None,
    use_descriptor_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load or extract database/query descriptors and variances.

    Pass ``cached_db_desc`` / ``cached_db_var`` from a prior query folder to
    skip re-extracting the database when shapes match the current dataset.

    Set ``use_descriptor_cache=False`` to fully disable on-disk caching (no
    load, no save). Used during training where the model changes per epoch
    and stale cached descriptors would silently corrupt validation metrics.

    Returns (db_desc, q_desc, db_var, q_var).
    """
    if use_descriptor_cache:
        db_cache_dir, query_cache_dir = _descriptor_cache_dirs(
            args,
            base_dataset_name or dataset_name,
            dataset_name,
        )
        disk_db_cached = _try_load_database_from_disk(db_cache_dir, test_ds)
        disk_query_cached = _try_load_queries_from_disk(query_cache_dir, test_ds)
    else:
        db_cache_dir = query_cache_dir = None
        disk_db_cached = disk_query_cached = None

    if disk_db_cached is not None and disk_query_cached is not None:
        db_desc, db_var = disk_db_cached
        q_desc, q_var = disk_query_cached
        if args["dry_run"]:
            db_desc = db_desc[: max(args["infer_batch_size"], args.get("num_preds_to_save", 10))]
            q_desc = q_desc[: max(1, args.get("num_queries_to_save", 3))]
            db_var = db_var[: len(db_desc)]
            q_var = q_var[: len(q_desc)]
        return db_desc, q_desc, db_var, q_var

    all_descriptors = np.zeros((len(test_ds), args["descriptors_dimension"]), dtype="float32")
    all_variances = np.zeros((len(test_ds), args["descriptors_dimension"]), dtype="float32")

    db_cache_found = False
    if disk_db_cached is not None:
        db_desc_cached, db_var_cached = disk_db_cached
        all_descriptors[: test_ds.num_database] = db_desc_cached
        all_variances[: test_ds.num_database] = db_var_cached
        db_cache_found = True
    elif (
        cached_db_desc is not None
        and cached_db_var is not None
        and cached_db_desc.shape[0] == test_ds.num_database
        and cached_db_var.shape[0] == test_ds.num_database
    ):
        logger.info("Using in-memory cached database features.")
        all_descriptors[: test_ds.num_database] = cached_db_desc
        all_variances[: test_ds.num_database] = cached_db_var
        db_cache_found = True

    query_cache_found = False
    if disk_query_cached is not None:
        q_desc_cached, q_var_cached = disk_query_cached
        query_start = test_ds.num_database
        query_end = query_start + test_ds.num_queries
        all_descriptors[query_start:query_end] = q_desc_cached
        all_variances[query_start:query_end] = q_var_cached
        query_cache_found = True

    with torch.inference_mode():
        if not db_cache_found:
            db_subset = Subset(test_ds, list(range(test_ds.num_database)))
            db_loader = DataLoader(
                db_subset,
                batch_size=args["infer_batch_size"],
                num_workers=args["num_workers"],
                pin_memory=(device.type == "cuda"),
            )
            logger.debug("Extracting database features...")
            for images, indices in tqdm(db_loader):
                desc, var = model(images.to(device))
                all_descriptors[indices.numpy(), :] = desc.cpu().numpy()
                all_variances[indices.numpy(), :] = var.cpu().numpy()
                if args["dry_run"]:
                    break

        if not query_cache_found:
            q_subset = Subset(
                test_ds,
                list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries)),
            )
            q_loader = DataLoader(
                q_subset,
                batch_size=1,
                num_workers=args["num_workers"],
                pin_memory=(device.type == "cuda"),
            )
            logger.debug("Extracting query features...")
            for images, indices in tqdm(q_loader):
                desc, var = model(images.to(device))
                all_descriptors[indices.numpy(), :] = desc.cpu().numpy()
                all_variances[indices.numpy(), :] = var.cpu().numpy()
                if args["dry_run"]:
                    break

    db_desc = all_descriptors[: test_ds.num_database]
    q_desc = all_descriptors[test_ds.num_database :]
    db_var = all_variances[: test_ds.num_database]
    q_var = all_variances[test_ds.num_database :]

    if args["dry_run"]:
        db_desc = db_desc[: max(args["infer_batch_size"], args.get("num_preds_to_save", 10))]
        q_desc = q_desc[: max(1, args.get("num_queries_to_save", 3))]
        db_var = db_var[: len(db_desc)]
        q_var = q_var[: len(q_desc)]

    if (
        use_descriptor_cache
        and args.get("save_descriptors")
        and not args["dry_run"]
        and db_cache_dir is not None
        and query_cache_dir is not None
    ):
        db_cache_dir.mkdir(parents=True, exist_ok=True)
        query_cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(db_cache_dir / "database_descriptors.npy", db_desc)
        np.save(query_cache_dir / "queries_descriptors.npy", q_desc)
        if args["model_mode"] == "uncertainty":
            np.save(db_cache_dir / "database_variances.npy", db_var)
            np.save(query_cache_dir / "queries_variances.npy", q_var)
        logger.info(
            "Saved database descriptors to %s and query descriptors to %s",
            db_cache_dir,
            query_cache_dir,
        )

    return db_desc, q_desc, db_var, q_var
