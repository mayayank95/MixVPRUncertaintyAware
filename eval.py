"""Visual place recognition evaluation entry point."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval_metrics.combined_eval import run_combined_ece_variants
from eval_metrics.dataset_eval import eval_dataset
from utils import commons, wandb_utils
from utils.runtime import build_config_and_datasets, init_model

logger = logging.getLogger(__name__)

__all__ = ["eval_dataset", "run_eval", "main"]


def _resolve_eval_path(datasets_paths: Dict, name: str) -> Path:
    eval_ds_path = datasets_paths[name]["test"]
    if not eval_ds_path.exists():
        eval_ds_path = datasets_paths[name]["validation"]
        logger.info("[%s] 'test' folder not found, using 'val' instead.", name)
    return eval_ds_path


def _list_query_folders(
    eval_ds_path: Path,
    only_names: Optional[List[str]] = None,
) -> List[Path]:
    """Discover query folders under a dataset split; optionally keep only ``only_names``."""
    query_folders = sorted([p for p in eval_ds_path.glob("queries*") if p.is_dir()])
    if not query_folders:
        query_folders = sorted([p for p in eval_ds_path.glob("query*") if p.is_dir()])
    if only_names:
        wanted = {name.strip() for name in only_names if name.strip()}
        query_folders = [p for p in query_folders if p.name in wanted]
    return query_folders


def _display_name(entry_name: str, q_folder_name: str, query_folders: List[Path]) -> str:
    if len(query_folders) == 1 and q_folder_name in ("queries", "query"):
        return entry_name
    return f"{entry_name}_{q_folder_name}"


def run_eval(cfg, entries, datasets_paths, device, model) -> None:
    """Evaluate all configured dataset entries and query-folder splits."""
    all_panel_data: List[Dict[str, Any]] = []
    all_wandb_images: List[Optional[Dict[str, Any]]] = []
    query_folder_filter = cfg.get("eval_query_folders")
    if query_folder_filter:
        logger.info("Eval query folders filter: %s", ", ".join(query_folder_filter))

    for entry in entries:
        name = entry["name"]
        eval_ds_path = _resolve_eval_path(datasets_paths, name)
        query_folders = _list_query_folders(eval_ds_path, query_folder_filter)
        if not query_folders:
            if query_folder_filter:
                logger.warning(
                    "[%s] No query folders matching %s under %s",
                    name,
                    query_folder_filter,
                    eval_ds_path,
                )
            else:
                logger.warning("[%s] No query folders found in %s", name, eval_ds_path)
            continue

        shared_db_desc = shared_db_var = None
        for q_folder in query_folders:
            q_folder_name = q_folder.name
            display_name = _display_name(name, q_folder_name, query_folders)
            logger.info(
                "\n%s\nEvaluating dataset \033[1m%s\033[0m - queries folder: %s",
                "=" * 30,
                name,
                q_folder_name,
            )

            results = eval_dataset(
                cfg,
                model,
                device,
                display_name,
                eval_ds_path,
                queries_folder_name=q_folder_name,
                base_dataset_name=name,
                cached_db_desc=shared_db_desc,
                cached_db_var=shared_db_var,
            )

            if results.panel_data is not None:
                all_panel_data.append(results.panel_data)
            all_wandb_images.append(results.wandb_images)
            shared_db_desc, shared_db_var = results.db_desc, results.db_var

    combined_outputs = run_combined_ece_variants(cfg, all_panel_data)
    wandb_utils.log_eval_results(cfg, all_panel_data, all_wandb_images, combined_outputs)
    logger.info("%s\nAll processes finished.", "=" * 30)


def main() -> None:
    cfg, entries, datasets_paths = build_config_and_datasets()
    commons.make_deterministic(cfg["seed"])
    wandb_utils.init_wandb(cfg, job_type="eval")
    device, model = init_model(cfg)
    commons.copy_resume_model_to_log_dir(cfg, logger)
    run_eval(cfg, entries, datasets_paths, device, model)
    wandb_utils.finish_run(cfg)


if __name__ == "__main__":
    main()
