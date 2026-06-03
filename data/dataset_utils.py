import os
import logging
from glob import glob
from PIL import ImageFile

# Ensure the training doesn't crash if an image is slightly corrupted or truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

def _resolve_cached_path(dataset_folder: str, cached_path: str) -> str:
    """
    Resolve one line from *_images_paths.txt into a valid absolute path.
    Supports relative cache entries and stale absolute entries from another machine.
    """
    normalized = cached_path.strip()
    if not normalized:
        return normalized

    # Expected cache format: relative path.
    if not os.path.isabs(normalized):
        return os.path.join(dataset_folder, normalized)

    # Absolute path that is still valid on this machine.
    if os.path.exists(normalized):
        return normalized

    # Stale absolute path from a cache file created on another machine
    # (prefix does not exist here; recover the suffix after the split folder name):
    split_name = os.path.basename(dataset_folder.rstrip("\\/"))
    marker = f"{os.sep}{split_name}{os.sep}"
    stale_abs = os.path.normpath(normalized)
    idx = stale_abs.lower().rfind(marker.lower())
    if idx != -1:
        suffix = stale_abs[idx + len(marker):]
        return os.path.join(dataset_folder, suffix)

    # Last fallback: preserve filename under current dataset folder.
    return os.path.join(dataset_folder, os.path.basename(stale_abs))

def read_images_paths(dataset_folder, get_abs_path=True):
    """
    Finds image paths within 'dataset_folder' efficiently.
    Works for both Train and Test (Database/Queries).
    
    Parameters
    ----------
    dataset_folder : str, folder containing images.
    get_abs_path : bool, if True return absolute paths, otherwise relative.
    
    Returns
    -------
    images_paths : list[str], paths of images found.
    """
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    # Normalize folder path to handle trailing slashes and different OS styles
    dataset_folder = os.path.normpath(dataset_folder)
    
    # If a .txt file is passed directly, use it as file_with_paths
    if os.path.isfile(dataset_folder) and dataset_folder.endswith(".txt"):
        file_with_paths = dataset_folder
        # Infer the base dataset folder for relative path resolution
        if dataset_folder.endswith("_images_paths.txt"):
            dataset_folder = dataset_folder[:-len("_images_paths.txt")]
        else:
            # Fallback: assume the folder is the parent directory
            dataset_folder = os.path.dirname(dataset_folder)
    else:
        file_with_paths = dataset_folder + "_images_paths.txt"
    
    # 1. FAST PATH: If the text file exists, read it directly (extremely fast for large datasets)
    if os.path.exists(file_with_paths):
        logger.debug(f"Reading paths from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        
        if get_abs_path:
            images_paths = [_resolve_cached_path(dataset_folder, path) for path in images_paths]
            
        if len(images_paths) == 0:
            raise FileNotFoundError(f"{file_with_paths} is empty")

        # Sanity check: ensure cached paths actually exist.
        sample_path = images_paths[0] if get_abs_path else os.path.join(dataset_folder, images_paths[0])
        if not os.path.exists(sample_path):
            first_existing = next(
                (p for p in images_paths if os.path.exists(p if get_abs_path else os.path.join(dataset_folder, p))),
                None,
            )
            if first_existing is None:
                raise FileNotFoundError(
                    f"Cached image paths in {file_with_paths} do not match files under {dataset_folder}. "
                    "Regenerate cache paths for this machine."
                )
            
    # 2. SLOW PATH: Use glob if no text file is provided
    else:
        logger.debug(f"Searching images in {dataset_folder} with glob() (this may be slow for large folders)")
        # Search for all files recursively
        all_files = glob(os.path.join(dataset_folder, "**", "*"), recursive=True)
        
        valid_exts = (".jpg", ".jpeg", ".png")
        
        # Optimization: Filter and process paths in a single pass without expensive system calls
        if get_abs_path:
            images_paths = sorted([
                p for p in all_files 
                if p.lower().endswith(valid_exts) and os.path.isfile(p)
            ])
        else:
            # Use simple string slicing instead of os.path.abspath to save time
            prefix_len = len(dataset_folder) + 1
            images_paths = sorted([
                p[prefix_len:] for p in all_files 
                if p.lower().endswith(valid_exts) and os.path.isfile(p)
            ])
        
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} contains no valid images {valid_exts}")
    
    return images_paths