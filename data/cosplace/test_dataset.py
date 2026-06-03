import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from data.dataset_utils import read_images_paths
from utils.commons import get_cache_path

logger = logging.getLogger(__name__)

# Process-local cache for database paths/UTMs, keyed by resolved DB folder path.
# Lets repeated TestDataset instantiations for the same database folder (e.g. when
# iterating over multiple query folders of the same dataset) skip the directory
# scan and the UTM parsing on every call after the first.
_DATABASE_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_database_entry(database_folder: str) -> Dict[str, Any]:
    """Return a cached dict of database metadata for ``database_folder``.

    The first call for a given folder reads the image paths from disk; later
    calls return the same list. ``utms`` is populated lazily by the caller.
    """
    key = str(Path(database_folder).resolve())
    entry = _DATABASE_CACHE.get(key)
    if entry is None:
        entry = {"paths": read_images_paths(database_folder)}
        _DATABASE_CACHE[key] = entry
    return entry


class TestDataset(data.Dataset):
    def __init__(self, database_folder, queries_folder, positive_dist_threshold=25, image_size=None, use_labels=True, resize_test_imgs=False):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()

        # Support both "queries" and "query" folder names
        qf = Path(queries_folder)
        if not qf.exists():
            alt_name = "query" if qf.name == "queries" else "queries"
            alt = qf.parent / alt_name
            if alt.exists():
                logger.info(f"Queries folder '{qf.name}' not found, using '{alt_name}' instead.")
                queries_folder = str(alt)

        if "msls" in database_folder.lower() and image_size is not None:
            resize_test_imgs = True
            logger.info("Forcing resize_test_imgs=True for MSLS dataset")

        db_entry = _get_database_entry(database_folder)
        self.database_paths = db_entry["paths"]
        self.queries_paths = read_images_paths(queries_folder)

        self.images_paths = list(self.database_paths) + list(self.queries_paths)

        self.num_database = len(self.database_paths)
        self.num_queries = len(self.queries_paths)
        logger.debug(f"Found {self.num_database} database images and {self.num_queries} queries.")

        if use_labels:
            # Read UTM coordinates, which must be contained within the paths
            # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
            if "utms" in db_entry:
                self.database_utms = db_entry["utms"]
            else:
                try:
                    # This is just a sanity check
                    image_path = self.database_paths[0]
                    utm_east = float(image_path.split("@")[1])
                    utm_north = float(image_path.split("@")[2])
                except Exception:
                    logger.error(f"Image path {image_path} does not contain UTM coordinates.")
                    raise ValueError(
                        "The path of images should be path/to/file/@utm_east@utm_north@...@.jpg "
                        f"but it is {image_path}, which does not contain the UTM coordinates."
                    )

                self.database_utms = np.array(
                    [(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]
                ).astype(float)
                db_entry["utms"] = self.database_utms

            self.queries_utms = np.array(
                [(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]
            ).astype(float)

            # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
            # Use cache if available (with fallback to local project cache)
            cache_path = get_cache_path(queries_folder, "positives", f"dist{positive_dist_threshold}.pkl")
            
            if cache_path.exists():
                logger.info("Using cached positives from %s (queries folder: %s)", cache_path, queries_folder)
                with open(cache_path, "rb") as f:
                    self.positives_per_query = pickle.load(f)
            else:
                logger.debug("Finding positives with NearestNeighbors (this might take a while)...")
                knn = NearestNeighbors(n_jobs=-1)
                knn.fit(self.database_utms)
                self.positives_per_query = knn.radius_neighbors(
                    self.queries_utms, radius=positive_dist_threshold, return_distance=False
                )
                
                # Save to cache
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump(self.positives_per_query, f)
                    logger.info(f"Saved positives cache to {cache_path}")
                except Exception as e:
                    logger.warning(f"Could not save positives cache: {e}")

        transformations = []
        if resize_test_imgs:
            transformations.append(transforms.Resize(size=(image_size, image_size), antialias=True))
        transformations.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = transforms.Compose(transformations)

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = Image.open(image_path).convert("RGB")
        normalized_img = self.transform(pil_img)
        return normalized_img, index

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< #queries: {self.num_queries}; #database: {self.num_database} >"

    def get_positives(self):
        return self.positives_per_query
