"""Exclude medoid images from the training query pool (frozen or live targets)."""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def exclude_medoid_training_images(
    places_df: pd.DataFrame,
    medoid_image_paths: Optional[Sequence[str]],
    *,
    log: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Drop rows whose ``image_path`` is a place medoid (not used as query images)."""
    if not medoid_image_paths:
        return places_df
    if "image_path" not in places_df.columns:
        raise ValueError(
            "exclude_medoid_training_images requires an image_path column on places_df"
        )
    exclude = set(medoid_image_paths)
    n_before = len(places_df)
    filtered = places_df[~places_df["image_path"].isin(exclude)].copy()
    n_after = len(filtered)
    log = log or logger
    if n_before != n_after:
        log.info(
            "Excluded %d medoid training images (%d -> %d rows)",
            n_before - n_after,
            n_before,
            n_after,
        )
    # P×K GSVCitiesDataset indexes by place_id; flat CSV uses a default RangeIndex.
    if filtered.index.name == "place_id":
        return filtered
    return filtered.reset_index(drop=True)
