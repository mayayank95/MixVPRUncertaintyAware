"""Per-place centroid and intra-place distance statistics in a model's descriptor space."""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.place_weights import IMAGENET_MEAN_STD, _PlaceImageDataset
from torchvision import transforms as T


def _build_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(**IMAGENET_MEAN_STD),
    ])


@torch.no_grad()
def encode_places_by_id(
    model: torch.nn.Module,
    device: torch.device,
    places_df: pd.DataFrame,
    image_size: int,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[int, torch.Tensor]:
    """Encode all images; return ``place_id -> [N, D]`` L2-normalized descriptors."""
    model.eval()
    dataset = _PlaceImageDataset(places_df, _build_transform(image_size))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    by_place: Dict[int, List[torch.Tensor]] = {}
    for images, place_ids in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        descriptors = out[0] if isinstance(out, tuple) else out
        descriptors = F.normalize(descriptors.detach().cpu(), dim=-1)
        for desc, place_id in zip(descriptors, place_ids.tolist()):
            by_place.setdefault(int(place_id), []).append(desc)

    return {pid: torch.stack(rows, dim=0) for pid, rows in by_place.items()}


def cosine_distances_to_centroid(
    descriptors: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """L2-normalized mean centroid; cosine distance ``1 - cos_sim`` per image."""
    if descriptors.ndim != 2 or descriptors.shape[0] == 0:
        raise ValueError(f"expected [N, D] descriptors, got {tuple(descriptors.shape)}")
    centroid = F.normalize(descriptors.mean(dim=0), dim=-1)
    cos_sim = (descriptors * centroid).sum(dim=-1).clamp(-1.0, 1.0)
    distances = 1.0 - cos_sim
    return centroid, distances


def loo_cosine_distances(descriptors: torch.Tensor) -> torch.Tensor:
    """Per-image distance to leave-one-out centroid (all other images in the place)."""
    n = descriptors.shape[0]
    if n <= 1:
        return torch.zeros(n, dtype=torch.float32)
    total = descriptors.sum(dim=0, keepdim=True)
    loo_centroids = F.normalize(total - descriptors, dim=-1)
    cos_sim = (descriptors * loo_centroids).sum(dim=-1).clamp(-1.0, 1.0)
    return 1.0 - cos_sim


def pairwise_cosine_distances(descriptors: torch.Tensor) -> torch.Tensor:
    """Cosine distances for all unordered image pairs within a place."""
    n = descriptors.shape[0]
    if n < 2:
        return torch.zeros(0, dtype=torch.float32)
    sim = descriptors @ descriptors.T
    dist = 1.0 - sim
    idx = torch.triu_indices(n, n, offset=1)
    return dist[idx[0], idx[1]]


def _mean_var(values: torch.Tensor) -> Tuple[float, float]:
    if values.numel() == 0:
        return float("nan"), float("nan")
    return float(values.mean().item()), float(values.var(unbiased=False).item())


def place_centroids_and_distances(
    descriptors_by_place: Dict[int, torch.Tensor],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Per-place full centroid and image-to-centroid cosine distances."""
    centroids: Dict[int, torch.Tensor] = {}
    centroid_dists: Dict[int, torch.Tensor] = {}
    for place_id, descriptors in descriptors_by_place.items():
        centroid, distances = cosine_distances_to_centroid(descriptors)
        centroids[place_id] = centroid
        centroid_dists[place_id] = distances
    return centroids, centroid_dists


def nearest_other_centroid_distances(centroids: Dict[int, torch.Tensor]) -> Dict[int, float]:
    """Cosine distance from each place centroid to its nearest *other* place centroid."""
    if not centroids:
        return {}
    if len(centroids) == 1:
        only = next(iter(centroids))
        return {only: float("nan")}

    place_ids = sorted(centroids)
    stacked = torch.stack([centroids[pid] for pid in place_ids], dim=0)
    dist = 1.0 - (stacked @ stacked.T).clamp(-1.0, 1.0)
    dist.fill_diagonal_(float("inf"))
    min_dists = dist.min(dim=1).values
    return {pid: float(min_dists[i].item()) for i, pid in enumerate(place_ids)}


def place_distance_stats(descriptors_by_place: Dict[int, torch.Tensor]) -> pd.DataFrame:
    """Per-place intra-place stats plus inter-place separation vs nearest neighbor."""
    centroids, centroid_dists_by_place = place_centroids_and_distances(descriptors_by_place)
    nearest_other = nearest_other_centroid_distances(centroids)

    rows = []
    for place_id in sorted(descriptors_by_place):
        descriptors = descriptors_by_place[place_id]
        centroid_dists = centroid_dists_by_place[place_id]
        loo_dists = loo_cosine_distances(descriptors)
        pair_dists = pairwise_cosine_distances(descriptors)
        centroid_mean, centroid_var = _mean_var(centroid_dists)
        loo_mean, loo_var = _mean_var(loo_dists)
        pair_mean, pair_var = _mean_var(pair_dists)
        nearest_dist = nearest_other[place_id]
        if nearest_dist > 0 and centroid_mean == centroid_mean:
            separation_ratio = nearest_dist / centroid_mean
        else:
            separation_ratio = float("nan")

        rows.append({
            "place_id": int(place_id),
            "n_images": int(descriptors.shape[0]),
            "mean_dist": centroid_mean,
            "var_dist": centroid_var,
            "loo_mean_dist": loo_mean,
            "loo_var_dist": loo_var,
            "pairwise_mean_dist": pair_mean,
            "pairwise_var_dist": pair_var,
            "nearest_other_centroid_dist": nearest_dist,
            "separation_ratio": separation_ratio,
        })
    return pd.DataFrame(rows)


def _rename_model_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    metric_cols = [
        "mean_dist",
        "var_dist",
        "loo_mean_dist",
        "loo_var_dist",
        "pairwise_mean_dist",
        "pairwise_var_dist",
        "nearest_other_centroid_dist",
        "separation_ratio",
    ]
    return df.rename(columns={col: f"{prefix}_{col}" for col in metric_cols})


def merge_model_stats(
    mixvpr_stats: pd.DataFrame,
    cosplace_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Join MixVPR and CosPlace per-place stats on ``place_id``."""
    return merge_many_model_stats([("mixvpr", mixvpr_stats), ("cosplace", cosplace_stats)])


def merge_many_model_stats(
    named_stats: List[Tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Join per-place stats from multiple models on ``place_id``."""
    if not named_stats:
        raise ValueError("named_stats must not be empty")
    merged = _rename_model_columns(named_stats[0][1], named_stats[0][0])
    for slug, stats in named_stats[1:]:
        renamed = _rename_model_columns(stats, slug)
        metric_cols = ["place_id"] + [c for c in renamed.columns if c.startswith(f"{slug}_")]
        merged = merged.merge(renamed[metric_cols], on="place_id", how="inner")
    return merged.sort_values("place_id").reset_index(drop=True)


HEADLINE_METRICS = [
    "mean_dist",
    "var_dist",
    "loo_mean_dist",
    "loo_var_dist",
    "pairwise_mean_dist",
    "pairwise_var_dist",
    "nearest_other_centroid_dist",
    "separation_ratio",
]


def headline_summary(stats: pd.DataFrame, slug: str) -> Dict[str, float]:
    """Mean of each metric column over places for one model."""
    out: Dict[str, float] = {}
    for metric in HEADLINE_METRICS:
        col = f"{slug}_{metric}"
        if col in stats.columns:
            out[f"{slug}_{metric}_over_places"] = float(stats[col].mean())
    return out
