"""Place image index and precomputed centroid weights for training."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from data.GSVCitiesDataset import GSVCitiesDataset

logger = logging.getLogger(__name__)

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def _resolve_image_path(
    row: pd.Series,
    base_path: Path,
    sfxl_train_root: Optional[Path],
    has_rel_path: bool,
) -> str:
    if has_rel_path and pd.notna(row.get("sfxl_rel_path")):
        if sfxl_train_root is None:
            raise ValueError("CSV has sfxl_rel_path but sfxl_train_root was not set")
        return str(sfxl_train_root / row["sfxl_rel_path"])
    img_name = GSVCitiesDataset.get_img_name(row)
    return str(base_path / "Images" / row["city_id"] / img_name)


def load_city_dataframes(
    cities: Sequence[str],
    base_path: Path,
) -> pd.DataFrame:
    """Load and merge per-city GSV CSVs with the same place_id prefixing as training."""
    cities = list(cities)
    if not cities:
        raise ValueError("At least one city is required")

    df = pd.read_csv(base_path / "Dataframes" / f"{cities[0]}.csv")
    for i in range(1, len(cities)):
        tmp_df = pd.read_csv(base_path / "Dataframes" / f"{cities[i]}.csv")
        tmp_df["place_id"] = tmp_df["place_id"] + (i * 10**5)
        df = pd.concat([df, tmp_df], ignore_index=True)
    return df


def filter_places_by_min_images(df: pd.DataFrame, min_img_per_place: int) -> pd.DataFrame:
    """Keep only places with at least ``min_img_per_place`` images."""
    if min_img_per_place < 1:
        raise ValueError(f"min_img_per_place must be >= 1, got {min_img_per_place}")
    counts = df.groupby("place_id")["place_id"].transform("size")
    return df[counts >= min_img_per_place].copy()


def build_places_index(
    cities: Sequence[str],
    base_path: Path,
    min_img_per_place: int = 1,
    sfxl_train_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Build a long-format table: one row per image with ``place_id`` and ``image_path``."""
    df = load_city_dataframes(cities, base_path)
    if min_img_per_place > 1:
        df = filter_places_by_min_images(df, min_img_per_place)
    has_rel_path = "sfxl_rel_path" in df.columns
    df = df.set_index("place_id", drop=False)
    df["image_path"] = [
        _resolve_image_path(row, base_path, sfxl_train_root, has_rel_path)
        for _, row in df.iterrows()
    ]
    return df


def write_places_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index(drop=True)
    out.to_csv(csv_path, index=False)
    n_places = out["place_id"].nunique()
    logger.info(
        "Wrote %s (%d images, %d places)",
        csv_path,
        len(out),
        n_places,
    )


def load_places_csv(csv_path: Path, min_img_per_place: int = 1) -> pd.DataFrame:
    """Load a places CSV and optionally filter by minimum images per place."""
    df = pd.read_csv(csv_path)
    if "place_id" not in df.columns:
        raise ValueError(f"places CSV missing place_id column: {csv_path}")
    if min_img_per_place > 1:
        df = filter_places_by_min_images(df, min_img_per_place)
    return df.reset_index(drop=True)


def filter_places_to_ids(df: pd.DataFrame, place_ids: Sequence[int]) -> pd.DataFrame:
    wanted = {int(pid) for pid in place_ids}
    return df[df["place_id"].isin(wanted)].reset_index(drop=True)


class GSVCitiesRandomImageDataset(Dataset):
    """One random image per sample; batch_size is number of images (not places)."""

    def __init__(
        self,
        places_df: pd.DataFrame,
        transform: Optional[T.Compose] = None,
        place_ids: Optional[Sequence[int]] = None,
    ):
        df = places_df
        if place_ids is not None:
            df = filter_places_to_ids(df, place_ids)
        if df.empty:
            raise ValueError("No training images after place filter")
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["place_id"])


class _PlaceImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: T.Compose):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["place_id"])


@torch.no_grad()
def compute_place_centroids(
    model: torch.nn.Module,
    device: torch.device,
    places_df: pd.DataFrame,
    image_size: int = 320,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode all place images.

    Returns ``(place_ids, centroids, place_sums, counts)``.

    Offline: one aggregate per place over **all** its images (no per-query LOO).
    ``centroids`` = L2-normalized mean; ``place_sums`` = unnormalized descriptor sum.
    """
    model.eval()
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(**IMAGENET_MEAN_STD),
    ])
    dataset = _PlaceImageDataset(places_df, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    unique_place_ids = sorted(int(x) for x in places_df["place_id"].unique())
    id_to_idx: Dict[int, int] = {pid: i for i, pid in enumerate(unique_place_ids)}
    dim: Optional[int] = None
    sums: Optional[torch.Tensor] = None
    counts = torch.zeros(len(unique_place_ids), dtype=torch.long)

    for images, place_ids in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        descriptors = out[0] if isinstance(out, tuple) else out
        descriptors = descriptors.detach().cpu()
        if dim is None:
            dim = int(descriptors.shape[1])
            sums = torch.zeros(len(unique_place_ids), dim, dtype=torch.float32)

        for desc, place_id in zip(descriptors, place_ids.tolist()):
            idx = id_to_idx[int(place_id)]
            sums[idx] += desc
            counts[idx] += 1

    assert sums is not None and dim is not None
    counts_f = counts.clamp_min(1).unsqueeze(1).to(sums.dtype)
    centroids = F.normalize(sums / counts_f, dim=-1)
    place_ids_t = torch.tensor(unique_place_ids, dtype=torch.long)
    return place_ids_t, centroids, sums, counts


def save_place_weights(
    path: Path,
    place_ids: torch.Tensor,
    centroids: torch.Tensor,
    place_sums: torch.Tensor,
    counts: torch.Tensor,
    metadata: Optional[dict] = None,
) -> None:
    """Save one full-place weight per ``place_id`` (all images; LOO is applied at train time)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "place_ids": place_ids.cpu(),
        "centroids": centroids.cpu(),
        "place_sums": place_sums.cpu(),
        "counts": counts.cpu(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    logger.info(
        "Saved place weights to %s (%d places, dim=%d)",
        path,
        place_ids.numel(),
        centroids.shape[1],
    )


def load_place_weights(
    path: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    place_sums = payload.get("place_sums")
    if place_sums is None:
        place_sums = payload.get("descriptor_sums")
    if place_sums is None:
        raise ValueError(
            f"{path} is missing place_sums; rebuild with pre_weights.py "
            "(full-place aggregates over all images, LOO at train time)."
        )
    return (
        payload["place_ids"],
        payload["centroids"],
        place_sums,
        payload["counts"],
        payload.get("metadata", {}),
    )


class PlaceCentroidTable:
    """Full-place offline weights (all images per place).

    Nothing is stored per query. At training time, vMF targets exclude the
    current query descriptor from the place aggregate, mirroring
    ``place_centroid_targets`` (``normalize(place_sum - query)``).
    """

    def __init__(
        self,
        place_ids: torch.Tensor,
        place_sums: torch.Tensor,
        centroids: Optional[torch.Tensor] = None,
        counts: Optional[torch.Tensor] = None,
    ):
        self.place_ids = place_ids.long()
        self.place_sums = place_sums.float()
        self.centroids = centroids.float() if centroids is not None else None
        self.counts = counts.long() if counts is not None else None
        self._id_to_idx = {
            int(pid): i for i, pid in enumerate(self.place_ids.tolist())
        }

    @classmethod
    def from_file(cls, path: Path) -> "PlaceCentroidTable":
        place_ids, centroids, place_sums, counts, _meta = load_place_weights(path)
        return cls(place_ids, place_sums, centroids=centroids, counts=counts)

    def _label_indices(self, labels: torch.Tensor) -> torch.Tensor:
        labels_list = labels.view(-1).tolist()
        missing = sorted({int(l) for l in labels_list if int(l) not in self._id_to_idx})
        if missing:
            raise KeyError(
                f"Place id(s) missing from pre_weights ({len(missing)} shown): {missing[:5]}"
            )
        return torch.tensor(
            [self._id_to_idx[int(l)] for l in labels_list],
            device="cpu",
            dtype=torch.long,
        )

    def targets_excluding_query(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """vMF target: remove the live query from the full-place offline aggregate."""
        idx = self._label_indices(labels)
        sums = self.place_sums[idx].to(device=z.device, dtype=z.dtype)
        return F.normalize(sums - z, dim=-1)

    def loo_targets(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`targets_excluding_query`."""
        return self.targets_excluding_query(z, labels)

def _subset_places_df(places_df: pd.DataFrame, max_places: int) -> pd.DataFrame:
    if max_places <= 0:
        return places_df
    keep = places_df["place_id"].drop_duplicates().head(max_places)
    return places_df[places_df["place_id"].isin(keep)].reset_index(drop=True)


def _debug_main() -> None:
    """Small CLI for debugging this module (see .vscode/launch.json)."""
    import argparse

    from configs.parser import load_config
    from utils.runtime import init_model

    parser = argparse.ArgumentParser(description="Debug data/place_weights.py")
    parser.add_argument(
        "--mode",
        choices=("index", "centroids", "loo"),
        default="centroids",
        help="index=build CSV; centroids=encode+aggregate; loo=check targets_excluding_query",
    )
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gsv_base_path", default="/home/shared/datasets/gsv_cities/")
    parser.add_argument("--csv_in", default="cache/gsv_preweights/places_all.csv")
    parser.add_argument("--csv_out", default="cache/gsv_preweights/places_debug.csv")
    parser.add_argument("--weights_in", default="cache/gsv_preweights/centroids_128_min16.pt")
    parser.add_argument("--weights_out", default="cache/gsv_preweights/centroids_debug.pt")
    parser.add_argument("--resume_model", default="./logs/MixVPR/resnet50_MixVPR_128_channels(64)_rows(2).ckpt")
    parser.add_argument("--min_img_per_place", type=int, default=16)
    parser.add_argument("--max_places", type=int, default=3, help="Limit places for fast debug runs")
    parser.add_argument("--city", default="Bangkok", help="Single city for index mode")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=320)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.mode == "index":
        places_df = build_places_index(
            cities=[args.city],
            base_path=Path(args.gsv_base_path),
            min_img_per_place=args.min_img_per_place,
        )
        places_df = _subset_places_df(places_df, args.max_places)
        write_places_csv(places_df, Path(args.csv_out))
        return

    places_df = load_places_csv(Path(args.csv_in), min_img_per_place=args.min_img_per_place)
    places_df = _subset_places_df(places_df, args.max_places)
    logger.info(
        "Debug subset: %d images, %d places",
        len(places_df),
        places_df["place_id"].nunique(),
    )

    if args.mode == "loo":
        table = PlaceCentroidTable.from_file(Path(args.weights_in))
        cfg = load_config(Path(args.config))
        cfg.update({
            "device": args.device,
            "method": "mixvpr",
            "model_mode": "basic",
            "descriptors_dimension": 128,
            "resume_model": args.resume_model,
        })
        device = torch.device(
            "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        _device, model = init_model(cfg)
        model = model.to(device).eval()
        transform = T.Compose([
            T.Resize((args.image_size, args.image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(**IMAGENET_MEAN_STD),
        ])
        dataset = _PlaceImageDataset(places_df, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        images, labels = next(iter(loader))
        with torch.no_grad():
            z = model(images.to(device))[0].cpu()
        targets = table.targets_excluding_query(z, labels)
        logger.info("LOO targets shape=%s, first target norm=%.4f", tuple(targets.shape), targets[0].norm().item())
        return

    cfg = load_config(Path(args.config))
    cfg.update({
        "device": args.device,
        "method": "mixvpr",
        "model_mode": "basic",
        "descriptors_dimension": 128,
        "resume_model": args.resume_model,
    })
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    _device, model = init_model(cfg)
    model = model.to(device)
    place_ids, centroids, place_sums, counts = compute_place_centroids(
        model,
        device,
        places_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    save_place_weights(
        Path(args.weights_out),
        place_ids,
        centroids,
        place_sums,
        counts,
        metadata={"debug": True, "max_places": args.max_places},
    )


if __name__ == "__main__":
    _debug_main()

