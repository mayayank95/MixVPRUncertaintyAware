"""Place image index and precomputed centroid weights for training."""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from data.GSVCitiesDataset import GSVCitiesDataset
from data.medoid_exclusion import exclude_medoid_training_images

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
        medoid_image_paths: Optional[Sequence[str]] = None,
    ):
        df = exclude_medoid_training_images(
            places_df,
            medoid_image_paths,
            log=logger,
        )
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
        return img, int(row["place_id"]), str(row["image_path"])


@torch.no_grad()
def _medoid_from_place_descriptors(
    descriptors: torch.Tensor,
    image_paths: List[str],
) -> Tuple[torch.Tensor, str]:
    """Medoid = argmax_j sum_k z_j^T z_k over L2-normalized place descriptors."""
    z = F.normalize(descriptors, dim=-1)
    scores = z @ z.sum(dim=0)
    best = int(scores.argmax().item())
    return z[best], image_paths[best]


@torch.no_grad()
def compute_place_centroids(
    model: torch.nn.Module,
    device: torch.device,
    places_df: pd.DataFrame,
    image_size: int = 320,
    batch_size: int = 64,
    num_workers: int = 4,
    *,
    compute_medoids: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[List[str]],
]:
    """Encode all place images.

    Returns ``(place_ids, centroids, place_sums, counts[, medoids[, medoid_image_paths]])``.

    Offline: one aggregate per place over **all** its images (no per-query LOO).
    ``centroids`` = L2-normalized mean; ``place_sums`` = unnormalized descriptor sum.
    ``medoids`` = L2-normalized descriptor of the place medoid
    (argmax_j sum_k z_j^T z_k); ``medoid_image_paths`` = that image's path.
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
    place_descs: DefaultDict[int, List[torch.Tensor]] = defaultdict(list)
    place_paths: DefaultDict[int, List[str]] = defaultdict(list)

    for images, place_ids, image_paths in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        descriptors = out[0] if isinstance(out, tuple) else out
        descriptors = descriptors.detach().cpu()
        if dim is None:
            dim = int(descriptors.shape[1])
            sums = torch.zeros(len(unique_place_ids), dim, dtype=torch.float32)

        for desc, place_id, image_path in zip(
            descriptors, place_ids.tolist(), image_paths
        ):
            pid = int(place_id)
            idx = id_to_idx[pid]
            sums[idx] += desc
            counts[idx] += 1
            if compute_medoids:
                place_descs[pid].append(desc)
                place_paths[pid].append(image_path)

    assert sums is not None and dim is not None
    counts_f = counts.clamp_min(1).unsqueeze(1).to(sums.dtype)
    centroids = F.normalize(sums / counts_f, dim=-1)
    place_ids_t = torch.tensor(unique_place_ids, dtype=torch.long)
    medoids = None
    medoid_image_paths = None
    if compute_medoids:
        medoids = torch.zeros(len(unique_place_ids), dim, dtype=torch.float32)
        medoid_image_paths = []
        for pid in unique_place_ids:
            idx = id_to_idx[pid]
            medoid, path = _medoid_from_place_descriptors(
                torch.stack(place_descs[pid]),
                place_paths[pid],
            )
            medoids[idx] = medoid
            medoid_image_paths.append(path)
    return place_ids_t, centroids, sums, counts, medoids, medoid_image_paths


def save_place_weights(
    path: Path,
    place_ids: torch.Tensor,
    centroids: torch.Tensor,
    place_sums: torch.Tensor,
    counts: torch.Tensor,
    metadata: Optional[dict] = None,
    medoids: Optional[torch.Tensor] = None,
    medoid_image_paths: Optional[List[str]] = None,
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
    if medoids is not None:
        payload["medoids"] = medoids.cpu()
    if medoid_image_paths is not None:
        payload["medoid_image_paths"] = list(medoid_image_paths)
    torch.save(payload, path)
    logger.info(
        "Saved place weights to %s (%d places, dim=%d%s%s)",
        path,
        place_ids.numel(),
        centroids.shape[1],
        ", with medoids" if medoids is not None else "",
        ", medoid index only" if medoids is None and medoid_image_paths is not None else "",
    )


def _load_pre_weights_payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _place_sums_from_payload(payload: dict) -> torch.Tensor:
    place_sums = payload.get("place_sums")
    if place_sums is None:
        place_sums = payload.get("descriptor_sums")
    if place_sums is None:
        raise ValueError(
            "pre_weights file is missing place_sums; rebuild with pre_weights.py "
            "(full-place aggregates over all images, LOO at train time)."
        )
    return place_sums


def load_place_weights(
    path: Path,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict,
    Optional[torch.Tensor],
    Optional[List[str]],
]:
    """Load every field from a pre_weights file (for debug / offline tools)."""
    payload = _load_pre_weights_payload(path)
    return (
        payload["place_ids"],
        payload["centroids"],
        _place_sums_from_payload(payload),
        payload["counts"],
        payload.get("metadata", {}),
        payload.get("medoids"),
        payload.get("medoid_image_paths"),
    )


class PlaceCentroidTable:
    """Precomputed per-place data for training.

    Only fields required by the active target mode are retained after loading
    (e.g. medoid-live keeps place ids + medoid paths; centroid LOO keeps
    place_sums). Large descriptor tensors are dropped so they can be freed.
    """

    def __init__(
        self,
        place_ids: torch.Tensor,
        *,
        place_sums: Optional[torch.Tensor] = None,
        centroids: Optional[torch.Tensor] = None,
        counts: Optional[torch.Tensor] = None,
        medoids: Optional[torch.Tensor] = None,
        medoid_image_paths: Optional[List[str]] = None,
    ):
        self.place_ids = place_ids.long()
        self.place_sums = place_sums.float() if place_sums is not None else None
        self.centroids = centroids.float() if centroids is not None else None
        self.counts = counts.long() if counts is not None else None
        self.medoids = medoids.float() if medoids is not None else None
        self.medoid_image_paths = (
            list(medoid_image_paths) if medoid_image_paths is not None else None
        )
        self._id_to_idx = {
            int(pid): i for i, pid in enumerate(self.place_ids.tolist())
        }
        self._medoid_path_by_place: Dict[int, str] = {}
        if self.medoid_image_paths is not None:
            for pid, path in zip(self.place_ids.tolist(), self.medoid_image_paths):
                self._medoid_path_by_place[int(pid)] = path

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        use_medoid_targets: bool = False,
        medoid_live_targets: bool = False,
        load_all: bool = False,
    ) -> "PlaceCentroidTable":
        """Load pre_weights, keeping only fields needed for the training mode."""
        payload = _load_pre_weights_payload(path)
        place_ids = payload["place_ids"]

        place_sums = None
        centroids = None
        counts = None
        medoids = None
        medoid_paths = None
        kept: List[str] = ["place_ids"]

        if load_all:
            place_sums = _place_sums_from_payload(payload)
            centroids = payload.get("centroids")
            counts = payload.get("counts")
            medoids = payload.get("medoids")
            medoid_paths = payload.get("medoid_image_paths")
            kept.extend(["place_sums", "centroids", "counts", "medoids", "medoid_image_paths"])
        elif use_medoid_targets:
            medoid_paths = payload.get("medoid_image_paths")
            kept.append("medoid_image_paths")
            if not medoid_live_targets:
                medoids = payload.get("medoids")
                kept.append("medoids")
        else:
            place_sums = _place_sums_from_payload(payload)
            kept.append("place_sums")

        del payload

        table = cls(
            place_ids,
            place_sums=place_sums,
            centroids=centroids,
            counts=counts,
            medoids=medoids,
            medoid_image_paths=medoid_paths,
        )
        logger.info(
            "Loaded pre_weights %s: %d places, retained %s",
            path,
            table.place_ids.numel(),
            ", ".join(kept),
        )
        return table

    def validate_for_training(
        self,
        *,
        use_medoid_targets: bool,
        medoid_live_targets: bool,
        path: Optional[Path] = None,
    ) -> str:
        """Check required fields for the chosen target mode; return target kind for logging."""
        if medoid_live_targets:
            target_kind = "medoid-live"
        elif use_medoid_targets:
            target_kind = "medoid"
        else:
            target_kind = "centroid"
        path_hint = f" in {path}" if path is not None else ""
        if use_medoid_targets and self.medoid_image_paths is None:
            raise ValueError(
                f"pre_weights{path_hint} has no medoid_image_paths; "
                "re-run pre_weights.py to rebuild weights."
            )
        if (
            use_medoid_targets
            and not medoid_live_targets
            and self.medoids is None
        ):
            raise ValueError(
                f"pre_weights{path_hint} has no medoids; re-run pre_weights.py to rebuild weights "
                "or pass --medoid_live_targets."
            )
        return target_kind

    @property
    def descriptor_dim(self) -> Optional[int]:
        if self.centroids is not None:
            return int(self.centroids.shape[1])
        if self.place_sums is not None:
            return int(self.place_sums.shape[1])
        if self.medoids is not None:
            return int(self.medoids.shape[1])
        return None

    @property
    def medoid_path_by_place(self) -> Dict[int, str]:
        return dict(self._medoid_path_by_place)

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
        if self.place_sums is None:
            raise ValueError(
                "centroid LOO targets require place_sums; reload pre_weights without medoid mode"
            )
        idx = self._label_indices(labels)
        sums = self.place_sums[idx].to(device=z.device, dtype=z.dtype)
        return F.normalize(sums - z, dim=-1)

    def medoid_targets(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """vMF target: precomputed place medoid (medoid images are excluded from training)."""
        if self.medoids is None:
            raise ValueError(
                "pre_weights file has no medoids; re-run pre_weights.py to rebuild weights "
                "or use --medoid_live_targets to re-encode medoid images each step."
            )
        idx = self._label_indices(labels)
        return self.medoids[idx].to(device=z.device, dtype=z.dtype)

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
        table = PlaceCentroidTable.from_file(Path(args.weights_in), load_all=True)
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
        images, labels, _image_paths, _rows = next(iter(loader))
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
    place_ids, centroids, place_sums, counts, medoids, medoid_image_paths = (
        compute_place_centroids(
        model,
        device,
        places_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        )
    )
    save_place_weights(
        Path(args.weights_out),
        place_ids,
        centroids,
        place_sums,
        counts,
        medoids=medoids,
        medoid_image_paths=medoid_image_paths,
        metadata={"debug": True, "max_places": args.max_places},
    )


if __name__ == "__main__":
    _debug_main()

