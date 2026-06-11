#!/usr/bin/env python3
"""Build a GSV-Cities-style index for SF-XL CosPlace training classes."""
from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "place_id",
    "year",
    "month",
    "northdeg",
    "city_id",
    "lat",
    "lon",
    "panoid",
    "sfxl_rel_path",
]


def normalize_sfxl_path(path: str, sfxl_train_root: Path | None = None) -> str:
    """Return path relative to ``sfxl_train_root`` (cache may store absolutes)."""
    p = Path(path)
    if sfxl_train_root is not None:
        try:
            return str(p.resolve().relative_to(sfxl_train_root.resolve()))
        except ValueError:
            pass
    m = re.search(r"(37\.\d+/.*)", path.replace("\\", "/"))
    if m:
        return m.group(1)
    return path.lstrip("/")


def parse_sfxl_rel_path(rel_path: str) -> dict:
    """Parse SF-XL relative path (CosPlace format) into GSV-like metadata."""
    parts = rel_path.split("@")
    if len(parts) < 14:
        raise ValueError(f"Unexpected SF-XL path format: {rel_path!r}")

    lat_folder = parts[0].rstrip("/")
    city_id = "SF" + lat_folder.replace(".", "")
    northdeg = int(float(parts[9]))
    year = int(parts[13][:4])
    month = int(parts[13][4:6])

    return {
        "year": year,
        "month": month,
        "northdeg": northdeg,
        "city_id": city_id,
        "lat": float(parts[5]),
        "lon": float(parts[6]),
        "panoid": parts[7],
        "sfxl_rel_path": rel_path,
        "lat_folder": lat_folder,
    }


def gsv_image_name(city_id: str, place_id: int, row: dict) -> str:
    """Match ``GSVCitiesDataset.get_img_name`` naming convention."""
    pl_id = str(place_id % 10**5).zfill(7)
    year = str(row["year"]).zfill(4)
    month = str(row["month"]).zfill(2)
    northdeg = str(row["northdeg"]).zfill(3)
    lat, lon = str(row["lat"]), str(row["lon"])
    panoid = row["panoid"]
    return (
        f"{city_id}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
    )


def load_images_per_class(cache_path: Path) -> dict:
    """Return CosPlace ``images_per_class`` dict from cache file."""
    obj = torch.load(cache_path, map_location="cpu", weights_only=False)
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj[1]
    if isinstance(obj, dict) and "images_per_class" in obj:
        return obj["images_per_class"]
    raise ValueError(
        f"Unrecognized cache format in {cache_path} (type={type(obj).__name__})"
    )


def build_bucket_place_ids(
    images_per_class: dict,
    sfxl_train_root: Path,
) -> dict[tuple, dict[str, int]]:
    """Assign sequential place_id per latitude bucket for each CosPlace class."""
    classes_per_bucket: dict[str, list[tuple]] = defaultdict(list)
    for class_id, paths in images_per_class.items():
        rel = normalize_sfxl_path(paths[0], sfxl_train_root)
        meta = parse_sfxl_rel_path(rel)
        classes_per_bucket[meta["lat_folder"]].append(class_id)

    class_to_place: dict[tuple, dict[str, int]] = {}
    for _lat_folder, class_ids in classes_per_bucket.items():
        class_ids.sort()
        for place_id, class_id in enumerate(class_ids, start=1):
            class_to_place[class_id] = {"place_id": place_id}
    return class_to_place


def iter_rows(
    images_per_class: dict,
    class_to_place: dict[tuple, dict[str, int]],
    sfxl_train_root: Path,
    max_images_per_place: int | None,
):
    """Yield CSV rows grouped by city_id."""
    rows_per_city: dict[str, list[dict]] = defaultdict(list)

    for class_id, paths in images_per_class.items():
        place_info = class_to_place[class_id]
        place_id = place_info["place_id"]
        if max_images_per_place is not None:
            paths = paths[:max_images_per_place]

        for raw_path in paths:
            rel_path = normalize_sfxl_path(raw_path, sfxl_train_root)
            meta = parse_sfxl_rel_path(rel_path)
            rows_per_city[meta["city_id"]].append(
                {
                    "place_id": place_id,
                    "year": meta["year"],
                    "month": meta["month"],
                    "northdeg": meta["northdeg"],
                    "city_id": meta["city_id"],
                    "lat": meta["lat"],
                    "lon": meta["lon"],
                    "panoid": meta["panoid"],
                    "sfxl_rel_path": rel_path,
                }
            )

    return rows_per_city


def write_dataframes(rows_per_city: dict[str, list[dict]], out_dir: Path) -> None:
    df_dir = out_dir / "Dataframes"
    df_dir.mkdir(parents=True, exist_ok=True)
    for city_id, rows in sorted(rows_per_city.items()):
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        out_csv = df_dir / f"{city_id}.csv"
        df.to_csv(out_csv, index=False)
        n_places = df["place_id"].nunique()
        logger.info(
            "Wrote %s (%d images, %d places)",
            out_csv,
            len(df),
            n_places,
        )


def create_symlinks(
    rows_per_city: dict[str, list[dict]],
    out_dir: Path,
    sfxl_train_root: Path,
) -> None:
    images_root = out_dir / "Images"
    images_root.mkdir(parents=True, exist_ok=True)

    for city_id, rows in rows_per_city.items():
        city_dir = images_root / city_id
        city_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            name = gsv_image_name(city_id, row["place_id"], row)
            link = city_dir / name
            if link.exists():
                continue
            src = sfxl_train_root / row["sfxl_rel_path"]
            if not src.exists():
                raise FileNotFoundError(f"Missing SF-XL image: {src}")
            link.symlink_to(src)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SF-XL CosPlace cache to GSV-Cities-style Dataframes."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sfxl-train-root", type=Path, required=True)
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create Images/<city_id>/ symlinks with GSV-style filenames",
    )
    parser.add_argument(
        "--max-images-per-place",
        type=int,
        default=None,
        help="Optional cap images per class/place",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 100 classes (for quick validation)",
    )
    args = parser.parse_args()

    if not args.cache.is_file():
        raise FileNotFoundError(args.cache)
    if not args.sfxl_train_root.is_dir():
        raise FileNotFoundError(args.sfxl_train_root)

    logger.info("Loading cache %s", args.cache)
    images_per_class = load_images_per_class(args.cache)
    logger.info("Loaded %d classes", len(images_per_class))

    if args.dry_run:
        keys = list(images_per_class.keys())[:100]
        images_per_class = {k: images_per_class[k] for k in keys}
        logger.info("Dry-run: using %d classes", len(images_per_class))

    class_to_place = build_bucket_place_ids(images_per_class, args.sfxl_train_root)
    rows_per_city = iter_rows(
        images_per_class,
        class_to_place,
        args.sfxl_train_root,
        args.max_images_per_place,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    write_dataframes(rows_per_city, args.out)

    if args.symlink:
        logger.info("Creating symlinks under %s/Images", args.out)
        create_symlinks(rows_per_city, args.out, args.sfxl_train_root)

    city_list = sorted(rows_per_city)
    logger.info("City buckets (%d): %s", len(city_list), ", ".join(city_list))


if __name__ == "__main__":
    main()
