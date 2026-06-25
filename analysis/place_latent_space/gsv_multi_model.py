#!/usr/bin/env python3
"""Per-place latent-space stats for multiple VPR models on GSV-Cities."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
_DIR = Path(__file__).resolve().parent
for _path in (_ROOT, _DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd
import torch

from model_loaders import load_model
from stats import (
    HEADLINE_METRICS,
    encode_places_by_id,
    headline_summary,
    merge_many_model_stats,
    place_distance_stats,
)
from configs.parser import load_config
from data.place_weights import load_places_csv

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("analysis/place_latent_space/outputs/gsv")
DEFAULT_CSV = Path("cache/gsv_preweights/places_all.csv")


def _subset_places(places_df: pd.DataFrame, max_places: int) -> pd.DataFrame:
    if max_places <= 0:
        return places_df
    keep = places_df["place_id"].drop_duplicates().head(max_places)
    return places_df[places_df["place_id"].isin(keep)].reset_index(drop=True)


def _cache_path(
    cache_dir: Optional[Path],
    slug: str,
    tag: str,
    spec: Dict[str, Any],
) -> Optional[Path]:
    if cache_dir is None:
        return None
    legacy = spec.get("legacy_cache")
    if legacy:
        legacy_path = Path(legacy)
        if legacy_path.exists():
            return legacy_path
    return cache_dir / f"{slug}_{tag}_encodings.pt"


def _run_model(
    spec: Dict[str, Any],
    device: torch.device,
    places_df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    cache_path: Optional[Path],
) -> pd.DataFrame:
    slug = spec["slug"]
    image_size = int(spec.get("image_size", 512))

    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached %s encodings from %s", slug, cache_path)
        descriptors_by_place = torch.load(cache_path, map_location="cpu", weights_only=False)
        return place_distance_stats(descriptors_by_place)

    logger.info("Encoding GSV images with %s (loader=%s, image_size=%d)", slug, spec["loader"], image_size)
    model = load_model(spec, device)
    descriptors_by_place = encode_places_by_id(
        model,
        device,
        places_df,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(descriptors_by_place, cache_path)
        logger.info("Cached %s encodings to %s", slug, cache_path)

    return place_distance_stats(descriptors_by_place)


def _load_preset(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="GSV per-place stats for multiple VPR models.")
    parser.add_argument("--preset", type=Path, help="JSON preset with tag + models list.")
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--places_csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--min_img_per_place", type=int, default=4)
    parser.add_argument("--max_places", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    if args.preset is None:
        parser.error("--preset is required")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    preset = _load_preset(args.preset)
    models: List[Dict[str, Any]] = preset["models"]
    tag = preset.get("tag", "custom")
    if args.min_img_per_place != 4:
        tag = f"{tag}_min{args.min_img_per_place}"
    if args.max_places > 0:
        tag = f"{tag}_maxplaces{args.max_places}"

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    places_df = load_places_csv(args.places_csv, min_img_per_place=args.min_img_per_place)
    places_df = _subset_places(places_df, args.max_places)
    logger.info(
        "Loaded %d images across %d places (min_img_per_place=%d)",
        len(places_df),
        places_df["place_id"].nunique(),
        args.min_img_per_place,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else out_dir / "cache"

    named_stats = []
    for spec in models:
        slug = spec["slug"]
        cache_path = _cache_path(cache_dir, slug, tag, spec)
        stats = _run_model(spec, device, places_df, args.batch_size, args.num_workers, cache_path)
        named_stats.append((slug, stats))

    merged = merge_many_model_stats(named_stats)
    out_csv = out_dir / f"gsv_place_centroid_stats_{tag}.csv"
    merged.to_csv(out_csv, index=False)

    summary: Dict[str, Any] = {
        "tag": tag,
        "n_places": int(len(merged)),
        "n_images": int(merged["n_images"].sum()) if "n_images" in merged.columns else int(len(places_df)),
        "min_img_per_place": args.min_img_per_place,
        "headline_metrics": HEADLINE_METRICS,
        "models": [],
    }
    for spec, (slug, _) in zip(models, named_stats):
        model_summary = {
            "slug": slug,
            "label": spec.get("label", slug),
            "loader": spec["loader"],
            "dim": spec["dim"],
            "image_size": spec.get("image_size"),
            "backbone": spec.get("backbone"),
            "ckpt": spec.get("ckpt"),
        }
        model_summary.update(headline_summary(merged, slug))
        summary["models"].append(model_summary)

    summary_path = out_dir / f"gsv_place_centroid_summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    logger.info("Wrote %s (%d places)", out_csv, len(merged))
    logger.info("Wrote %s", summary_path)
    for model_summary in summary["models"]:
        slug = model_summary["slug"]
        sep = model_summary.get(f"{slug}_separation_ratio_over_places")
        mean_dist = model_summary.get(f"{slug}_mean_dist_over_places")
        logger.info("%s: mean_dist=%.4f separation_ratio=%.4f", slug, mean_dist, sep)


if __name__ == "__main__":
    main()
