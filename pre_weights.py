#!/usr/bin/env python3
"""Build place image index CSV and precomputed place-centroid weights for training."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from configs.parser import load_config
from data.GSVCitiesDataloader import TRAIN_CITIES
from data.place_weights import (
    build_places_index,
    compute_place_centroids,
    load_places_csv,
    save_place_weights,
    write_places_csv,
    _subset_places_df,
)
from models.model_mode import build_model_mode
from utils.runtime import init_model

logger = logging.getLogger(__name__)


def _parse_cities(raw: str | None) -> list[str]:
    if raw is None:
        return list(TRAIN_CITIES)
    cities = [c.strip() for c in raw.split(",") if c.strip()]
    if not cities:
        raise ValueError("train_cities must list at least one city")
    return cities


def _load_encoder(cfg: dict, device: torch.device) -> torch.nn.Module:
    if cfg.get("resume_model"):
        _device, model = init_model(cfg)
        return model.to(device)
    return build_model_mode(cfg).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build place image CSV and/or precomputed place-centroid weights.",
    )
    parser.add_argument("--config", type=str, default="configs/datasets.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="mixvpr")
    parser.add_argument("--model_mode", type=str, default="basic", choices=["basic", "uncertainty"])
    parser.add_argument("--descriptors_dimension", type=int, default=512)
    parser.add_argument("--resume_model", type=str, default=None)
    parser.add_argument("--ckpt_state_dict_key", type=str, default="state_dict")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--train_cities", type=str, default=None)
    parser.add_argument("--gsv_base_path", type=str, default="/home/shared/datasets/gsv_cities/")
    parser.add_argument("--sfxl_train_root", type=str, default=None)
    parser.add_argument("--min_img_per_place", type=int, default=4)
    parser.add_argument("--csv_out", type=str, default=None, help="Output CSV (one row per image).")
    parser.add_argument("--weights_out", type=str, default=None, help="Output .pt place-centroid weights.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--csv_in", type=str, default=None, help="Reuse existing places CSV instead of rebuilding.")
    parser.add_argument("--max_places", type=int, default=0, help="Debug: keep only the first N places.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config(Path(args.config))
    cfg.update({
        "config": args.config,
        "device": args.device,
        "method": args.method,
        "model_mode": args.model_mode,
        "descriptors_dimension": args.descriptors_dimension,
        "resume_model": args.resume_model,
        "ckpt_state_dict_key": args.ckpt_state_dict_key,
        "image_size": args.image_size,
        "gsv_base_path": args.gsv_base_path,
        "sfxl_train_root": args.sfxl_train_root,
        "min_img_per_place": args.min_img_per_place,
    })

    if not args.csv_out and not args.csv_in:
        parser.error("Provide --csv_out to build the index, or --csv_in to reuse an existing CSV.")

    cities = _parse_cities(args.train_cities)
    base_path = Path(args.gsv_base_path)
    sfxl_root = Path(args.sfxl_train_root) if args.sfxl_train_root else None
    csv_out = Path(args.csv_out) if args.csv_out else Path(args.csv_in)

    if args.csv_in:
        places_df = load_places_csv(Path(args.csv_in), min_img_per_place=args.min_img_per_place)
        logger.info(
            "Loaded places CSV %s (%d images, %d places, min_img>=%d)",
            args.csv_in,
            len(places_df),
            places_df["place_id"].nunique(),
            args.min_img_per_place,
        )
    else:
        places_df = build_places_index(
            cities=cities,
            base_path=base_path,
            min_img_per_place=args.min_img_per_place,
            sfxl_train_root=sfxl_root,
        )
        write_places_csv(places_df, csv_out)

    if args.max_places > 0:
        places_df = _subset_places_df(places_df, args.max_places)
        logger.info(
            "Debug subset: %d images, %d places (max_places=%d)",
            len(places_df),
            places_df["place_id"].nunique(),
            args.max_places,
        )

    if not args.weights_out:
        return

    if args.resume_model is None:
        raise ValueError("--weights_out requires --resume_model for descriptor extraction")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = _load_encoder(cfg, device)
    place_ids, centroids, place_sums, counts, medoids, medoid_image_paths = compute_place_centroids(
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
        medoids=medoids,
        medoid_image_paths=medoid_image_paths,
        metadata={
            "csv_out": str(csv_out),
            "min_img_per_place": args.min_img_per_place,
            "descriptors_dimension": args.descriptors_dimension,
            "resume_model": args.resume_model,
            "train_cities": cities,
            "gsv_base_path": str(base_path),
            "sfxl_train_root": str(sfxl_root) if sfxl_root else None,
        },
    )


if __name__ == "__main__":
    main()
