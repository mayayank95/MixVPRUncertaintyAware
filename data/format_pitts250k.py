#!/usr/bin/env python3
"""Materialize Pitts250k splits the same way Pitts30k was formatted.

Reads ``pitts250k_{split}.mat`` once and creates

    {out}/images/{split}/database
    {out}/images/{split}/queries

Each image is copied to a UTM filename, so ``TestDataset`` can load the split
with no Pitts250k-specific eval path. The copy does not stay linked to the
raw tree, including when that tree lives under the Pitts30k download.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from pathlib import Path

import utm
from scipy.io import loadmat

logger = logging.getLogger(__name__)

_PITCH_YAW = re.compile(r"pitch(\d+)_yaw(\d+)\.")
_NOTE = re.compile(r"_(.+)\.jpg$")


def _format_coord(num: float, left: int, right: int) -> str:
    return f"{float(num):0={left + right + 1}.{right}f}"


def dst_image_name(utm_east: float, utm_north: float, src_name: str) -> str:
    """Same filename as ``format_pitts30k.py`` / ``util.get_dst_image_name``."""
    latitude, longitude = utm.to_latlon(utm_east, utm_north, 17, "T")
    easting, northing, zone_number, zone_letter = utm.from_latlon(float(latitude), float(longitude))
    pitch = int(_PITCH_YAW.findall(src_name)[0][0]) - 1
    yaw = int(_PITCH_YAW.findall(src_name)[0][1]) - 1
    note = _NOTE.findall(src_name)[0]
    tile_num = pitch * 24 + yaw
    pano_id = src_name.split("_", 1)[0]
    return (
        f"@{_format_coord(easting, 7, 2)}"
        f"@{_format_coord(northing, 7, 2)}"
        f"@{zone_number:02d}@{zone_letter}"
        f"@{_format_coord(latitude, 3, 5)}"
        f"@{_format_coord(longitude, 4, 5)}"
        f"@{pano_id}@{tile_num:02d}@@@@@@{note}@.jpg"
    )


def _legacy_image_name(utm_east: float, utm_north: float, src_name: str) -> str:
    """Name used by the first copy, before the Pitts30k lat/lon round-trip."""
    pitch, yaw = _PITCH_YAW.findall(src_name)[0]
    tile_num = (int(pitch) - 1) * 24 + (int(yaw) - 1)
    note = _NOTE.findall(src_name)[0]
    pano_id = src_name.split("_", 1)[0]
    return (
        f"@{_format_coord(utm_east, 7, 2)}"
        f"@{_format_coord(utm_north, 7, 2)}"
        f"@17@T@000.00000@-000.00000"
        f"@{pano_id}@{tile_num:02d}@@@@@@{note}@.jpg"
    )


def _copy_image(src: Path, dst: Path, legacy: Path | None = None) -> None:
    """Copy ``src`` onto ``dst``. Rename a previous zero-lat/lon file in place."""
    if legacy is not None and legacy.is_file() and not legacy.is_symlink() and legacy != dst:
        if dst.is_file() and not dst.is_symlink():
            legacy.unlink()
        else:
            legacy.rename(dst)
    if dst.is_file() and not dst.is_symlink():
        return
    if not src.is_file():
        raise FileNotFoundError(f"Missing Pitts250k image: {src}")
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _copy_split(raw_root: Path, dst_folder: Path, rel_paths: list[str], utms) -> None:
    dst_folder.mkdir(parents=True, exist_ok=True)
    total = len(rel_paths)
    for i, (rel, (utm_east, utm_north)) in enumerate(zip(rel_paths, utms), start=1):
        src_name = Path(rel).name
        east, north = float(utm_east), float(utm_north)
        dst = dst_folder / dst_image_name(east, north, src_name)
        legacy = dst_folder / _legacy_image_name(east, north, src_name)
        _copy_image(raw_root / rel, dst, legacy)
        if i % 10000 == 0 or i == total:
            logger.info("%s %d/%d", dst_folder.name, i, total)


def format_split(raw_root: Path, out_root: Path, split: str) -> None:
    mat_path = raw_root / "datasets" / f"pitts250k_{split}.mat"
    if not mat_path.is_file():
        raise FileNotFoundError(f"Missing {mat_path}")
    mat = loadmat(str(mat_path))["dbStruct"].item()
    db_rel = [f[0].item() for f in mat[1]]
    q_rel = [f"queries_real/{f[0].item()}" for f in mat[3]]
    images = out_root / "images" / split
    logger.info("%s: %d database, %d queries", split, len(db_rel), len(q_rel))
    _copy_split(raw_root, images / "database", db_rel, mat[2].T)
    _copy_split(raw_root, images / "queries", q_rel, mat[4].T)
    logger.info("Wrote %s", images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/shared/datasets/pitts30k/orig"),
        help="Raw Pittsburgh root with 000/…, queries_real/, and datasets/*.mat.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/shared/datasets/pitts250k"),
        help="Destination dataset root. Splits land in {out}/images/{split}/.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated mats to materialize: train,val,test.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        format_split(args.raw_root, args.out, split)


if __name__ == "__main__":
    main()
