"""Validation datasets for train_mixvpr using the repo TestDataset + datasets.json paths."""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, Union

from data.test_dataset import TestDataset
from data.dataset_paths import resolve_dataset_paths

# MixVPR val_set_name -> (datasets.json entry name, split folder key)
VAL_SET_MAP: Dict[str, Tuple[str, str]] = {
    "pitts30k_val": ("pitts30k", "validation"),
    "pitts30k_test": ("pitts30k", "test"),
    "msls_val": ("msls-val", "validation"),
    "sf_xl_val": ("sf_xl", "validation"),
}


class MixVPRValDataset:
    """Thin wrapper so MixVPR Lightning code can use TestDataset (db/queries layout)."""

    def __init__(self, test_ds: TestDataset):
        self._ds = test_ds
        self.dbStruct = SimpleNamespace(numDb=test_ds.num_database)
        self.num_references = test_ds.num_database
        self.pIdx = test_ds.get_positives()

    def __getitem__(self, index):
        return self._ds[index]

    def __len__(self):
        return len(self._ds)

    def getPositives(self):
        return self._ds.get_positives()


def load_val_dataset_paths(config_path: Optional[Union[str, Path]] = None) -> Dict:
    config_path = Path(config_path or Path(__file__).resolve().parents[1] / "configs" / "datasets.json")
    cfg = json.loads(config_path.read_text())
    needed = {name for name, _ in VAL_SET_MAP.values()}
    entries = [e for e in cfg["entries"] if e.get("name") in needed]
    missing = needed - {e["name"] for e in entries}
    if missing:
        raise ValueError(f"datasets.json missing entries for validation: {sorted(missing)}")
    paths = resolve_dataset_paths({"data_folder": cfg["data_folder"]}, entries)
    return paths


def build_val_dataset(
    val_set_name: str,
    image_size: Union[int, Tuple[int, int]],
    paths: Optional[Dict] = None,
    datasets_config: Optional[Union[str, Path]] = None,
    positive_dist_threshold: int = 25,
) -> MixVPRValDataset:
    key = val_set_name.lower()
    if key not in VAL_SET_MAP:
        raise ValueError(f"Unknown val set {val_set_name!r}; expected one of {list(VAL_SET_MAP)}")

    entry_name, split_key = VAL_SET_MAP[key]
    if paths is None:
        paths = load_val_dataset_paths(datasets_config)

    split_dir = paths[entry_name][split_key]
    database = split_dir / "database"
    queries = split_dir / "queries"
    if not queries.is_dir():
        alt = split_dir / "query"
        if alt.is_dir():
            queries = alt

    if isinstance(image_size, (tuple, list)):
        size = int(image_size[0])
    else:
        size = int(image_size)

    test_ds = TestDataset(
        str(database),
        str(queries),
        positive_dist_threshold=positive_dist_threshold,
        image_size=size,
        resize_test_imgs=True,
    )
    return MixVPRValDataset(test_ds)
