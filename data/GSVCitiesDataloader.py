import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from data.GSVCitiesDataset import GSVCitiesDataset
from data.place_weights import (
    GSVCitiesRandomImageDataset,
    PlaceCentroidTable,
    load_places_csv,
)
from data.mixvpr_val_dataset import build_val_dataset, load_val_dataset_paths
from prettytable import PrettyTable

logger = logging.getLogger(__name__)

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

TRAIN_CITIES = [
    "Bangkok", "BuenosAires", "LosAngeles", "MexicoCity", "OSL", "Rome",
    "Barcelona", "Chicago", "Madrid", "Miami", "Phoenix", "TRT", "Boston",
    "Lisbon", "Medellin", "Minneapolis", "PRG", "WashingtonDC", "Brussels",
    "London", "Melbourne", "Osaka", "PRS",
]

SF_XL_CITIES = [
    "SF3770", "SF3771", "SF3772", "SF3773", "SF3774", "SF3775",
    "SF3776", "SF3777", "SF3778", "SF3779", "SF3780", "SF3781",
]


def _safe_dataloader_kwargs(
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
) -> Dict[str, Any]:
    """DataLoader kwargs that avoid fork-after-CUDA worker crashes."""
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "shuffle": shuffle,
        # pin_memory in workers can trigger CUDA init in child processes.
        "pin_memory": num_workers == 0,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["multiprocessing_context"] = mp.get_context("spawn")
    return kwargs


class GSVCitiesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        image_size=(320, 320),
        num_workers=4,
        show_data_stats=True,
        cities=TRAIN_CITIES,
        mean_std=IMAGENET_MEAN_STD,
        batch_sampler=None,
        random_sample_from_each_place=True,
        val_set_names=None,
        datasets_config=None,
        positive_dist_threshold=25,
        base_path=None,
        sfxl_train_root=None,
        random_images=False,
        places_csv_path=None,
        pre_weights_path=None,
        place_centroids: Optional[PlaceCentroidTable] = None,
        use_medoid_targets=False,
        medoid_live_targets=False,
        max_train_places=0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.show_data_stats = show_data_stats
        self.cities = cities
        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names or ["pitts30k_val", "msls_val"]
        self.datasets_config = datasets_config
        self.positive_dist_threshold = positive_dist_threshold
        self.base_path = base_path
        self.sfxl_train_root = sfxl_train_root
        self.random_images = bool(random_images)
        self.places_csv_path = places_csv_path
        self.pre_weights_path = pre_weights_path
        self.place_centroids = place_centroids
        self.use_medoid_targets = bool(use_medoid_targets)
        self.medoid_live_targets = bool(medoid_live_targets)
        self.max_train_places = int(max_train_places)
        self.save_hyperparameters(ignore=["place_centroids"])

        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        val_workers = max(1, self.num_workers // 2) if self.num_workers > 0 else 0
        self.train_loader_config = _safe_dataloader_kwargs(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_all,
        )
        self.valid_loader_config = _safe_dataloader_kwargs(
            batch_size=self.batch_size,
            num_workers=val_workers,
            shuffle=False,
        )

    def setup(self, stage):
        if stage == "fit":
            self.reload()
            # val_paths = load_val_dataset_paths(self.datasets_config)
            # self.val_datasets = [
            #     build_val_dataset(
            #         name,
            #         self.image_size,
            #         paths=val_paths,
            #         positive_dist_threshold=self.positive_dist_threshold,
            #     )
            #     for name in self.val_set_names
            # ]
            self.val_datasets = []
            if self.val_set_names:
                val_paths = load_val_dataset_paths(self.datasets_config)
                self.val_datasets = [
                    build_val_dataset(
                        name,
                        self.image_size,
                        paths=val_paths,
                        positive_dist_threshold=self.positive_dist_threshold,
                    )
                    for name in self.val_set_names
                ]
            if self.show_data_stats:
                self.print_stats()

    def _resolve_place_centroids(self) -> Optional[PlaceCentroidTable]:
        if self.place_centroids is not None:
            return self.place_centroids
        if not self.pre_weights_path:
            return None
        self.place_centroids = PlaceCentroidTable.from_file(
            Path(self.pre_weights_path),
            use_medoid_targets=self.use_medoid_targets,
            medoid_live_targets=self.medoid_live_targets,
        )
        return self.place_centroids

    def reload(self):
        if self.random_images:
            if not self.places_csv_path:
                raise ValueError("random_images training requires places_csv_path")
            places_df = load_places_csv(
                self.places_csv_path,
                min_img_per_place=self.min_img_per_place,
            )
            place_ids = None
            medoid_image_paths = None
            pre_weights = self._resolve_place_centroids()
            if pre_weights is not None:
                place_ids = pre_weights.place_ids.tolist()
                if self.use_medoid_targets:
                    medoid_image_paths = pre_weights.medoid_image_paths
            self.train_dataset = GSVCitiesRandomImageDataset(
                places_df,
                transform=self.train_transform,
                place_ids=place_ids,
                medoid_image_paths=medoid_image_paths,
            )
            return

        kwargs = {}
        if self.base_path is not None:
            kwargs["base_path"] = self.base_path
        if self.sfxl_train_root is not None:
            kwargs["sfxl_train_root"] = self.sfxl_train_root

        allowed_place_ids = None
        medoid_path_by_place = None
        pre_weights = self._resolve_place_centroids()
        if pre_weights is not None:
            allowed_place_ids = pre_weights.place_ids.tolist()
            if self.use_medoid_targets:
                medoid_path_by_place = pre_weights.medoid_path_by_place
                logger.info(
                    "Place-based training: restrict to %d pre_weight places "
                    "(medoid targets enabled)",
                    len(allowed_place_ids),
                )
                if self.medoid_live_targets:
                    logger.info(
                        "Medoid-live: load medoid images in DataLoader workers "
                        "(num_workers=%d)",
                        self.num_workers,
                    )

        self.train_dataset = GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
            allowed_place_ids=allowed_place_ids,
            medoid_path_by_place=medoid_path_by_place,
            medoid_transform=(
                self.valid_transform
                if self.medoid_live_targets and not self.random_images
                else None
            ),
            max_train_places=self.max_train_places,
            **kwargs,
        )

    def train_dataloader(self):
        self.reload()
        loader_cfg = dict(self.train_loader_config)
        if self.random_images:
            loader_cfg["shuffle"] = True
        return DataLoader(dataset=self.train_dataset, **loader_cfg)

    def val_dataloader(self):
        return [
            DataLoader(dataset=ds, **self.valid_loader_config)
            for ds in self.val_datasets
        ]

    def print_stats(self):
        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.align["Data"] = "l"
        table.align["Value"] = "l"
        table.header = False
        table.add_row(["# of cities", f"{len(self.cities)}"])
        if self.random_images:
            table.add_row(["# of places", f"{self.train_dataset.df['place_id'].nunique()}"])
            table.add_row(["# of images", f"{len(self.train_dataset)}"])
            table.add_row(["sampling", "random images (flat batch)"])
        else:
            table.add_row(["# of places", f"{self.train_dataset.__len__()}"])
            table.add_row(["# of images", f"{self.train_dataset.total_nb_images}"])
        print(table.get_string(title="Training Dataset"))

        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.align["Data"] = "l"
        table.align["Value"] = "l"
        table.header = False
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i + 1}", val_set_name])
        print(table.get_string(title="Validation Datasets"))

        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.align["Data"] = "l"
        table.align["Value"] = "l"
        table.header = False
        table.add_row(["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(["# of iterations", f"{self.train_dataset.__len__() // self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))
