import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from data.GSVCitiesDataset import GSVCitiesDataset
from data.mixvpr_val_dataset import build_val_dataset, load_val_dataset_paths
from prettytable import PrettyTable

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

TRAIN_CITIES = [
    "Bangkok", "BuenosAires", "LosAngeles", "MexicoCity", "OSL", "Rome",
    "Barcelona", "Chicago", "Madrid", "Miami", "Phoenix", "TRT", "Boston",
    "Lisbon", "Medellin", "Minneapolis", "PRG", "WashingtonDC", "Brussels",
    "London", "Melbourne", "Osaka", "PRS",
]


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
        self.save_hyperparameters()

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
        self.train_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": self.shuffle_all,
        }
        self.valid_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": max(1, self.num_workers // 2),
            "drop_last": False,
            "pin_memory": True,
            "shuffle": False,
        }

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

    def reload(self):
        self.train_dataset = GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
        )

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

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
        table.add_row(["# of cities", f"{len(TRAIN_CITIES)}"])
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
