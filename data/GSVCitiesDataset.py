# https://github.com/amaralibey/gsv-cities

import logging
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from data.medoid_exclusion import exclude_medoid_training_images

logger = logging.getLogger(__name__)

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = "/home/shared/datasets/gsv_cities/" # '../datasets/gsv_cities/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

class GSVCitiesDataset(Dataset):
    def __init__(self,
                 cities=['London', 'Boston'],
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH,
                 sfxl_train_root=None,
                 allowed_place_ids=None,
                 medoid_path_by_place=None,
                 medoid_transform=None,
                 max_train_places=0,
                 ):
        super(GSVCitiesDataset, self).__init__()
        self.base_path = base_path
        self.sfxl_train_root = sfxl_train_root
        self.cities = cities

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        self.allowed_place_ids = (
            {int(pid) for pid in allowed_place_ids}
            if allowed_place_ids is not None
            else None
        )
        self.medoid_path_by_place = dict(medoid_path_by_place or {})
        self.medoid_transform = medoid_transform
        
        # Load CSVs → resolve paths → exclude medoids (before place filtering / sampling).
        self.dataframe = self.__getdataframes()
        self._has_rel_path = "sfxl_rel_path" in self.dataframe.columns
        self.dataframe = self._attach_image_paths(self.dataframe)
        self.dataframe = exclude_medoid_training_images(
            self.dataframe,
            self.medoid_path_by_place.values(),
            log=logger,
        )

        self.places_ids = pd.unique(self.dataframe.index)
        if self.allowed_place_ids is not None:
            self.places_ids = [
                pid for pid in self.places_ids if int(pid) in self.allowed_place_ids
            ]
        if max_train_places > 0:
            self.places_ids = self.places_ids[:max_train_places]
            keep = {int(pid) for pid in self.places_ids}
            self.dataframe = self.dataframe[self.dataframe.index.isin(keep)]
            logger.info("Debug: capped training to %d places", len(self.places_ids))
        self.total_nb_images = len(self.dataframe)
        
    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(Path(self.base_path) / 'Dataframes' / f'{self.cities[0]}.csv')
        df = df.sample(frac=1)  # shuffle the city dataframe
        

        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(
                Path(self.base_path) / 'Dataframes' / f'{self.cities[i]}.csv')

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            
            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')
    
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        if isinstance(place, pd.Series):
            place = place.to_frame().T

        # Medoid rows are excluded once in __init__; here we only need the path
        # (when medoid_transform is set) to load the target image for live encoding.
        medoid_path = self.medoid_path_by_place.get(int(place_id))
        if len(place) < self.img_per_place:
            raise ValueError(
                f"place_id={place_id}: only {len(place)} train images after medoid exclusion, "
                f"need img_per_place={self.img_per_place} (raise min_img_per_place)."
            )
        
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:  # always get the same most recent images
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]
            
        imgs = []
        for i, row in place.iterrows():
            img_path = self._resolve_image_path(row)
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)

        labels = torch.tensor(place_id).repeat(self.img_per_place)
        if self.medoid_transform is not None:
            if medoid_path is None:
                raise ValueError(
                    f"place_id={place_id}: medoid_transform set but no medoid path"
                )
            medoid_img = self.medoid_transform(self.image_loader(medoid_path))
            return torch.stack(imgs), labels, medoid_img

        # NOTE: contrary to image classification where __getitem__ returns only one image
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), labels

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    def _resolve_image_path(self, row):
        if self._has_rel_path and pd.notna(row.get("sfxl_rel_path")):
            if self.sfxl_train_root is None:
                raise ValueError(
                    "CSV has sfxl_rel_path but sfxl_train_root was not set")
            return str(Path(self.sfxl_train_root) / row["sfxl_rel_path"])
        img_name = self.get_img_name(row)
        return str(Path(self.base_path) / 'Images' / row['city_id'] / img_name)

    def _attach_image_paths(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        out = dataframe.copy()
        out["image_path"] = [
            self._resolve_image_path(row) for _, row in out.iterrows()
        ]
        return out

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']
        
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name
