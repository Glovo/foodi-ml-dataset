import os
import torch
import pickle
import boto3
import numpy as np
from PIL import Image
import PIL
PIL.Image.LOAD_TRUNCATED_IMAGES = True # Otherwise we got ValueError: Decompressed data too large
from addict import Dict
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import io

from ..utils.logger import get_logger
from ..utils.file_utils import read_txt, load_pickle
from .preprocessing import get_transform

logger = get_logger()


class ImageDataset(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name,
        data_split, tokenizer, lang='en',
        resize_to=256, crop_size=224,
    ):
        from .adapters import FoodiML

        logger.debug(f'ImageDataset\n {[data_path, data_split, tokenizer, lang]}')
        self.tokenizer = tokenizer
        self.lang = lang
        self.data_split = data_split
        self.split = '.'.join([data_split, lang])
        self.data_path = Path(data_path)
        self.data_name = Path(data_name)

        self.data_wrapper = (
            FoodiML(
                self.data_path,
                data_split=data_split,
            )
        )

        self._fetch_captions()
        self.length = len(self.ids)

        self.transform = get_transform(
            data_split, resize_to=resize_to, crop_size=crop_size
        )

        self.captions_per_image = 1

        logger.debug(f'Split size: {len(self.ids)}')

    def _fetch_captions(self,):
        self.captions = []
        for image_id in self.data_wrapper.image_ids:
            self.captions.extend(
                self.data_wrapper.get_captions_by_image_id(image_id)
            )

        self.ids = range(len(self.captions))
        logger.debug(f'Loaded {len(self.captions)} captions')

    def load_img(self, image_id):

        s3_key = self.data_wrapper.get_s3_key_by_image_id(image_id)

        #TODO: change to 'glovo-products-dataset-d1c9720d'
        bucket_name = "test-bucket-glovocds"

        # Boto 3
        session = boto3.Session()
        s3_resource = session.resource('s3')
        bucket = s3_resource.Bucket(bucket_name)

        # Get image as bytes an open image as image PIL
        try:
            obj = bucket.Object(s3_key)
            file_stream = io.BytesIO()
            obj.download_fileobj(file_stream)
            pil_im = Image.open(file_stream)
            image = self.transform(pil_im)
        except OSError:
            print('Error to load image: ', s3_key)
            image = torch.zeros(3, 224, 224,)

        return image

    def __getitem__(self, index):
        print("Inside __getitem__")
        image_id = self.data_wrapper.image_ids[index]
        print("Image_id : ", image_id)
        image = self.load_img(image_id)
        print("image loaded")
        caption = self.captions[index]
        print("caption: ", caption)
        cap_tokens = self.tokenizer(caption)
        
        batch = Dict(
            image=image,
            caption=cap_tokens,
            index=index,
            img_id=image_id,
        )
        print("batch")
        return batch

    def __len__(self):
        return self.length

    def __repr__(self):
        return f'ImageDataset.{self.data_name}.{self.split}'

    def __str__(self):
        return f'{self.data_name}.{self.split}'

    
class InDiskImageDataset(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name,
        data_split, tokenizer, lang='en',
        resize_to=256, crop_size=224,
    ):
        from .adapters import FoodiML

        logger.debug(f'ImageDataset\n {[data_path, data_split, tokenizer, lang]}')
        self.tokenizer = tokenizer
        self.lang = lang
        self.data_split = data_split
        self.split = '.'.join([data_split, lang])
        self.data_path = Path(data_path)
        self.data_name = Path(data_name)
        #self.data_name = Path("dataset")
        self.full_path = self.data_path / self.data_name
        print("self._data_path: ", self.data_path)
        print(self.data_name)
        print("self.full_path: ", self.full_path)
        self.data_wrapper = (
            FoodiML(
                self.data_path,
                data_split=data_split,
            )
        )

        self._fetch_captions()
        self.length = len(self.ids)

        self.transform = get_transform(
            data_split, resize_to=resize_to, crop_size=crop_size
        )

        self.captions_per_image = 1

        logger.debug(f'Split size: {len(self.ids)}')

    def _fetch_captions(self,):
        self.captions = []
        for image_id in self.data_wrapper.image_ids:
            self.captions.extend(
                self.data_wrapper.get_captions_by_image_id(image_id)
            )

        self.ids = range(len(self.captions))
        logger.debug(f'Loaded {len(self.captions)} captions')

    def load_img(self, image_id):
        
        filename = self.data_wrapper.get_filename_by_image_id(image_id)
        #print("filename: ", filename)
        feat_path = self.full_path / filename
        try:
            image = default_loader(feat_path)
            image = self.transform(image)
        except OSError:
            print('Error to load image: ', feat_path)
            image = torch.zeros(3, 224, 224,)

        return image

    def __getitem__(self, index):
        # handle the image redundancy
        #print("getitem begins")
        seq_id = self.ids[index]
        #print("length of self.data_wrapper.image_ids: ", len(self.data_wrapper.image_ids))
        #print("seq_id: ", seq_id)
        image_id = self.data_wrapper.image_ids[seq_id]
        
        image = self.load_img(image_id)
        #print("Image loaded successfully")
        caption = self.captions[index]
        #print("caption loaded successfully")
        cap_tokens = self.tokenizer(caption)
        # What does the model do with these tokens?
        #print("Caption tokens loaded successfully")
        batch = Dict(
            image=image,
            caption=cap_tokens,
            index=index,
            img_id=image_id,
        )
        #print("image size: ", image.size())
        #return image, cap_tokens, index, image_id
        return batch

    def __len__(self):
        return self.length

    def __repr__(self):
        return f'ImageDataset.{self.data_name}.{self.split}'

    def __str__(self):
        return f'{self.data_name}.{self.split}'

    

   