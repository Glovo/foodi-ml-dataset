from torch.utils.data import Dataset
from torchvision.transforms import transforms
import warnings
from PIL import Image
import pandas as pd
import os
import torch
import ast
import PIL

PIL.Image.LOAD_TRUNCATED_IMAGES = True # Avoid Decompressed data too large
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

class FoodiMLDataset(Dataset):
    def __init__(self, data_path, train):

        self.df = pd.read_csv(os.path.join(data_path, "foodi-ml-train.csv"))
        
        if train:
            self.df = self.df[self.df["split"]!="val"]
        else:
            self.df = self.df[self.df["split"]=="val"]
        
        self.labels = self.df["class_label"].values
        warnings.simplefilter("ignore")

    def __getitem__(self, index):
        with warnings.catch_warnings():
            img = Image.open(self.df["s3_path"].iloc[index]).convert("RGB")
        label = self.df["class_label"].iloc[index]
        return img, label

    def __len__(self):
        return self.df.shape[0]