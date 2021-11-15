import warnings

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class AnonymizedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size):
        self.df = df
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),  # 3 x H' x W'
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        warnings.simplefilter("ignore")
    def __getitem__(self, index):
        with warnings.catch_warnings():
            img = self.transform(Image.open(self.df["s3_path"].iloc[index]).convert("RGB"))
        caption = self.df["caption"].iloc[index]

        return {"img": img, "caption": caption}

    def __len__(self):
        return self.df.shape[0]


class AnonymizedDatasetText(Dataset):
    # variant of the dataset that returns only text
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __getitem__(self, index):
        caption = self.df["caption"].iloc[index]

        return {"caption": caption}

    def __len__(self):
        return self.df.shape[0]
