import os

import pandas as pd
from PIL import PngImagePlugin
from torch.utils.data import DataLoader

from benchmarks.wit.dataset_class import AnonymizedDataset, AnonymizedDatasetText
from benchmarks.wit.evaluator import (adapter, compute_sim_matrix,
                                      compute_valid_answers,
                                      evaluate_bigdata_new_metrics)
from benchmarks.wit.network import WIT_NN, load_saved_model

# set PIL to handle large images
LARGE_ENOUGH_NUMBER = 100
device = "cuda"
epochs_trained = 19
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

# make sure to run the code from the DATASET_NAME folder
df = pd.read_csv("spanish_subset.csv")
# rename images
root_path = "./spanish_subset/"
df["s3_path"] = df["s3_path"].apply(lambda x: os.path.join(root_path, x.split("/")[-1]))
# drop duplicates of [target,feats]

df_val = df[df.split == "val"]
answers = compute_valid_answers(df_val)
adapter = adapter(df_val)
adapter = adapter.get_adapter()

ds_val = AnonymizedDataset(df_val, (224, 224))
ds_val_textonly = AnonymizedDatasetText(df_val)
# batch size here can be increased for speed, we use here a fast divisor 509
dataloader_val = DataLoader(dataset=ds_val, batch_size=509, drop_last=True)
dataloader_val_textonly = DataLoader(
    dataset=ds_val_textonly, batch_size=509, drop_last=True
)
# load trained model
model = load_saved_model(device=device, path=f"./trained_model_{epochs_trained}.pth")
# calculate similarity matrix
sims = compute_sim_matrix(model, dataloader_val, dataloader_val_textonly)
# evaluate the model

metrics = evaluate_bigdata_new_metrics(
    model, sims, "cuda", answers, shared_size=128, return_sims=False, adapter=adapter
)
