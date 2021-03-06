import os
import argparse
import pandas as pd
import torch
from PIL import PngImagePlugin
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from benchmarks.wit.dataset_class import FoodiMLDataset
from benchmarks.wit.network import WIT_NN
from benchmarks.wit.trainer import train_wit_network

# set PIL to handle large images
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str,
                    help="Path of the downloaded dataset",
                    default="/mnt/data/foodi-ml/")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=160)

args = parser.parse_args()
DATASET_PATH = args.dataset_path
epochs = args.epochs
batch_size = args.batch_size

df = pd.read_parquet(os.path.join(DATASET_PATH, 'samples', 'split=train'))
df_train = df.drop_duplicates(subset=["hash", "caption"])
print(f"Number of training samples {df_train.shape[0]}")

epoch_start = 0

# define torch dataset and dataloader
ds_train = FoodiMLDataset(df_train, (224, 224))
dataloader_train = DataLoader(
    dataset=ds_train, batch_size=batch_size, drop_last=True, shuffle=True,num_workers=16
)

# model definition
model = WIT_NN(device=device)
criterion = CrossEntropyLoss(reduction="mean")

# we train attention layers and further dense layers
attention_layer_1 = [
    "0.0.auto_model.encoder.layer.11.attention.self.query.weight",
    "0.0.auto_model.encoder.layer.11.attention.self.query.bias",
    "0.0.auto_model.encoder.layer.11.attention.self.key.weight",
    "0.0.auto_model.encoder.layer.11.attention.self.key.bias",
    "0.0.auto_model.encoder.layer.11.attention.self.value.weight",
    "0.0.auto_model.encoder.layer.11.attention.self.value.bias",
    "0.0.auto_model.encoder.layer.11.attention.output.dense.weight",
    "0.0.auto_model.encoder.layer.11.attention.output.dense.bias",
    "0.0.auto_model.encoder.layer.11.attention.output.LayerNorm.weight",
    "0.0.auto_model.encoder.layer.11.attention.output.LayerNorm.bias",
]

ingestion_layer = [
    "0.0.auto_model.encoder.layer.11.intermediate.dense.weight",
    "0.0.auto_model.encoder.layer.11.intermediate.dense.bias",
    "0.0.auto_model.encoder.layer.11.output.dense.weight",
    "0.0.auto_model.encoder.layer.11.output.dense.bias",
]

dense_layers = [
    "1.linear.weight",
    "1.linear.bias",
    "0.0.auto_model.pooler.dense.weight",
    "0.0.auto_model.pooler.dense.bias",
    "0.0.auto_model.encoder.layer.11.output.LayerNorm.bias",
    "0.0.auto_model.encoder.layer.11.output.LayerNorm.weight",
]
layers_imagehead = [
    "fc.0.weight",
    "fc.0.bias",
    "layer4.2.bn3.bias",
    "layer4.2.bn3.weight",
    "layer4.2.conv3.weight",
]
attention_layer_1.extend(dense_layers)
attention_layer_1.extend(ingestion_layer)
model.language_head.change_trainable_parameters(attention_layer_1)
model.cnn.change_trainable_parameters(layers_imagehead)

final_layers_vision_params = list(map(id, model.cnn.network.fc.parameters()))
base_params = filter(
    lambda p: id(p) not in final_layers_vision_params,
    model.cnn.network.parameters(),
)
language_params = model.language_head.network.parameters()
optimizer = torch.optim.Adam(
    [
        {"params": base_params},
        {"params": model.cnn.network.fc.parameters(), "lr": 5e-4},
        {"params": language_params, "lr": 2e-5},
    ],
    lr=3e-4,
)

# cast language and text towers to device
model.language_head.network = model.language_head.network.to(device)
model.cnn.network = model.cnn.network.to(device)

# train the WIT model
model = train_wit_network(
    model, device, dataloader_train, optimizer, criterion, epochs, epoch_start=epoch_start
)