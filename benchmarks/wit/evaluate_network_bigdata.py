import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import PngImagePlugin
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmarks.wit.dataset_class import FoodiMLDataset
from benchmarks.wit.evaluator import adapter, compute_valid_answers
from benchmarks.wit.network import load_saved_model

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)


def generate_embeddings(model, dataloader_val, EMBEDDING_SIZE=512):
    """Generate image and text embeddings with model for the images and
    captions provided by dataloader_val.

    Parameters
    ----------
    model : WIT_NN
        Must implement the forward_embeds function.
    dataloader_val : torch.utils.data.DataLoader
        Dataloader for the validation set.

    Returns
    -------
    img_embs : torch.Tensor
        tensor of size (len(dataloader_val.dataset), EMBEDDING_SIZE) containing all image embeddings for all the dataset.

    txt_embs : torch.Tensor
        tensor of size (len(dataloader_val.dataset), EMBEDDING_SIZE) containing all text embeddings for all the dataset.
    """
    batch_size = dataloader_val.batch_size
    img_embs = torch.zeros(len(dataloader_val.dataset), EMBEDDING_SIZE)
    txt_embs = torch.zeros(len(dataloader_val.dataset), EMBEDDING_SIZE)
    assert len(len(dataloader_val.dataset)) % batch_size == 0, f"batch size similarity ({batch_size}) must be divisor of num_embeddings ({len(dataloader_val.dataset)}), here's a list of some of them below 1000 {compute_divisors(len(dataloader_val.dataset))}"
    for i, batch in tqdm(enumerate(dataloader_val)):
        img_emb, txt_emb = model.forward_embeds(batch)
        img_embs[i * batch_size: i * batch_size + batch_size,
        :] = img_emb.cpu()
        txt_embs[i * batch_size: i * batch_size + batch_size,
        :] = txt_emb.cpu()
    return img_embs, txt_embs


def init_recalls(k_list, length):
    """
    Initializes the binary arrays for each top K recalls that we want to assess
    k_list: list of the top K positions of a given set of ordered hits (i.e [1, 5, 10])
    length: number of total queries that we will make, for each query we will have a 0 or 1 in that position
        of the array, indicating if we found the query in the top hits (=1) or not (=0)
    """
    r_at_dict = {}
    for k in k_list:
        r_at_dict[k] = np.zeros(length)
    return r_at_dict


def report(task, recall_dict):
    report_dict = {}
    for k in recall_dict:
        report_dict[k] = 100.0 * np.round(
            (np.sum(recall_dict[k]) / len(recall_dict[k])), 4)
        print(f"{task}: Recall at {k}: ", np.round(report_dict[k], 2), "%")
    return report_dict


def compute_divisors(n):
    return [i for i in range(1, 1000) if n % i == 0]

def sim_matrix(a, b):
    return torch.mm(a, b.transpose(0, 1))

def compute_metrics_sequentially(im, tx, valid_answers, adapter,
                                 metric="t2i",
                                 batch_size_similarity=70):
    """Compute recall at k for

    Parameters
    ----------
    im : torch.Tensor (N, EMBEDDING_SIZE)
        Tensor containing image embeddings for all the validation set.
    tx : torch.Tensor (N, EMBEDDING_SIZE)
        Tensor containing text embeddings for all the validation set.
    valid_answers : dict
        Dictionary containing the valid answers for each item on the validation set. This dictionary is the result of running the function compute_valid_answers over the validation set. (Code of this function can be found in benchmarks/wit/evaluator.py)
    adapter:
        class handling the indexes of the valid answers of the validation dataframe.
    metric : str
        Either 't2i' or 'i2t'
    batch_size_similarity : int
        Batch size to iterate over the validation set, must be a divisor of the validation set size.

    Returns
    -------
    r_at_k : dict
        Dictionary containing the Recall at k (R@k)
            - keys (different k for recall), typically 1, 5, 10
            - Values: Recalls
    """
    assert metric == "t2i" or metric == "i2t", f"metric should be either t2i or i2t, {metric} is not recognized as a metric"

    num_embeddings = im.size()[0]
    ks = [1, 5, 10]
    r_at_k = init_recalls(ks, num_embeddings)

    assert num_embeddings % batch_size_similarity == 0, f"batch size similarity ({batch_size_similarity}) must be divisor of num_embeddings ({num_embeddings}), here's a list of some of them below 1000 {compute_divisors(num_embeddings)}"

    for i in tqdm(range(int(num_embeddings / batch_size_similarity))):
        if metric == "t2i":
            txt_i = tx[
                    i * batch_size_similarity: i * batch_size_similarity + batch_size_similarity,
                    :]  # b_s_s x 512
            sims = sim_matrix(im, txt_i)

        elif metric == "i2t":
            img_i = im[
                    i * batch_size_similarity: i * batch_size_similarity + batch_size_similarity,
                    :]
            sims = sim_matrix(tx, img_i)

        pos_matrix = i * batch_size_similarity
        sims = sims.cpu().numpy()

        inds = np.argsort(sims, axis=0)[::-1]
        for j in range(batch_size_similarity):
            image_id = adapter.image_ids[pos_matrix + j]
            for k in ks:
                intersection = np.intersect1d(inds[:k, j],
                                              valid_answers[image_id])
                r_at_k[k][pos_matrix + j] = 1 if len(intersection) > 0 else 0
    return r_at_k


if __name__ == '__main__':
    """
    This script computes the metrics for the validation set of foodi-ml-dataset. To run this script:
    python benchmarks/wit/evaluate_network_bigdata.py --dataset-path PATH_TO_DATASET_FOLDER --code-path PATH_TO_REPO_FOLDER --model-weights PATH_TO_MODEL_WEIGHTS
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str,
                        help="Path of the downloaded dataset",
                        default="../../../dataset")
    parser.add_argument("--code-path", type=str, help="Path to DATASET_NAME repository", default="../../")
    parser.add_argument("--model-weights", type=str, help="Path to the model weights", default="./trained_model_30.pth")

    args = parser.parse_args()
    DATASET_PATH = args.dataset_path
    CODE_PATH = args.code_path
    WEIGHTS_PATH = args.model_weights
    PATH_T2I_METRICS = "./t2i_metrics.pkl"
    PATH_I2T_METRICS = "./i2t_metrics.pkl"
    PATH_IMG_EMB = os.path.join(DATASET_PATH, f"img_embeddings.pt")
    PATH_TXT_EMB = os.path.join(DATASET_PATH, f"txt_embeddings.pt")
    PARQUET_PATH = os.path.join(DATASET_PATH, 'samples', 'split=val')

    df_val = pd.read_parquet(PARQUET_PATH)
    print(f"df_val shape: {df_val.shape}")
    print("Computing valid answers...")
    answers = compute_valid_answers(df_val)
    adapter = adapter(df_val)
    adapter = adapter.get_adapter()

    ds_val = FoodiMLDataset(df_val, (224, 224))

    BATCH_SIZE_VALIDATION = 70 # optimized for a machine with 1 GPU and 32 GB
    dataloader_val = DataLoader(dataset=ds_val,
                                batch_size=BATCH_SIZE_VALIDATION,
                                drop_last=True, num_workers=8)

    t2i_epochs = {}
    i2t_epochs = {}

    # Load or generate embeddings
    if not os.path.isfile(PATH_IMG_EMB):
        # load trained model
        print(f"Loading trained model:  {WEIGHTS_PATH}")
        model = load_saved_model(device=device,
                                 path=WEIGHTS_PATH)
        im, tx = generate_embeddings(model, dataloader_val)
        torch.save(im, PATH_IMG_EMB)
        torch.save(tx, PATH_TXT_EMB)
    else:
        im = torch.load(PATH_IMG_EMB)
        tx = torch.load(PATH_TXT_EMB)

    metrics = compute_metrics_sequentially(im, tx, answers, adapter,
                                           metric="t2i",
                                           batch_size_similarity=BATCH_SIZE_VALIDATION)
    t2i_epochs[WEIGHTS_PATH] = report("t2i", metrics)

    metrics = compute_metrics_sequentially(im, tx, answers, adapter,
                                           metric="i2t",
                                           batch_size_similarity=BATCH_SIZE_VALIDATION)
    i2t_epochs[WEIGHTS_PATH] = report("i2t", metrics)

    with open(PATH_T2I_METRICS, 'wb') as fh:
        pickle.dump(t2i_epochs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH_I2T_METRICS, 'wb') as fh:
        pickle.dump(i2t_epochs, fh, protocol=pickle.HIGHEST_PROTOCOL)