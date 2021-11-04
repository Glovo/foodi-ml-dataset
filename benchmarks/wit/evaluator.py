from collections import defaultdict
from timeit import default_timer as dt

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


@torch.no_grad()
def compute_sim_matrix(model, dataloader_val, dataloader_val_textonly=None):
    if dataloader_val_textonly is None:
        dataloader_val_textonly = dataloader_val
    sim_matrix = np.zeros([len(dataloader_val.dataset), len(dataloader_val.dataset)])
    sim_matrix.fill(np.nan)
    batch_len1 = dataloader_val.batch_size
    batch_len2 = dataloader_val_textonly.batch_size
    for idx1, batch1 in tqdm(enumerate(dataloader_val)):
        composed_batch = batch1
        start_r = idx1 * batch_len1
        end_r = start_r + batch_len1
        for idx2, batch2 in enumerate(dataloader_val_textonly):
            # take images from batch1 and text from batch2 and calculate similarities
            start_rc = idx2 * batch_len2
            end_rc = start_rc + batch_len2
            composed_batch["caption"] = batch2["caption"]
            sims = model.forward(composed_batch).cpu().detach().numpy().T
            sim_matrix[start_r:end_r, start_rc:end_rc] = sims
    # fill remaining nan with 0
    sim_matrix = np.nan_to_num(sim_matrix)
    return sim_matrix


def compute_valid_answers(data: pd.DataFrame):
    """Generates the valid answers taking into account mutiple images and captions.
    For the following dataset we will create the dictionary with valid_answers:
    id    caption    hash    |    valid_answers
    0     ABC        X       |    0,1,2,4
    1     EFG        X       |    0,1,4
    2     ABC        Y       |    0,2
    3     HIJ        Z       |    3,
    4     KLM        X       |    0,1,4
    """
    valid_answers = {}
    print("Computing valid answers: ")
    for i in range(data.shape[0]):
        idxs_where_duplication = (data["caption"] == data["caption"].iloc[i]) | (
            data["s3_path"] == data["s3_path"].iloc[i]
        )
        list_indexes_duplication = list(
            np.where(np.array(idxs_where_duplication.to_list()) == True)[0]
        )
        valid_answers[data["img_id"].iloc[i]] = list_indexes_duplication
    return valid_answers


class adapter:
    def __init__(self, dataframe):
        self.data = dataframe

    def get_adapter(self):
        # for a given dataframe this adapter gives the image_id given an index of the dataframe
        # for consistency the adapter will be a dictionary with a key image_ids, that will be a list
        # that returns at index i the image id corresponding to it
        image_ids = list(self.data["img_id"].values)
        img_dict = {}
        annotations = defaultdict(list)
        for _, row in self.data.iterrows():
            img_dict[row["img_id"]] = row["s3_path"]
            annotations[row["img_id"]].extend([row["caption"]])
        self.image_ids = image_ids
        self.img_dict = img_dict
        self.annotations = annotations
        return self


@torch.no_grad()
def evaluate_bigdata_new_metrics(
    model, sims, device, valid_answers, shared_size=128, return_sims=False, adapter=None
):
    model.eval()
    _metrics_ = ("r1", "r5", "r10", "medr", "meanr")

    begin_pred = dt()
    end_pred = dt()

    # sims = sims.cpu().numpy()
    end_sim = dt()
    _metrics_ = ("r1", "r5", "r10", "medr", "meanr")
    i2t_metrics = i2t_duplicated_idxs(sims, valid_answers, adapter)
    print("i2t_metrics:", i2t_metrics)
    t2i_metrics = t2i_duplicated_idxs(sims, valid_answers, adapter)
    print("t2i_metrics:", t2i_metrics)
    rsum = np.sum(i2t_metrics[:3]) + np.sum(t2i_metrics[:3])

    i2t_metrics = {f"i2t_{k}": v for k, v in zip(_metrics_, i2t_metrics)}
    t2i_metrics = {f"t2i_{k}": v for k, v in zip(_metrics_, t2i_metrics)}

    metrics = {
        "pred_time": end_pred - begin_pred,
        "sim_time": end_sim - end_pred,
    }
    metrics.update(i2t_metrics)
    metrics.update(t2i_metrics)
    metrics["rsum"] = rsum

    if return_sims:
        return metrics, sims

    return metrics


def t2i_duplicated_idxs(sims, valid_answers_imgs: dict, adapter):
    npts, ncaps = sims.shape
    captions_per_image = ncaps // npts

    r_at = {
        1: np.zeros(captions_per_image * npts),
        5: np.zeros(captions_per_image * npts),
        10: np.zeros(captions_per_image * npts),
    }

    sims = sims.T
    for index in range(npts):
        image_id = adapter.image_ids[index]
        inds = np.argsort(sims[captions_per_image * index])[::-1]
        for k in r_at.keys():  # 1,5,10
            intersection = np.intersect1d(inds[:k], valid_answers_imgs[image_id])
            r_at[k][captions_per_image * index] = 1 if len(intersection) > 0 else 0
    r1 = 100.0 * (np.sum(r_at[1]) / len(r_at[1]))
    r5 = 100.0 * (np.sum(r_at[5]) / len(r_at[5]))
    r10 = 100.0 * (np.sum(r_at[10]) / len(r_at[10]))
    return r1, r5, r10


def i2t_duplicated_idxs(sims, valid_answers_caps: dict, adapter):
    npts, ncaps = sims.shape
    captions_per_image = ncaps // npts

    r_at = {
        1: np.zeros(captions_per_image * npts),
        5: np.zeros(captions_per_image * npts),
        10: np.zeros(captions_per_image * npts),
    }

    for index in range(npts):
        # the adapter serves to map the indexes in the similarity matrix to the image ids.
        image_id = adapter.image_ids[index]
        inds = np.argsort(sims[captions_per_image * index])[::-1]
        for k in r_at.keys():  # 1,5,10
            intersection = np.intersect1d(inds[:k], valid_answers_caps[image_id])
            r_at[k][captions_per_image * index] = 1 if len(intersection) > 0 else 0
    r1 = 100.0 * (np.sum(r_at[1]) / len(r_at[1]))
    r5 = 100.0 * (np.sum(r_at[5]) / len(r_at[5]))
    r10 = 100.0 * (np.sum(r_at[10]) / len(r_at[10]))
    return r1, r5, r10
