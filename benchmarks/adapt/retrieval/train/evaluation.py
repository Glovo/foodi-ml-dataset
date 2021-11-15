from timeit import default_timer as dt

import numpy as np
import torch
from tqdm import tqdm

from ..utils import layers


@torch.no_grad()
def predict_loader(model, data_loader, device):
    img_embs, cap_embs, cap_lens = None, None, None
    max_n_word = 200
    model.eval()

    pbar_fn = lambda x: x
    if model.master:
        pbar_fn = lambda x: tqdm(
            x,
            total=len(x),
            desc="Pred  ",
            leave=False,
        )
    print("Evaluation begins")
    max_n_samples = 1000
    count = 0
    for batch in pbar_fn(data_loader):
        ids = batch["index"]
        if len(batch["caption"][0]) == 2:
            (_, _), (_, lengths) = batch["caption"]
        else:
            cap, lengths = batch["caption"]
        img_emb, cap_emb = model.forward_batch(batch)

        if img_embs is None:
            if len(img_emb.shape) == 3:
                is_tensor = True

                img_embs = np.zeros((max_n_samples, img_emb.size(1), img_emb.size(2)))
                cap_embs = np.zeros((max_n_samples, max_n_word, cap_emb.size(2)))
            else:
                is_tensor = False
                img_embs = np.zeros((max_n_samples, img_emb.size(1)))
                cap_embs = np.zeros((max_n_samples, cap_emb.size(1)))
            cap_lens = [0] * max_n_samples
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy()
        if is_tensor:
            cap_embs[ids, : max(lengths), :] = cap_emb.data.cpu().numpy()
        else:
            cap_embs[
                ids,
            ] = cap_emb.data.cpu().numpy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = lengths[j]
        count = count + len(ids)
        if count == max_n_samples:
            break

    # No redundancy in number of captions per image
    if img_embs.shape[0] == cap_embs.shape[0]:
        img_embs = remove_img_feat_redundancy(img_embs, data_loader)

    return img_embs, cap_embs, cap_lens


@torch.no_grad()
def predict_loader_smart(model, data_loader, device):
    img_embs, cap_embs, cap_lens = None, None, None
    max_n_word = 200
    model.eval()

    pbar_fn = lambda x: x
    if model.master:
        pbar_fn = lambda x: tqdm(
            x,
            total=len(x),
            desc="Pred  ",
            leave=False,
        )

    max_samples_eval = 20000
    count = 0
    for batch in pbar_fn(data_loader):
        ids = batch["index"]
        if len(batch["caption"][0]) == 2:
            (_, _), (_, lengths) = batch["caption"]
        else:
            cap, lengths = batch["caption"]

        img_emb, cap_emb = model.forward_batch(batch)
        if img_embs is None:
            if len(img_emb.shape) == 3:
                is_tensor = True
                img_embs = np.zeros((max_samples_eval, img_emb.size(1)), dtype=np.float)
                cap_embs = np.zeros((max_samples_eval, cap_emb.size(2)), dtype=np.float)
            else:
                is_tensor = False
                img_embs = np.zeros((max_samples_eval, img_emb.size(1)), dtype=np.float)
                cap_embs = np.zeros((max_samples_eval, cap_emb.size(1)), dtype=np.float)
            cap_lens = [0] * max_samples_eval

        # cache embeddings
        img_embs[ids] = img_emb.mean(-1).data.cpu().numpy()
        cap_emb = cap_emb.to(device)
        cap_emb = cap_emb.permute(0, 2, 1)[
            ..., :34
        ]  # To replicate behaviour of line #230 of similarity.py

        cap_emb = model.similarity.similarity.norm(cap_emb)
        for i in ids:
            img_vector = torch.from_numpy(img_embs[i]).unsqueeze(0)
            img_vector = img_vector.float()
            img_vector = img_vector.to(device)

            txt_output = model.similarity.similarity.adapt_txt(
                value=cap_emb, query=img_vector
            )
            txt_output = model.similarity.similarity.fovea(txt_output)
            txt_vector = txt_output.max(dim=-1)[0]
            cap_embs[i, :] = txt_vector.cpu().numpy()

        count = count + len(ids)
        if count == max_samples_eval:
            break

    for j, nid in enumerate(ids):
        cap_lens[nid] = lengths[j]

    return img_embs, cap_embs, cap_lens


def remove_img_feat_redundancy(img_embs, data_loader):
    return img_embs[
        np.arange(
            start=0,
            stop=img_embs.shape[0],
            step=data_loader.dataset.captions_per_image,
        ).astype(np.int)
    ]


@torch.no_grad()
def evaluate(
    model, img_emb, txt_emb, lengths, device, shared_size=128, return_sims=False
):
    model.eval()
    _metrics_ = ("r1", "r5", "r10", "medr", "meanr")

    begin_pred = dt()
    # commenting if it suffices to CPU this
    img_emb = torch.FloatTensor(img_emb).to(device)
    txt_emb = torch.FloatTensor(txt_emb).to(device)
    # img_emb = torch.FloatTensor(img_emb)
    # txt_emb = torch.FloatTensor(txt_emb)
    end_pred = dt()
    # Compute metrics with original code of ADAPT
    # sims = model.get_sim_matrix_shared(
    #    embed_a=img_emb,
    #    embed_b=txt_emb,
    #    lens=lengths
    # )
    print("Beginning get_sim_matrix_shared_eval")
    sims = model.get_sim_matrix_eval(embed_a=img_emb, embed_b=txt_emb, lens=lengths)
    # sims = layers.tensor_to_numpy(sims)
    end_sim = dt()

    i2t_metrics = i2t(sims)
    print("i2t_metrics:", i2t_metrics)
    t2i_metrics = t2i(sims)
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


def i2t_10(sims):
    # this function will return just the top 10 closest examples
    npts, ncaps = sims.shape
    captions_per_image = ncaps // npts

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        # for each image index we are ordering all closest texts
        inds = np.argsort(sims[index])[::-1]
        # Score, check where the actual caption is
        rank = 1e20
        begin = captions_per_image * index
        end = captions_per_image * index + captions_per_image
        for i in range(begin, end, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = np.round(100.0 * len(np.where(ranks < 1)[0]) / len(ranks), 2)
    r5 = np.round(100.0 * len(np.where(ranks < 5)[0]) / len(ranks), 2)
    r10 = np.round(100.0 * len(np.where(ranks < 10)[0]) / len(ranks), 2)
    medr = np.round(np.floor(np.median(ranks)) + 1, 2)
    meanr = np.round(ranks.mean() + 1, 2)

    return (r1, r5, r10, medr, meanr)


def t2i(sims):
    npts, ncaps = sims.shape
    captions_per_image = ncaps // npts

    ranks = np.zeros(captions_per_image * npts)
    top1 = np.zeros(captions_per_image * npts)

    # --> (5N(caption), N(image))
    sims = sims.T
    for index in range(npts):
        for i in range(captions_per_image):
            inds = np.argsort(sims[captions_per_image * index + i])[::-1]
            ranks[captions_per_image * index + i] = np.where(inds == index)[0][0]
            top1[captions_per_image * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.round(np.floor(np.median(ranks)) + 1, 5)
    meanr = np.round(ranks.mean() + 1, 5)

    return (r1, r5, r10, medr, meanr)


def i2t(sims):
    # trying newer implementation
    npts, ncaps = sims.shape
    captions_per_image = ncaps // npts

    ranks = np.zeros(captions_per_image * npts)
    top1 = np.zeros(captions_per_image * npts)

    # --> (5N(caption), N(image))
    # sims = sims.T
    for index in range(npts):
        for i in range(captions_per_image):
            inds = np.argsort(sims[captions_per_image * index + i])[::-1]
            ranks[captions_per_image * index + i] = np.where(inds == index)[0][0]
            top1[captions_per_image * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.round(np.floor(np.median(ranks)) + 1, 5)
    meanr = np.round(ranks.mean() + 1, 5)

    return (r1, r5, r10, medr, meanr)
