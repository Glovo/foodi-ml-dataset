import torch
import numpy as np

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(
        dim=dim, keepdim=True
    ).sqrt() + eps
    X = torch.div(X, norm)
    return X
def l2norm_numpy(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = np.power(X, 2).sum(
        dim=dim, keepdim=True
    ).sqrt() + eps
    X = np.true_divide(X, norm)
    return X

def cosine_sim(im, s,):
    """
        Cosine similarity between all the
        image and sentence pairs
    """
    return im.mm(s.t())


def cosine_sim_numpy(im, s):
    """
        Cosine similarity between all the
        image and sentence pairs
    """
    return im.dot(s.T)

