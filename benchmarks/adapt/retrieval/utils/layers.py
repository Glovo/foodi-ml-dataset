import torch
import torch.nn as nn


def default_initializer(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)
    elif type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)
    elif type(m) == nn.Embedding:
        m.weight.data.uniform_(-0.1, 0.1)


def tensor_to_numpy(x):
    return x.data.cpu().numpy()
