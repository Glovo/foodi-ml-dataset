import torch


def mean_pooling(texts, lengths):
    out = torch.stack([t[:l].mean(0) for t, l in zip(texts, lengths)], dim=0)
    return out


def max_pooling(texts, lengths):
    out = torch.stack([t[:l].max(0)[0] for t, l in zip(texts, lengths)], dim=0)
    return out


def last_hidden_state_pool(texts, lengths):
    out = torch.stack([t[l - 1] for t, l in zip(texts, lengths)], dim=0)
    return out


def none(x, l):
    return x


# def last_hidden_state_pool(texts, lengths):
#     I = torch.LongTensor(lengths).view(-1, 1, 1)
#     I = I.expand(texts.size(0), 1, texts[0].size(1))-1

#     if torch.cuda.is_available():
#         I = I.cuda()

#     out = torch.gather(texts, 1, I).squeeze(1)
