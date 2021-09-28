import torch
import torch.nn as nn

from ..similarity.measure import l2norm
from ...utils.layers import default_initializer

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from ...model.layers import attention, convblocks
from .embedding import PartialConcat, GloveEmb, PartialConcatScale
from . import pooling

import numpy as np


# RNN Based Language Model
class GloveRNNEncoder(nn.Module):

    def __init__(
        self, tokenizers, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_type=nn.GRU, glove_path=None, add_rand_embed=False):

        super().__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

        assert len(tokenizers) == 1

        num_embeddings = len(tokenizers[0])

        self.embed = GloveEmb(
            num_embeddings,
            glove_dim=embed_dim,
            glove_path=glove_path,
            add_rand_embed=add_rand_embed,
            rand_dim=embed_dim,
        )


        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = rnn_type(
            self.embed.final_word_emb,
            latent_size, num_layers,
            batch_first=True,
            bidirectional=use_bi_gru
        )

        if hasattr(self.embed, 'embed'):
            self.embed.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, batch):
        """Handles variable size captions
        """
        captions, lengths = batch['caption']
        captions = captions.to(self.device)
        # Embed word ids to vectors
        emb = self.embed(captions)

        # Forward propagate RNN
        # self.rnn.flatten_parameters()
        cap_emb, _ = self.rnn(emb)

        if self.use_bi_gru:
            b, t, d = cap_emb.shape
            cap_emb = cap_emb.view(b, t, 2, d//2).mean(-2)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, lengths


# RNN Based Language Model
class RNNEncoder(nn.Module):

    def __init__(
        self, tokenizers, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_type=nn.GRU):

        super(RNNEncoder, self).__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

        assert len(tokenizers) == 1
        num_embeddings = len(tokenizers[0])

        # word embedding
        self.embed = nn.Embedding(num_embeddings, embed_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = rnn_type(
            embed_dim, latent_size, num_layers,
            batch_first=True,
            bidirectional=use_bi_gru
        )

        self.apply(default_initializer)

    def forward(self, batch):
        """Handles variable size captions
        """
        captions, lengths = batch['caption']
        captions = captions.to(self.device)

        # Embed word ids to vectors
        x = self.embed(captions)
        # Forward propagate RNN
        # self.rnn.flatten_parameters()
        cap_emb, _ = self.rnn(x)

        if self.use_bi_gru:
            b, t, d = cap_emb.shape
            cap_emb = cap_emb.view(b, t, 2, d//2).mean(-2)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, lengths
