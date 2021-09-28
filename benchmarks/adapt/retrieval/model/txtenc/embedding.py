import torch
import torch.nn as nn
from ..layers import convblocks
from ..layers import attention


class PartialConcat(nn.Module):

    def __init__(
            self,
            num_embeddings,
            embed_dim=300,
            liwe_char_dim=24,
            liwe_neurons=[128, 256],
            liwe_dropout=0.0,
            liwe_wnorm=True,
            liwe_batch_norm=True,
            liwe_activation=nn.ReLU(inplace=True),
            max_chars = 26,
            **kwargs
        ):
        super(PartialConcat, self).__init__()

        weight_norm = nn.Identity
        if liwe_wnorm:
            from torch.nn.utils import weight_norm

        self.embed = nn.Embedding(num_embeddings, liwe_char_dim)

        self.total_embed_size = liwe_char_dim * max_chars

        layers = []
        liwe_neurons = liwe_neurons + [embed_dim]
        in_sizes = [liwe_char_dim * max_chars] + liwe_neurons

        batch_norm = nn.BatchNorm1d
        if not liwe_batch_norm:
            batch_norm = nn.Identity

        for n, i in zip(liwe_neurons, in_sizes):
            layer = nn.Sequential(*[
                weight_norm(
                    nn.Conv1d(i, n, 1)
                ),
                nn.Dropout(liwe_dropout),
                batch_norm(n),
                liwe_activation,
            ])
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward_embed(self, x):

        partial_words = x.view(self.B, -1) # (B, W*Ct)
        char_embed = self.embed(partial_words) # (B, W*Ct, Cw)
        # char_embed = l2norm(char_embed, 2)
        char_embed = char_embed.view(self.B, self.W, -1)
        # a, b, c = char_embed.shape
        # left = self.total_embed_size - c
        # char_embed = nn.ReplicationPad1d(left//2)(char_embed)
        char_embed = char_embed.permute(0, 2, 1)
        word_embed = self.layers(char_embed)
        return word_embed

    def forward(self, x):
        '''
            x: (batch, nb_words, nb_characters [tokens])
        '''
        self.B, self.W, self.Ct = x.size()
        return self.forward_embed(x)


class PartialConcat(nn.Module):

    def __init__(
            self,
            num_embeddings,
            embed_dim=300,
            liwe_char_dim=24,
            liwe_neurons=[128, 256],
            liwe_dropout=0.0,
            liwe_wnorm=True,
            liwe_batch_norm=True,
            liwe_activation=nn.ReLU(inplace=True),
            max_chars=26,
            **kwargs
        ):
        super(PartialConcat, self).__init__()

        if type(liwe_activation) == str:
            liwe_activation = eval(liwe_activation)

        weight_norm = nn.Identity
        if liwe_wnorm:
            from torch.nn.utils import weight_norm

        self.embed = nn.Embedding(num_embeddings, liwe_char_dim)

        self.attn = attention.SelfAttention(
            liwe_char_dim * max_chars,
            activation=liwe_activation,
            k=4,
        )

        self.total_embed_size = liwe_char_dim * max_chars

        layers = []
        liwe_neurons = liwe_neurons + [embed_dim]
        in_sizes = [liwe_char_dim * max_chars] + liwe_neurons

        batch_norm = nn.BatchNorm1d
        if not liwe_batch_norm:
            batch_norm = nn.Identity

        for n, i in zip(liwe_neurons, in_sizes):
            layer = nn.Sequential(*[
                weight_norm(
                    nn.Conv1d(i, n, 1)
                ),
                nn.Dropout(liwe_dropout),
                batch_norm(n),
                liwe_activation,
            ])
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward_embed(self, x):

        partial_words = x.view(self.B, -1) # (B, W*Ct)
        char_embed = self.embed(partial_words) # (B, W*Ct, Cw)
        # char_embed = l2norm(char_embed, 2)
        char_embed = char_embed.view(self.B, self.W, -1)
        # a, b, c = char_embed.shape
        # left = self.total_embed_size - c
        # char_embed = nn.ReplicationPad1d(left//2)(char_embed)
        char_embed = char_embed.permute(0, 2, 1)
        word_embed = self.layers(char_embed)
        return word_embed

    def forward(self, x):
        '''
            x: (batch, nb_words, nb_characters [tokens])
        '''
        self.B, self.W, self.Ct = x.size()
        return self.forward_embed(x)


class PartialGRUs(nn.Module):

    def __init__(
            self,
            num_embeddings,
            embed_dim=300,
            liwe_char_dim=24,
            **kwargs
        ):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings, liwe_char_dim)

        self.rnn = nn.GRU(liwe_char_dim, embed_dim, 1,
            batch_first=True, bidirectional=False)

    def forward_embed(self, x):

        partial_words = x.view(self.B, -1) # (B, W*Ct)
        char_embed = self.embed(partial_words) # (B, W*Ct, Cw)
        char_embed = char_embed.view(self.B*self.W, self.Ct, -1)
        x, _ = self.rnn(char_embed)

        b, t, d = x.shape
        # x = x.view(b, t, 2, d//2).mean(-2)
        x = x.max(1)[0]

        return x

    def forward(self, x):
        '''
            x: (batch, nb_words, nb_characters [tokens])
        '''
        x = x[:,:,:30]
        self.B, self.W, self.Ct = x.size()
        embed_word = self.forward_embed(x)
        embed_word = embed_word.view(self.B, self.W, -1)

        return embed_word.permute(0, 2, 1)


class PartialConcatScale(nn.Module):

    def __init__(
            self,
            num_embeddings,
            embed_dim=300,
            liwe_char_dim=24,
            liwe_neurons=[128, 256],
            liwe_dropout=0.0,
            liwe_wnorm=True,
            max_chars = 26,
            liwe_activation=nn.ReLU(),
            liwe_batch_norm=True,
        ):
        super(PartialConcatScale, self).__init__()

        if type(liwe_activation) == str:
            liwe_activation = eval(liwe_activation)

        if not liwe_wnorm:
            weight_norm = nn.Identity
        else:
            from torch.nn.utils import weight_norm

        self.embed = nn.Embedding(num_embeddings, liwe_char_dim)
        self.embed_dim = embed_dim
        self.max_chars = max_chars
        self.total_embed_size = liwe_char_dim * max_chars

        layers = []
        liwe_neurons = liwe_neurons + [embed_dim]
        in_sizes = [liwe_char_dim * max_chars] + liwe_neurons

        batch_norm = nn.BatchNorm1d
        if not liwe_batch_norm:
            batch_norm = nn.Identity

        for n, i in zip(liwe_neurons, in_sizes):
            layer = nn.Sequential(*[
                weight_norm(
                    nn.Conv1d(i, n, 1)
                ),
                nn.Dropout(liwe_dropout),
                batch_norm(n),
                liwe_activation,
            ])
            layers.append(layer)

        self.scale = nn.Parameter(torch.ones(1))

        self.layers = nn.Sequential(*layers)

    def forward_embed(self, x):
        positive_mask = (x == 0)
        words_length = positive_mask.sum(-1)
        words_length = words_length.view(self.B, -1, 1, 1)
        words_length = words_length.float()

        partial_words = x.view(self.B, -1) # (B, W*Ct)
        char_embed = self.embed(partial_words) # (B, W*Ct, Cw)

        mask = (partial_words == 0)

        char_embed[mask] = 0
        # char_embed = l2norm(char_embed, 2)
        char_embed = char_embed.view(self.B, self.W, -1)

        char_embed_scale = char_embed.view(
            self.B, self.W, self.max_chars, -1
        )

        char_embed_scale = char_embed_scale * (torch.sqrt(words_length) * self.scale)
        char_embed_scale = char_embed_scale.view(self.B, self.W, -1)

        # a, b, c = char_embed.shape
        # left = self.total_embed_size - c
        # char_embed = nn.ReplicationPad1d(left//2)(char_embed)
        char_embed_scale = char_embed_scale.permute(0, 2, 1)
        word_embed_scaled = self.layers(char_embed_scale)

        # char_embed = char_embed.permute(0, 2, 1)
        # word_embed_nonscaled = self.layers(char_embed)

        # sample = word_embed_nonscaled[0][:20]

        # print(sample.mean(-1))
        # print(sample.std(-1))
        # print('\n')
        # sample = word_embed_scaled[0][:20]
        # print(sample.mean(-1))
        # print(sample.std(-1))
        # exit()
        return word_embed_scaled

    def forward(self, x):
        '''
            x: (batch, nb_words, nb_characters [tokens])
        '''
        self.B, self.W, self.Ct = x.size()
        return self.forward_embed(x)


class PartialGRUProj(nn.Module):

    def __init__(
            self,
            num_embeddings,
            hidden_size=384,
            embed_dim=300,
            liwe_char_dim=24,
            **kwargs
        ):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings, liwe_char_dim)

        self.rnn = nn.GRU(liwe_char_dim, hidden_size, 1,
            batch_first=True, bidirectional=False)

        self.fc = nn.Linear(hidden_size, embed_dim)

    def forward_embed(self, x):

        partial_words = x.view(self.B, -1) # (B, W*Ct)
        char_embed = self.embed(partial_words) # (B, W*Ct, Cw)
        char_embed = char_embed.view(self.B*self.W, self.Ct, -1)
        x, _ = self.rnn(char_embed)

        b, t, d = x.shape
        # x = x.view(b, t, 2, d//2).mean(-2)
        x = x.max(1)[0]

        x = self.fc(x)

        return x

    def forward(self, x):
        '''
            x: (batch, nb_words, nb_characters [tokens])
        '''
        x = x[:,:,:30]
        self.B, self.W, self.Ct = x.size()
        embed_word = self.forward_embed(x)
        embed_word = embed_word.view(self.B, self.W, -1)

        return embed_word.permute(0, 2, 1)


class GloveEmb(nn.Module):

    def __init__(
            self,
            num_embeddings,
            glove_dim,
            glove_path,
            add_rand_embed=False,
            rand_dim=None,
            **kwargs
        ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.add_rand_embed = add_rand_embed
        self.glove_dim = glove_dim
        self.final_word_emb = glove_dim

        # word embedding #TODO: dev change
        self.glove = nn.Embedding(num_embeddings, glove_dim)
        #glove = nn.Parameter(torch.load(glove_path))
        #self.glove.weight = glove
        self.glove.requires_grad = True  # default to False
        
        if add_rand_embed:
            self.embed = nn.Embedding(num_embeddings, rand_dim)
            self.final_word_emb = glove_dim + rand_dim

    def get_word_embed_size(self,):
        return self.final_word_emb

    def forward(self, x):
        '''
            x: (batch, nb_words, nb_characters [tokens])
        '''
        emb = self.glove(x)
        if self.add_rand_embed:
            emb2 = self.embed(x)
            emb = torch.cat([emb, emb2], dim=2)

        return emb
