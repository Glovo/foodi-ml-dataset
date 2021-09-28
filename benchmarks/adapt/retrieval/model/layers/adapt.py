import torch
import torch.nn as nn


class ADAPT(nn.Module):

    def __init__(
        self, value_size, k=None, query_size=None,
        nonlinear_proj=False, groups=1,
    ):
        '''
            value_size (int): size of the features from the value matrix
            query_size (int): size of the global query vector
            k (int, optional): only used for non-linear projection
            nonlinear_proj (bool): whether to project gamma and beta non-linearly
            groups (int): number of feature groups (default=1)
        '''
        super().__init__()

        self.query_size = query_size
        self.groups = groups

        if query_size is None:
            query_size = value_size

        if nonlinear_proj:
            self.fc_gamma = nn.Sequential(
                nn.Linear(query_size, value_size//k),
                nn.ReLU(inplace=True),
                nn.Linear(value_size//k, value_size),
            )

            self.fc_beta = nn.Sequential(
                nn.Linear(query_size, value_size//k),
                nn.ReLU(inplace=True),
                nn.Linear(value_size//k, value_size),
            )
        else:
            self.fc_gamma = nn.Sequential(
                nn.Linear(query_size, value_size//groups),
            )

            self.fc_beta = nn.Sequential(
                nn.Linear(query_size, value_size//groups),
            )

            # self.fc_gamma = nn.Linear(cond_vector_size, in_features)
            # self.fc_beta = nn.Linear(cond_vector_size, in_features)

    def forward(self, value, query):
        '''

        Adapt embedding matrix (value) given a query vector.
        Dimension order is the same of the convolutional layers.

        Arguments:
            feat_matrix {torch.FloatTensor}
                -- shape: batch, features, timesteps
            cond_vector {torch.FloatTensor}
                -- shape: ([1 or batch], features)

        Returns:
            torch.FloatTensor
                -- shape: batch, features, timesteps

        Special cases:
            When query shape is (1, features) it is performed
            one-to-many embedding adaptation. A single vector is
            used to filter all instances from the value matrix
            leveraging the brodacast of the query vector.
            This is the default option for retrieval.

            When query shape is (batch, features) it is performed
            pairwise embedding adaptation. i.e., adaptation is performed
            line by line, and value and query must be aligned.
            This could be used for VQA or other tasks that don't require
            ranking all instances from a set.

        '''

        B, D, _ = value.shape
        Bv, Dv = query.shape

        value = value.view(
            B, D//self.groups, self.groups, -1
        )

        gammas = self.fc_gamma(query).view(
            Bv, Dv//self.groups, 1, 1
        )
        betas  = self.fc_beta(query).view(
            Bv, Dv//self.groups, 1, 1
        )

        normalized = value * (gammas + 1) + betas
        normalized = normalized.view(B, D, -1)
        return normalized
