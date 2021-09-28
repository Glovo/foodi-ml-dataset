from collections import OrderedDict

import torch
import torch.nn as nn

from ...utils.layers import default_initializer
from ..similarity.measure import l1norm, l2norm
from ..layers import attention, convblocks

import numpy as np


def load_state_dict_with_replace(state_dict, own_state):
    new_state = OrderedDict()
    for name, param in state_dict.items():
        if name in own_state:
            new_state[name] = param
    return new_state


class SCANImagePrecomp(nn.Module):

    def __init__(self, img_dim, latent_size, no_imgnorm=False, ):
        super(SCANImagePrecomp, self).__init__()
        self.latent_size = latent_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, latent_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, batch):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        images = batch['image'].to(self.device)
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        new_state = load_state_dict_with_replace(
            state_dict=state_dict, own_state=self.state_dict()
        )

        super(SCANImagePrecomp, self).load_state_dict(new_state)


class SimplePrecomp(nn.Module):

    def __init__(self, img_dim, latent_size, no_imgnorm=False, ):
        super(SimplePrecomp, self).__init__()
        self.latent_size = latent_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, latent_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, batch):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        images = batch['image'].to(self.device)
        features = self.fc(images)
        features = nn.LeakyReLU(0.1)(features)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        new_state = load_state_dict_with_replace(
            state_dict=state_dict, own_state=self.state_dict()
        )

        super(SCANImagePrecomp, self).load_state_dict(new_state)


class VSEImageEncoder(nn.Module):

    def __init__(self, img_dim, latent_size, no_imgnorm=False, device=None):
        super(VSEImageEncoder, self).__init__()
        self.device = device
        self.latent_size = latent_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, latent_size)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.apply(default_initializer)

    def forward(self, batch):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        images = batch['image'].to(self.device).to(self.device)

        images = self.pool(images.permute(0, 2, 1)) # Global pooling
        images = images.permute(0, 2, 1)
        features = self.fc(images)
        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        new_state = load_state_dict_with_replace(
            state_dict=state_dict, own_state=self.state_dict()
        )

        super(VSEImageEncoder, self).load_state_dict(new_state)

