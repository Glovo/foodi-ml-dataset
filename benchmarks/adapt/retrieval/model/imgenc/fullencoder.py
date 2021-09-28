'''Neural Network Assembler and Extender'''
import types
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torchvision import models

from ...model.layers import attention, convblocks
from ...utils.layers import default_initializer
from ..similarity.measure import l1norm, l2norm
from .common import load_state_dict_with_replace


class BaseFeatures(nn.Module):

    def __init__(self, model):
        super(BaseFeatures, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.num_features = model.fc.in_features

    def forward(self, _input):
        x = self.conv1(_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class HierarchicalFeatures(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.num_features = model.fc.in_features

    def forward(self, _input):
        x = self.conv1(_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        return a, b, c, d


class Aggregate(nn.Module):

    def __init__(self, ):
        super(Aggregate, self).__init__()

    def forward(self, _input):
        x = nn.AdaptiveAvgPool2d(1)(_input)
        x = x.squeeze(2).squeeze(2)
        return x


class ImageEncoder(nn.Module):

    def __init__(
        self, cnn, img_dim,
        latent_size, pretrained=True,
    ):
        super().__init__()
        self.latent_size = latent_size

        # Full text encoder
        self.cnn = BaseFeatures(cnn(pretrained))

    def forward(self, images):
        """Extract image feature vectors."""
        if type(images) == dict:
            batch = images
            assert 'image' in batch, 'Key image not in batch dictionary'
            images = batch['image']
        # assuming that the precomputed features are already l2-normalized
        features = self.cnn(images)
        B, D, H, W = features.shape
        features = features.view(B, D, H*W)
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        new_state = load_state_dict_with_replace(
            state_dict=state_dict, own_state=self.state_dict()
        )

        super().load_state_dict(new_state)

# tutorials/09 - Image Captioning
class VSEPPEncoder(nn.Module):

    def __init__(self, latent_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(VSEPPEncoder, self).__init__()
        self.latent_size = latent_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        if finetune:
            if cnn_type.startswith('alexnet') or cnn_type.startswith('vgg'):
                model.features = nn.DataParallel(model.features)
                model.cuda()
            else:
                model = nn.DataParallel(model).cuda()

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(
                self.cnn.classifier._modules['6'].in_features,
                latent_size
            )
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1]
            )
        elif cnn_type.startswith('resnet'):
            if hasattr(self.cnn, 'module'):
                self.fc = nn.Linear(self.cnn.module.fc.in_features, latent_size)
                self.cnn.module.fc = nn.Sequential()
            else:
                self.fc = nn.Linear(self.cnn.fc.in_features, latent_size)
                self.cnn.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)
        # normalization in the image embedding space
        features = l2norm(features, dim=-1)
        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


class FullImageEncoder(nn.Module):

    def __init__(
        self, cnn, img_dim, latent_size,
        no_imgnorm=False, pretrained=True,
        proj_regions=True, finetune=False
    ):
        super(FullImageEncoder, self).__init__()
        self.latent_size = latent_size
        self.proj_regions = proj_regions
        self.no_imgnorm = no_imgnorm

        # Full text encoder
        self.cnn = BaseFeatures(cnn(pretrained))

        # # For efficient memory usage.
        # for param in self.cnn.parameters():
        #     param.requires_grad = finetune

        # Only applies pooling when region_pool is enabled
        self.region_pool = nn.AdaptiveAvgPool1d(1)
        # if proj_regions:
        #     self.region_pool = lambda x: x

        self.fc = nn.Linear(img_dim, latent_size)

        # self.apply(default_initializer)
        self.init_weights()

        # self.aggregate = Aggregate()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        import numpy as np
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""

        # assuming that the precomputed features are already l2-normalized
        features = self.cnn(images)

        features = features.view(features.shape[0], features.shape[1], -1)
        features = l2norm(features, dim=1)

        if not self.proj_regions:
            features = self.region_pool(features)

        features = features.permute(0, 2, 1)

        features = self.fc(features)

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

        super(FullImageEncoder, self).load_state_dict(new_state)

class FullHierImageEncoder(nn.Module):

    def __init__(
        self, cnn, img_dim, latent_size,
        no_imgnorm=False, pretrained=True,
        proj_regions=True,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.proj_regions = proj_regions
        self.no_imgnorm = no_imgnorm

        # Full text encoder
        self.cnn = HierarchicalFeatures(cnn(pretrained))

        # Only applies pooling when region_pool is enabled
        self.region_pool = nn.AdaptiveAvgPool1d(1)
        # if proj_regions:
        #     self.region_pool = lambda x: x

        self.fc = nn.Linear(5888, latent_size)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.apply(default_initializer)

        # self.aggregate = Aggregate()

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        a, b, c, d = self.cnn(images)
        vectors = [self.max_pool(x) for x in [a, b, c, d]]
        d = self.avg_pool(d)
        vectors = torch.cat(vectors + [d], dim=1)
        vectors = vectors.squeeze(-1)#.squeeze(-1)
        vectors = vectors.permute(0, 2, 1)
        latent = self.fc(vectors)

        return latent

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        new_state = load_state_dict_with_replace(
            state_dict=state_dict, own_state=self.state_dict()
        )

        super(FullHierImageEncoder, self).load_state_dict(new_state)
