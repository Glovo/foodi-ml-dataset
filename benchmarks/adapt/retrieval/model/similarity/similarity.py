from timeit import default_timer as dt

import numpy as np
import torch
from addict import Dict
from torch import nn
from torch.nn import _VF
from torch.nn import functional as F
from tqdm import tqdm

from ...utils import helper
from ...utils.logger import get_logger
from .. import txtenc
from ..layers import attention, adapt
from ..txtenc.pooling import mean_pooling
from ..txtenc import pooling
from ..txtenc import factory
from .measure import cosine_sim, l2norm, l2norm_numpy, cosine_sim_numpy

logger = get_logger()


class Similarity(nn.Module):

    def __init__(self, device, similarity_object, **kwargs):
        super().__init__()
        self.device = device
        self.similarity = similarity_object
        # self.similarity = factory.get_similarity_object(similarity_name, device=device, **kwargs)
        logger.info(f'Created similarity: {similarity_object}')
        self.set_master_()

    def set_master_(self, is_master=True):
        self.master = is_master

    def forward(self, img_embed, cap_embed, lens, shared=False):
        logger.debug((
            f'Similarity - img_shape: {img_embed.shape} '
            'cap_shape: {cap_embed.shape}'
        ))

        return self.similarity(img_embed, cap_embed, lens)
    
    def forward_eval(self, img_embed, cap_embed, lens, shared=False):
        logger.debug((
            f'Similarity - img_shape: {img_embed.shape} '
            'cap_shape: {cap_embed.shape}'
        ))

        return self.similarity.forward_eval(img_embed, cap_embed, lens)
    

    def forward_shared(self, img_embed, cap_embed, lens, shared_size=128):
        """
        Compute pairwise i2t image-caption distance with locality sharding
        """

        #img_embed = img_embed.to(self.device)
        #cap_embed = cap_embed.to(self.device)
        n_im_shard = (len(img_embed)-1)//shared_size + 1
        n_cap_shard = (len(cap_embed)-1)//shared_size + 1
        logger.debug('Calculating shared similarities')

        pbar_fn = lambda x: range(x)
        if self.master:
            pbar_fn = lambda x: tqdm(
                range(x), total=x,
                desc='Test  ',
                leave=False,
            )

        d = torch.zeros(len(img_embed), len(cap_embed)).cpu()
        for i in pbar_fn(n_im_shard):
            im_start = shared_size*i
            im_end = min(shared_size*(i+1), len(img_embed))
            for j in range(n_cap_shard):
                cap_start = shared_size*j
                cap_end = min(shared_size*(j+1), len(cap_embed))
                im = img_embed[im_start:im_end]
                s = cap_embed[cap_start:cap_end]
                l = lens[cap_start:cap_end]
                sim = self.forward(im, s, l)
                d[im_start:im_end, cap_start:cap_end] = sim

        logger.debug('Done computing shared similarities.')
        return d
    
    def forward_shared_eval(self, img_embed, cap_embed, lens, shared_size=128):
        """
        Compute pairwise i2t image-caption distance with locality sharding
        """

        #img_embed = img_embed.to(self.device)
        #cap_embed = cap_embed.to(self.device)

        logger.debug('Calculating shared similarities')
        
        d = torch.zeros(len(img_embed), len(cap_embed)).cpu()
        cap_embed = l2norm(cap_embed, dim=-1)
        for i, img_tensor in enumerate(img_embed):
            img_vector = img_tensor.unsqueeze(0)
            img_vector = l2norm(img_vector, dim=-1)
            sim = cosine_sim(img_vector, cap_embed)
            sim = sim.squeeze(-1)
            d[i,:] = sim
        
        logger.debug('Done computing shared similarities.')
        return d

class Similarity_Ev(Similarity):
        
    def forward(self, img_embed, cap_embed, lens, shared=False):
        logger.debug((
            f'Similarity - img_shape: {img_embed.shape} '
            'cap_shape: {cap_embed.shape}'
        ))
        print(self.similarity)
        return self.similarity(img_embed, cap_embed, lens)

    def forward_shared(self, img_embed, cap_embed, lens, shared_size=128):
        """
        Compute pairwise i2t image-caption distance with locality sharding
        """

        #img_embed = img_embed.to(self.device)
        #cap_embed = cap_embed.to(self.device)

        n_im_shard = (len(img_embed)-1)//shared_size + 1
        n_cap_shard = (len(cap_embed)-1)//shared_size + 1

        logger.debug('Calculating shared similarities')

        pbar_fn = lambda x: range(x)
        if self.master:
            pbar_fn = lambda x: tqdm(
                range(x), total=x,
                desc='Test  ',
                leave=False,
            )

        d = torch.zeros(len(img_embed), len(cap_embed)).cpu()
        for i in pbar_fn(n_im_shard):
            im_start = shared_size*i
            im_end = min(shared_size*(i+1), len(img_embed))
            for j in range(n_cap_shard):
                cap_start = shared_size*j
                cap_end = min(shared_size*(j+1), len(cap_embed))
                im = img_embed[im_start:im_end]
                s = cap_embed[cap_start:cap_end]
                l = lens[cap_start:cap_end]
                sim = self.forward(im, s, l)
                d[im_start:im_end, cap_start:cap_end] = sim

        logger.debug('Done computing shared similarities.')
        return d
class Cosine(nn.Module):

    def __init__(self, device, latent_size=1024):
        super().__init__()
        self.device = device

    def forward(self, img_embed, cap_embed, *args, **kwargs):
        img_embed = img_embed.to(self.device)
        cap_embed = cap_embed.to(self.device)

        img_embed = l2norm(img_embed, dim=1)
        cap_embed = l2norm(cap_embed, dim=1)

        return cosine_sim(img_embed, cap_embed)#.cpu()


class Fovea(nn.Module):

    def __init__(self, smooth=10, train_smooth=False):
        super().__init__()

        self.smooth = smooth
        self.train_smooth = train_smooth
        self.softmax = nn.Softmax(dim=-1)

        if train_smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + self.smooth)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        mask = self.softmax(x * self.smooth)
        output = mask * x
        return output

    def __repr__(self):
        return (
            f'Fovea(smooth={self.smooth},'
            f'train_smooth: {self.train_smooth})'
        )


class Normalization(nn.Module):

    def __init__(self, latent_size, norm_method=None):
        super().__init__()
        if norm_method is None:
            self.norm = nn.Identity()
        elif norm_method == 'batchnorm':
            self.norm = nn.BatchNorm1d(latent_size, affine=False)
        elif norm_method == 'instancenorm':
            self.norm = nn.InstanceNorm1d(latent_size, affine=False)

    def forward(self, x):
        return self.norm(x)

# FIND
class AdaptiveEmbeddingT2I(nn.Module):

    def __init__(
            self, device, latent_size=1024, k=1,
            gamma=10, train_gamma=False, clip_embeddings=True,
            normalization='batchnorm', use_fovea=True
        ):
        super().__init__()

        self.device = device

        self.norm = Normalization(latent_size, normalization)

        self.adapt_img = adapt.ADAPT(latent_size, k)

        self.fovea = nn.Identity()
        if use_fovea:
            self.fovea = Fovea(smooth=gamma, train_smooth=train_gamma)

    def forward(self, img_embed, cap_embed, lens, **kwargs):
        '''
            img_embed: (B, 36, latent_size)
            cap_embed: (B, T, latent_size)
            lens (List[int]): (B)
        '''
        # (B, 1024, T)
        cap_embed = cap_embed.permute(0, 2, 1)
        img_embed = img_embed.permute(0, 2, 1)

        img_embed = self.norm(img_embed)
        # cap_embed = self.norm(cap_embed)

        sims = torch.zeros(
            img_embed.shape[0], cap_embed.shape[0]
        ).to(self.device)

        for i, cap_tensor in enumerate(cap_embed):
            # cap_tensor: 1, 1024, T
            # img_embed : B, 1024, 36
            n_words = lens[i]

            # Global textual representation
            # cap_vector: 1, 1024
            cap_repr = cap_tensor[:,:n_words].mean(-1).unsqueeze(0)

            img_output = self.adapt_img(img_embed, cap_repr)
            img_output = self.fovea(img_output)
            # Filtered global representation of the images
            img_vector = img_output.mean(-1)

            img_vector = l2norm(img_vector, dim=-1)
            cap_vector = l2norm(cap_repr, dim=-1)

            # sim = cosine_sim(img_vector, cap_vector)
            sim = cosine_sim(img_vector, cap_vector).squeeze(-1)

            # sim = sim.squeeze(-1)
            sims[:,i] = sim

        return sims

    
    
class AdaptiveEmbeddingI2T(nn.Module):

    def __init__(
            self, device, latent_size=1024, k=1,
            gamma=1, train_gamma=False,
            normalization='batchnorm', use_fovea=True
        ):
        super().__init__()

        self.device = device
        self.latent_size = latent_size

        if normalization:
            self.norm = Normalization(latent_size, normalization)

        self.adapt_txt = adapt.ADAPT(latent_size, k)

        if use_fovea:
            self.fovea = Fovea(smooth=gamma, train_smooth=train_gamma)
        else:
            self.fovea = nn.Identity()

    def forward(self, img_embed, cap_embed, lens, **kwargs):
        '''
            img_embed: (B, 36, latent_size)
            cap_embed: (B, T, latent_size)
        '''
        # (B, 1024, T)
        #
        BB, LT, KK = img_embed.shape
        cap_embed = cap_embed.permute(0, 2, 1)
        if LT != self.latent_size:
            img_embed = img_embed.permute(0, 2, 1)

        cap_embed = self.norm(cap_embed)

        sims = torch.zeros(
            img_embed.shape[0], cap_embed.shape[0]
        )
        sims = sims.to(self.device)

        # Global image representation
        img_embed = img_embed.mean(-1)

        for i, img_tensor in enumerate(img_embed):
            # cap_tensor : B, 1024, T
            # image_embed: 1, 1024

            img_vector = img_tensor.unsqueeze(0)
            txt_output = self.adapt_txt(value=cap_embed, query=img_vector)
            txt_output = self.fovea(txt_output)
            
            txt_vector = txt_output.max(dim=-1)[0]

            txt_vector = l2norm(txt_vector, dim=-1)
            img_vector = l2norm(img_vector, dim=-1)
            sim = cosine_sim(img_vector, txt_vector)
            sim = sim.squeeze(-1)
            sims[i,:] = sim

        return sims
    
    def forward_eval(self, img_embed, cap_embed, lens, **kwargs):
        '''
            img_embed: (B, 36, latent_size)
            cap_embed: (B, T, latent_size)
        '''
        # (B, 1024, T)
        #
        #BB, LT, KK = img_embed.shape
        #cap_embed = cap_embed.permute(0, 2, 1)
        #if LT != self.latent_size:
        #    img_embed = img_embed.permute(0, 2, 1)
        img_embed=img_embed.numpy()
        sims = np.zeros(
            img_embed.shape[0], cap_embed.shape[0]
        )
        #sims = sims.to(self.device)

        
        for i, img_tensor in enumerate(img_embed):
            img_vector = img_tensor.unsqueeze(0) # 1, 2048
            txt_vector = cap_embed[i].unsqueeze(0)
            txt_vector = l2norm_numpy(txt_vector, dim=-1)
            img_vector = l2norm_numpy(img_vector, dim=-1)
            sim = cosine_sim_numpy(img_vector, txt_vector)
            sim = sim.squeeze(-1)
            sims[i,:] = sim

        return sims


class AdaptiveEmbeddingI2T_eval(nn.Module):

    def __init__(
            self, device, latent_size=1024, k=1,
            gamma=1, train_gamma=False,
            normalization='batchnorm', use_fovea=True
        ):
        print('new similarity class initialised')
        super().__init__()

        self.device = device
        self.latent_size = latent_size

        if normalization:
            self.norm = Normalization(latent_size, normalization)

        self.adapt_txt = adapt.ADAPT(latent_size, k)

        if use_fovea:
            self.fovea = Fovea(smooth=gamma, train_smooth=train_gamma)
        else:
            self.fovea = nn.Identity()

    def forward(self, img_embed, cap_embed, lens, **kwargs):
        '''
            img_embed: (B, 36, latent_size)
            cap_embed: (B, T, latent_size)
        '''
        # (B, 1024, T)
        #
        BB, LT, KK = img_embed.shape
        cap_embed = cap_embed.permute(0, 2, 1)
        if LT != self.latent_size:
            img_embed = img_embed.permute(0, 2, 1)

        cap_embed = self.norm(cap_embed)

        sims = torch.zeros(
            img_embed.shape[0], cap_embed.shape[0]
        )
        sims = sims.to(self.device)

        # Global image representation
        img_embed = img_embed.mean(-1)

        for i, img_tensor in enumerate(img_embed):
            # cap_tensor : B, 1024, T
            # image_embed: 1, 1024

            img_vector = img_tensor.unsqueeze(0)
            txt_output = self.adapt_txt(value=cap_embed, query=img_vector)
            txt_output = self.fovea(txt_output)

            txt_vector = txt_output.max(dim=-1)[0]

            txt_vector = l2norm(txt_vector, dim=-1)
            img_vector = l2norm(img_vector, dim=-1)
            sim = cosine_sim(img_vector, txt_vector)
            sim = sim.squeeze(-1)
            sims[i,:] = sim

        return sims


class LogSumExp(nn.Module):
    def __init__(self, lambda_lse):
        self.lambda_lse = lambda_lse

    def forward(self, x):
        x.mul_(self.lambda_lse).exp_()
        x = x.sum(dim=1, keepdim=True)
        x = torch.log(x)/self.lambda_lse
        return x


class ClippedL2Norm(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return l2norm(self.leaky(x), 2)


class StackedAttention(nn.Module):

    def __init__(
        self, i2t=True, agg_function='Mean',
        feature_norm='softmax', lambda_lse=None,
        smooth=4, **kwargs,
    ):
        super().__init__()
        self.i2t = i2t
        self.lambda_lse = lambda_lse
        self.agg_function = agg_function
        self.feature_norm = feature_norm
        self.lambda_lse = lambda_lse
        self.smooth = smooth
        self.kwargs = kwargs

        self.attention = Attention(
            smooth=smooth, feature_norm=feature_norm,
        )

        if agg_function == 'LogSumExp':
            self.aggregate_function = LogSumExp(lambda_lse)
        elif agg_function == 'Max':
            self.aggregate_function = lambda x: x.max(dim=1, keepdim=True)[0]
        elif agg_function == 'Sum':
            self.aggregate_function = lambda x: x.sum(dim=1, keepdim=True)
        elif agg_function == 'Mean':
            self.aggregate_function = lambda x: x.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(agg_function))

        self.task = 'i2t' if i2t else 't2i'

    def forward(self, images, captions, cap_lens, ):
        """
        Images: (n_image, n_regions, d) matrix of images
        Captions: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = []
        n_image = images.size(0)
        n_caption = captions.size(0)

        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            """
                word(query): (n_image, n_word, d)
                image(context): (n_image, n_regions, d)
                weiContext: (n_image, n_word, d) or (n_image, n_region, d)
                attn: (n_image, n_region, n_word)
            """
            emb_a = cap_i_expand
            emb_b = images
            if self.i2t:
                emb_a = images
                emb_b = cap_i_expand

            weiContext, attn = self.attention(emb_a, emb_b)
            emb_a = emb_a.contiguous()
            weiContext = weiContext.contiguous()
            # (n_image, n_word)
            row_sim = cosine_similarity(emb_a, weiContext, dim=2)
            row_sim = self.aggregate_function(row_sim)
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)

        return similarities

    def __repr__(self, ):
        return (
            f'StackedAttention(task: {self.task},'
            f'i2t: {self.i2t}, '
            f'attention: {self.attention}, '
            f'lambda_lse: {self.lambda_lse}, '
            f'agg_function: {self.agg_function}, '
            f'feature_norm: {self.feature_norm}, '
            f'lambda_lse: {self.lambda_lse}, '
            f'smooth: {self.smooth}, '
            f'kwargs: {self.kwargs})'
        )



def attn_softmax(attn):
    batch_size, sourceL, queryL = attn.shape
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)
    # --> (batch, sourceL, queryL)
    attn = attn.view(batch_size, sourceL, queryL)
    return attn


class Attention(nn.Module):

    def __init__(self, smooth, feature_norm='softmax'):
        super().__init__()
        self.smooth = smooth
        self.feature_norm = feature_norm

        if feature_norm == "softmax":
            self.normalize_attn = attn_softmax
        # elif feature_norm == "l2norm":
        #     attn = lambda x: l2norm(x, 2)
        elif feature_norm == "clipped_l2norm":
            self.normalize_attn = ClippedL2Norm()
        # elif feature_norm == "l1norm":
        #     attn = l1norm_d(attn, 2)
        # elif feature_norm == "clipped_l1norm":
        #     attn = nn.LeakyReLU(0.1)(attn)
        #     attn = l1norm_d(attn, 2)
        elif feature_norm == "clipped":
            self.normalize_attn = lambda x: nn.LeakyReLU(0.1)(x)
        elif feature_norm == "no_norm":
            self.normalize_attn = lambda x: x
        else:
            raise ValueError("unknown first norm type:", feature_norm)

    def forward(self, query, context, ):
        batch_size_q, queryL = query.size(0), query.size(1)
        batch_size, sourceL = context.size(0), context.size(1)

         # Get attention
        # --> (batch, d, queryL)
        queryT = torch.transpose(query, 1, 2)

        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        attn = torch.bmm(context, queryT)
        attn = self.normalize_attn(attn)
        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size*queryL, sourceL)
        attn = nn.Softmax(dim=-1)(attn*self.smooth)
        # --> (batch, queryL, sourceL)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> (batch, sourceL, queryL)
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # --> (batch, d, sourceL)
        contextT = torch.transpose(context, 1, 2)
        # (batch x d x sourceL)(batch x sourceL x queryL)
        # --> (batch, d, queryL)
        weightedContext = torch.bmm(contextT, attnT)
        # --> (batch, queryL, d)
        weightedContext = torch.transpose(weightedContext, 1, 2)

        return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
