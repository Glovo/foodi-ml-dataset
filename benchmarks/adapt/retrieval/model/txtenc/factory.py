import math

import torch
import torch.nn as nn

from . import embedding, pooling, txtenc

__text_encoders__ = {
    "gru": {
        "class": txtenc.RNNEncoder,
        "args": {
            "use_bi_gru": True,
            "rnn_type": nn.GRU,
        },
    },
    "gru_glove": {
        "class": txtenc.GloveRNNEncoder,
        "args": {},
    },
    # 'scan': {
    #     'class': txtenc.EncoderText,
    #     'args': {
    #         'use_bi_gru': True,
    #         'num_layers': 1,
    #     },
    # },
}


def get_available_txtenc():
    return __text_encoders__.keys()


def get_text_encoder(name, tokenizers, **kwargs):
    model_class = __text_encoders__[name]["class"]
    model = model_class(tokenizers=tokenizers, **kwargs)
    return model


def get_txt_pooling(pool_name):

    _pooling = {
        "mean": pooling.mean_pooling,
        "max": pooling.max_pooling,
        "lens": pooling.last_hidden_state_pool,
        "none": pooling.none,
    }

    return _pooling[pool_name]
