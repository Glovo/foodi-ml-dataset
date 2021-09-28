from . import precomp
from . import fullencoder
from . import pooling
import torchvision


_image_encoders = {
    'simple': {
        'class': precomp.SimplePrecomp,
        'args': {}
    },
    'scan': {
        'class': precomp.SCANImagePrecomp,
        'args': {
            'img_dim': 2048,
        },
    },
    'vsepp_precomp': {
        'class': precomp.VSEImageEncoder,
        'args': {
            'img_dim': 2048,
        },
    },
    'full_image': {
        'class': fullencoder.ImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet152,
            'img_dim': 248,
        },
    },
    'resnet50': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet50,
            'img_dim': 2048,
        },
    },
    'resnet50_ft': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet50,
            'img_dim': 2048,
            'finetune': True,
        },
    },
    'resnet101': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet101,
            'img_dim': 2048,
            'proj_regions': False,
        },
    },
    'resnet101_ft': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet101,
            'img_dim': 2048,
            'finetune': True,
        },
    },
    'resnet152': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet152,
            'img_dim': 2048,
            'proj_regions': False,
        },
    },
    'resnet152_ft': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet152,
            'img_dim': 2048,
            'finetune': True,
        },
    },
    'vsepp_pt': {
        'class': fullencoder.VSEPPEncoder,
        'args': {
            'cnn_type': 'resnet152',
        },
    },
}


def get_available_imgenc():
    return _image_encoders.keys()


def get_image_encoder(name, **kwargs):
     model_settings = _image_encoders[name]
     model_class = model_settings['class']
     model_args = model_settings['args']
     arg_dict = dict(kwargs)
     arg_dict.update(model_args)
     model = model_class(**arg_dict)
     return model


#def get_image_encoder(name, **kwargs):
#    model_class = _image_encoders[name]['class']
#    model = model_class(**kwargs)
#    return model


def get_img_pooling(pool_name):

    _pooling = {
        'mean': pooling.mean_pooling,
        'max': pooling.max_pooling,
        'none': lambda x: x,
    }

    return _pooling[pool_name]
