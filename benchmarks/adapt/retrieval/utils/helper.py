import os
import torch
from tensorboardX import SummaryWriter

from ..utils.logger import get_logger

logger = get_logger()

def save_checkpoint(
    outpath, model, optimizer=None,
    is_best=False, save_all=False, **kwargs
):

    if hasattr(model, 'module'):
        model = model.module

    state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    state_dict.update(**kwargs)

    if not save_all:
        epoch = -1

    torch.save(
        obj=state_dict,
        f=os.path.join(outpath, f'checkpoint_eval_{epoch}.pkl'),
    )

    if is_best:
        import shutil
        shutil.copy(
            os.path.join(outpath, f'checkpoint_eval_{epoch}.pkl'),
            os.path.join(outpath, 'best_model_eval.pkl'),
        )

def save_checkpoint_foodi(
        outpath, model, optimizer=None,
        is_best=False, epoch=1):

    if hasattr(model, 'module'):
        model = model.module

    state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    # state_dict.update(**kwargs)

    torch.save(
        obj=state_dict,
        f=os.path.join(outpath, f'check_foodi_{epoch}.pkl'),
    )

    if is_best:
        import shutil
        shutil.copy(
            os.path.join(outpath, f'check_foodi_{epoch}.pkl'),
            os.path.join(outpath, 'best_model_foodi.pkl'),
        )

def load_model(path):

    from .. import model
    from addict import Dict
    from ..data.tokenizer import Tokenizer

    checkpoint = torch.load(
        path,  map_location=lambda storage, loc: storage
    )
    vocab_paths = checkpoint['args']['dataset']['vocab_paths']
    tokenizers = [Tokenizer(vocab_path=x) for x in vocab_paths]

    model_params = Dict(**checkpoint['args']['model'])
    model = model.Retrieval(**model_params, tokenizers=tokenizers)
    model.load_state_dict(checkpoint['model'])

    return model

def restore_checkpoint(path, model=None, optimizer=False):
    state_dict = torch.load(
        path,  map_location=lambda storage, loc: storage
    )
    new_state = {}
    for k, v in state_dict['model'].items():
        new_state[k.replace('module.', '')] = v

    if model is None:
        from .. import model
        model_params = state_dict['args']['model_args']
        model = model.Retrieval(**model_params)

    model.load_state_dict(new_state)
    state_dict['model'] = model

    if optimizer:
        optimizer = state_dict['optimizer']
        state_dict['optimizer'] = None

    return state_dict


def get_tb_writer(logger_path):
    if logger_path == 'runs/':
        tb_writer = SummaryWriter()
        logger_path = tb_writer.file_writer.get_logdir()
    else:
        tb_writer = SummaryWriter(logger_path)
    return tb_writer


def get_device(gpu_id):
    if gpu_id >= 0:
        return torch.device('cuda:{}'.format(gpu_id))
    return torch.device('cpu')


def reset_pbar(pbar):
    from time import time
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.start_t = time()
    pbar.last_print_t = time()
    pbar.update()
    return pbar


def print_tensor_dict(tensor_dict, print_fn):
    line = []
    for k, v in sorted(tensor_dict.items()):
        try:
            v = v.item()
        except AttributeError:
            pass
        line.append(f'{k.title()}: {v:10.6f}')
    print_fn(', '.join(line))


def set_tensorboard_logger(path):
    if path is not None:
        if os.path.exists(path):
            a = input(f'{path} already exists! Do you want to rewrite it? [y/n] ')
            if a.lower() == 'y':
                import shutil
                shutil.rmtree(path)
                tb_writer = get_tb_writer(path)
            else:
                exit()
        else:
            tb_writer = get_tb_writer(path)
    else:
        tb_writer = get_tb_writer()
    return tb_writer
