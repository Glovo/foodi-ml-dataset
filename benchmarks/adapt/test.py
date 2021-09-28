import os
import sys
import torch
from tqdm import tqdm
from pathlib import Path

import params
from run import load_model, get_tokenizers
from retrieval.data.loaders import get_loader
from retrieval.model import model
from retrieval.train.train import Trainer
from retrieval.utils import file_utils, helper
from retrieval.utils.logger import create_logger
from run import load_yaml_opts, parse_loader_name, get_data_path


if __name__ == '__main__':
    args = params.get_test_params()
    opt = load_yaml_opts(args.options)
    logger = create_logger(level='debug' if opt.engine.debug else 'info')

    logger.info(f'Used args   : \n{args}')
    logger.info(f'Used options: \n{opt}')

    data_path = get_data_path(opt)

    loaders = []
    for data_info in opt.dataset.val.data:
        _, lang = parse_loader_name(data_info)
        loaders.append(
            get_loader(
                data_split=args.data_split,
                data_path=data_path,
                data_info=data_info,
                loader_name=opt.dataset.loader_name,
                local_rank=args.local_rank,
                text_repr=opt.dataset.text_repr,
                vocab_paths=opt.dataset.vocab_paths,
                ngpu=torch.cuda.device_count(),
                **opt.dataset.val
            )
        )

    tokenizers = get_tokenizers(loaders[0])
    model = helper.load_model(f'{opt.exp.outpath}/best_model.pkl')
    print_fn = (lambda x: x) if not model.master else tqdm.write

    trainer = Trainer(
        model=model,
        args={'args': args, 'model_args': opt.model},
        sysoutlog=print_fn,
    )

    result, rs = trainer.evaluate_loaders(loaders)
    logger.info(result)

    if args.outpath is not None:
        outpath = args.outpath
    else:
        filename = f'{data_info}.{lang}:{args.data_split}:results.json'
        outpath = Path(opt.exp.outpath) / filename

    logger.info('Saving into {}'.format(outpath))
    file_utils.save_json(outpath, result)
