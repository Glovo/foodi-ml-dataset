from pathlib import Path

import numpy as np
import torch
from params import get_test_params
from retrieval.data.loaders import get_loader
from retrieval.train import evaluation
from retrieval.utils.file_utils import (load_yaml_opts, parse_loader_name,
                                        save_json)
from retrieval.utils.logger import create_logger
from run import get_data_path, get_tokenizers, load_model

if __name__ == '__main__':
    args = get_test_params(ensemble=True)
    opt = load_yaml_opts(args.options[0])
    logger = create_logger(level='debug' if opt.engine.debug else 'info')

    logger.info(f'Used args   : \n{args}')
    logger.info(f'Used options: \n{opt}')

    train_data = opt.dataset.train_data

    data_path = get_data_path(opt)

    data_name, lang = parse_loader_name(opt.dataset.train.data)

    loader = get_loader(
        data_split=args.data_split,
        data_path=data_path,
        data_info=opt.dataset.train.data,
        loader_name=opt.dataset.loader_name,
        local_rank=args.local_rank,
        text_repr=opt.dataset.text_repr,
        vocab_paths=opt.dataset.vocab_paths,
        ngpu=torch.cuda.device_count(),
        **opt.dataset.val,
    )

    device = torch.device(args.device)
    # device = torch.device('cuda')
    tokenizers = get_tokenizers(loader)

    sims = []
    for options in args.options:
        options = load_yaml_opts(options)
        _model = load_model(f'{options.exp.outpath}/best_model.pkl')

        img_emb, cap_emb, lens = evaluation.predict_loader(_model, loader, device)
        _, sim_matrix = evaluation.evaluate(_model, img_emb, cap_emb, lens, device, return_sims=True)
        sims.append(sim_matrix)

    sims = np.array(sims)
    sims = sims.mean(0)

    i2t_metrics = evaluation.i2t(sims)
    t2i_metrics = evaluation.t2i(sims)

    _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')
    metrics = {}

    rsum = np.sum(i2t_metrics[:3]) + np.sum(t2i_metrics[:3])

    i2t_metrics = {
        f'i2t_{k}': float(v) for k, v in zip(_metrics_, i2t_metrics)
    }
    t2i_metrics = {
        f't2i_{k}': float(v) for k, v in zip(_metrics_, t2i_metrics)
    }

    metrics.update(i2t_metrics)
    metrics.update(t2i_metrics)
    metrics['rsum'] = rsum
    logger.info(metrics)

    if args.outpath is not None:
        outpath = args.outpath
    else:
        filename = f'{data_name}.{lang}:{args.data_split}:ens_results.json'
        outpath = Path(opt.exp.outpath) / filename

    logger.info(f'Saving into: {outpath}')
    save_json(outpath, metrics)
