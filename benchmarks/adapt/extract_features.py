import os

import torch
from params import get_extractfeats_params
from retrieval.data.collate_fns import default_padding
from retrieval.data.loaders import get_loader
from retrieval.model.similarity.measure import l2norm
from retrieval.utils.file_utils import load_pickle, load_yaml_opts
from retrieval.utils.logger import create_logger
from run import get_data_path, get_tokenizers, load_model
from tqdm import tqdm

if __name__ == '__main__':
    args = get_extractfeats_params()
    opt = load_yaml_opts(args.options)
    logger = create_logger(level='debug' if opt.engine.debug else 'info')

    logger.info(f'Used args   : \n{args}')
    logger.info(f'Used options: \n{opt}')

    data_path = get_data_path(opt)

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

    tokenizer = get_tokenizers(loader)

    path = args.captions_path
    outpath_file = args.outpath
    outpath_folder = os.path.dirname(args.outpath)
    file = load_pickle(path)
    model = load_model(opt, [tokenizer])
    model.eval()

    with torch.no_grad():
        outfile = {}
        for k, v in tqdm(file.items(), total=len(file)):
            tv, l = default_padding([tokenizer(x) for x in v])
            batch = {'caption': (tv, l)}
            cap = l2norm(model.embed_captions(batch).cpu(), dim=-1)

            torch.save(cap, outpath_folder / f'{k}.pkl')
            outfile[k] = cap.cpu()
        torch.save(outfile, outpath_file)
