import torch
from torch.utils.data import DataLoader

from . import datasets
from . import collate_fns
from .tokenizer import Tokenizer
from ..utils.logger import get_logger

from retrieval.utils.file_utils import load_yaml_opts, parse_loader_name


logger = get_logger()

__loaders__ = {
    'image': {
        'class': datasets.ImageDataset
    },
    'indisk_image': {
        'class': datasets.InDiskImageDataset
    },
}

def get_dataset_class(loader_name):
    loader = __loaders__[loader_name]
    return loader['class']


def prepare_ml_data(instance, device):
    targ_a, lens_a, targ_b, lens_b, ids = instance
    targ_a = targ_a.to(device).long()
    targ_b = targ_b.to(device).long()
    return targ_a, lens_a, targ_b, lens_b, ids


def get_loader(
    loader_name, data_path, data_info, data_split,
    batch_size, vocab_paths, text_repr,
    workers=4, ngpu=1, local_rank=0,
    **kwargs
):
    data_name, lang = parse_loader_name(data_info)
    if not lang:
        lang = 'en'
    logger.debug('Get loader')
    dataset_class = get_dataset_class(loader_name)
    logger.debug(f'Dataset class is {dataset_class}')

    tokenizers = []
    for vocab_path in vocab_paths:
        tokenizers.append(Tokenizer(vocab_path))
        logger.debug(f'Tokenizer built: {tokenizers[-1]}')

    dataset = dataset_class(
        data_path=data_path,
        data_name=data_name,
        data_split=data_split,
        tokenizer=tokenizers[0],
        lang=lang,
    )
    logger.debug(f'Dataset built: {dataset}')

    sampler = None
    shuffle = (data_split == 'train')
    if ngpu > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=ngpu,
            rank=local_rank,
        )
        shuffle = False

    collate = collate_fns.Collate(text_repr)

    if loader_name == 'lang' and text_repr == 'liwe':
        collate = collate_fns.collate_lang_liwe
    if loader_name == 'lang' and text_repr == 'word':
        collate = collate_fns.collate_lang_word

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate,
        num_workers=workers,
        sampler=sampler,
    )
    logger.debug(f'Loader built: {loader}')

    return loader


def get_loaders(data_path, local_rank, opt):
    train_loader = get_loader(
        data_split='train',
        data_path=data_path,
        data_info=opt.dataset.train.data,
        loader_name=opt.dataset.loader_name,
        local_rank=local_rank,
        text_repr=opt.dataset.text_repr,
        vocab_paths=opt.dataset.vocab_paths,
        ngpu=torch.cuda.device_count(),
        **opt.dataset.train
    )

    val_loaders = []
    for val_data in opt.dataset.val.data:
        val_loaders.append(
            get_loader(
                data_split='dev',
                data_path=data_path,
                data_info=val_data,
                loader_name=opt.dataset.loader_name,
                local_rank=local_rank,
                text_repr=opt.dataset.text_repr,
                vocab_paths=opt.dataset.vocab_paths,
                ngpu=1,
                **opt.dataset.val
            )
        )
    assert len(val_loaders) > 0

    adapt_loaders = []
    for adapt_data in opt.dataset.adapt.data:
        adapt_loaders.append(
            get_loader(
                data_split='train',
                data_path=data_path,
                data_info=adapt_data,
                loader_name='lang',
                local_rank=local_rank,
                text_repr=opt.dataset.text_repr,
                vocab_paths=opt.dataset.vocab_paths,
                ngpu=1,
                **opt.dataset.adapt
            )
        )
    logger.info(f'Adapt loaders: {len(adapt_loaders)}')
    return train_loader, val_loaders, adapt_loaders


# def get_loaders(
#         data_path, loader_name, data_name,
#         vocab_path, batch_size,
#         workers, text_repr,
#         splits=['train', 'val', 'test'],
#         langs=['en', 'en', 'en'],
#     ):

#     loaders = []
#     loader_class = get_dataset_class(loader_name)
#     for split, lang in zip(splits, langs):
#         logger.debug(f'Getting loader {loader_class}/  {split} / Lang {lang}')
#         loader = get_loader(
#             loader_name=loader_name,
#             data_path=data_path,
#             data_name=data_name,
#             batch_size=batch_size,
#             workers=workers,
#             text_repr=text_repr,
#             data_split=split,
#             lang=lang,
#             vocab_path=vocab_path,
#         )
#         loaders.append(loader)
#     return tuple(loaders)
