import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append('../')
from retrieval.data.tokenizer import Tokenizer
from retrieval.utils.logger import create_logger
from params import get_vocab_alignment_params


def loadEmbModel(embFile, logger):
    """Loads W2V or Glove Model"""
    logger.info("Loading Embedding Model")
    f = open(embFile,'r')
    model = {}
    v = []
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        try:
            embedding = np.array([float(val) for val in splitLine[1:]])
        except:
            logger.info(len(v), line)
        model[word] = embedding
        v.append(embedding)
    mean = np.array(v).mean(0)
    logger.info(mean.shape)
    model['<unk>'] = torch.tensor(mean)
    model['<pad>'] = torch.zeros(embedding.shape)
    model['<start>'] = torch.zeros(embedding.shape)
    model['<end>'] = torch.zeros(embedding.shape)
    logger.info("Done.",len(model)," words loaded!")
    return model


def align_vocabs(emb_model, tokenizer):
    """Align vocabulary and embedding model"""
    hi_emb = emb_model['hi']
    logger.info(hi_emb.shape)
    total_unk = 0
    nmax = max(tokenizer.vocab.idx2word.keys()) + 1
    word_matrix = torch.zeros(nmax, hi_emb.shape[-1])
    logger.info(word_matrix.shape)

    for k, v in tqdm(tokenizer.vocab.idx2word.items(), total=len(tokenizer)):
        try:
            word_matrix[k] = torch.tensor(emb_model[v])
        except KeyError:
            word_matrix[k] = emb_model['<unk>']
            total_unk += 1
    return word_matrix, total_unk


def load_tokenizer(args):
    """Load tokenizer"""
    tokenizer = Tokenizer()
    tokenizer.load(args.vocab_path)
    return tokenizer

if __name__ == '__main__':
    args = get_vocab_alignment_params()
    logger = create_logger(level='debug')

    tokenizer = load_tokenizer(args)

    emb_model = loadEmbModel(args.emb_path, logger)

    word_matrix, total_unk = align_vocabs(emb_model, tokenizer)

    logger.info(f'Finished. Total UNK: {total_unk}')
    torch.save(word_matrix, args.outpath)
    logger.info(f'Saved into: {args.outpath}')
