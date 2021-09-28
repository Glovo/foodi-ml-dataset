import json
import logging
from collections import Counter

import torch
from tqdm import tqdm

import nltk

from ..utils.logger import get_logger
from ..utils.file_utils import read_txt


logger = get_logger()

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_word(self, idx):
        if idx in self.idx2word:
            return self.idx2word[idx]
        else:
            return '<unk>'

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class Tokenizer(object):
    """
    This class converts texts into character or word-level tokens
    """

    def __init__(
        self, vocab_path=None, char_level=False,
        maxlen=None, download_tokenizer=False
    ):
        # Create a vocab wrapper and add some special tokens.
        self.char_level = char_level
        self.maxlen = maxlen

        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<unk>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')

        if char_level:
            vocab.add_word(' ') # Space is allways #2

        self.vocab = vocab

        if download_tokenizer:
            nltk.download('punkt')

        if vocab_path is not None:
            self.load(vocab_path)

        logger.info(f'Loaded from {vocab_path}.')
        logger.info(f'Created tokenizer with init {len(self.vocab)} tokens.')

    def fit_on_files(self, txt_files):
        logger.debug('Fit on files.')
        for file in txt_files:
            logger.info(f'Updating vocab with {file}')
            sentences = read_txt(file)
            self.fit(sentences)

    def fit(self, sentences, threshold=4):
        logger.debug(
            f'Fit word vocab on {len(sentences)} and t={threshold}'
        )
        counter = Counter()

        for tokens in tqdm(sentences, total=len(sentences)):
            if not self.char_level:
                tokens = self.split_sentence(tokens)
            counter.update(tokens)

        # Discard if the occurrence of the word is less than threshold
        tokens = [
            token for token, cnt in counter.items()
            if cnt >= threshold
        ]

        # Add words to the vocabulary.
        for token in tokens:
            self.vocab.add_word(token)

        logger.info(f'Vocab built. Tokens found {len(self.vocab)}')
        return self.vocab

    def save(self, outpath):
        logger.debug(f'Saving vocab to {outpath}')

        state = {
            'word2idx': self.vocab.word2idx,
            'char_level': self.char_level,
            'max_len': self.maxlen
        }

        with open(outpath, "w") as f:
            json.dump(state, f)

        logger.info(
            f'Vocab stored into {outpath} with {len(self.vocab)} tokens.'
        )

    def load(self, path):
        logger.debug(f'Loading vocab from {path}')
        with open(path) as f:
            state = json.load(f)

        vocab = Vocabulary()
        vocab.word2idx = state['word2idx']
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        vocab.idx = max(vocab.idx2word)
        self.vocab = vocab
        self.char_level = state['char_level']
        self.maxlen = state['max_len']
        logger.info(f'Loaded vocab containing {len(self.vocab)} tokens')
        return self

    def split_sentence(self, sentence):
        tokens = nltk.tokenize.word_tokenize(
            sentence.lower()
        )
        return tokens

    def tokens_to_int(self, tokens):
        return [self.vocab(token) for token in tokens]

    def tokenize(self, sentence):
        tokens = self.split_sentence(sentence)
        if self.char_level:
            tokens = ' '.join(tokens)
        tokens = (
            [self.vocab('<start>')]
            + self.tokens_to_int(tokens)
            + [self.vocab('<end>')]
        )
        return torch.LongTensor(tokens)

    def decode_tokens(self, tokens):
        logger.debug(f'Decode tokens {tokens}')
        join_char = '' if self.char_level else ' '
        text = join_char.join([
            self.vocab.get_word(token) for token in tokens
        ])
        return text

    def __len__(self):
        return len(self.vocab)

    def __call__(self, sentence):
        return self.tokenize(sentence)
