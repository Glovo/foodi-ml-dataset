import argparse

from addict import Dict


def get_train_params():
    """Get arguments to train model"""
    parser = argparse.ArgumentParser()
    parser.add_argument('options', type=str, help='YAML path to training options')
    parser.add_argument('-local_rank', type=int, default=0)

    args = parser.parse_args()
    args = Dict(vars(args))
    return args


def get_test_params(ensemble=False):
    """Get arguments to test model"""
    parser = argparse.ArgumentParser()
    if ensemble:
        parser.add_argument('options', type=str, nargs='+', help='YAML paths to test options')
    else:
        parser.add_argument('options', type=str, help='YAML path to test options')
    parser.add_argument('-local_rank', type=int, default=0)
    parser.add_argument('-device', default='cuda', help='Device option to run test script')
    parser.add_argument('-data_split', '-s', default='dev', help='Data split to run test script')
    parser.add_argument('-outpath', '-o', default=None, help='Output file')

    args = parser.parse_args()
    args = Dict(vars(args))
    return args


def get_extractfeats_params():
    """Get arguments to test model"""
    parser = argparse.ArgumentParser()
    parser.add_argument('options', type=str, help='YAML path to test options')
    parser.add_argument('-local_rank', type=int, default=0)
    parser.add_argument('-device', default='cuda', help='Device option to run test script')
    parser.add_argument('-data_split', '-s', default='dev', help='Data split to run test script')
    parser.add_argument('-captions_path', default=None, help='Captions file (.pkl)')
    parser.add_argument('-outpath', '-o', default=None, help='Output file')

    args = parser.parse_args()
    args = Dict(vars(args))
    return args


def get_vocab_alignment_params():
    """Get arguments to test model"""
    parser = argparse.ArgumentParser()

    parser.add_argument('vocab_path', help='Vocabulary path')
    parser.add_argument('-emb_path', help='Path to glove or w2v file')
    parser.add_argument('-outpath', help='Output path')

    args = parser.parse_args()
    return args


def get_vocab_builder_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', help='Data path')
    parser.add_argument('-char_level', action='store_true', help='Define character or word level')
    parser.add_argument('-data_name', nargs='+', default=['f30k_precomp'], help='Name of dataset')
    parser.add_argument('-outpath', help='Output file name')
    args = parser.parse_args()
    return args
