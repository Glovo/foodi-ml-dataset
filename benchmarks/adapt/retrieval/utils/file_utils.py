import os
import copy
import yaml
import json
import pickle
from yaml import Dumper
from addict import Dict
import pandas as pd


# FoodiML - Load parquet samples
def load_samples(path, split):
    return pd.read_parquet(
        path=os.path.join(path, f'split={split}'),
        engine="pyarrow"
    )

def read_txt(path):
    return open(path).read().strip().split('\n')


def save_json(path, obj):
    with open(path, 'w') as fp:
        json.dump(obj, fp)


def load_json(path):
    with open(path, 'rb') as fp:
        return json.load(fp)


def save_yaml_opts(path_yaml, opts):
    # Warning: copy is not nested
    options = copy.copy(opts)

    # https://gist.github.com/oglops/c70fb69eef42d40bed06
    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())
    Dumper.add_representer(Dict, dict_representer)

    with open(path_yaml, 'w') as yaml_file:
        yaml.dump(options, yaml_file, Dumper=Dumper, default_flow_style=False)


def merge_dictionaries(dict1, dict2):
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            merge_dictionaries(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]


def load_yaml_opts(path_yaml):
    """ Load options dictionary from a yaml file
    """
    result = {}
    with open(path_yaml, 'r') as yaml_file:
        options_yaml = yaml.safe_load(yaml_file)
        includes = options_yaml.get('__include__', False)
        if includes:
            if type(includes) != list:
                includes = [includes]
            for include in includes:
                filename = '{}/{}'.format(os.path.dirname(path_yaml), include)
                if os.path.isfile(filename):
                    parent = load_yaml_opts(filename)
                else:
                    parent = load_yaml_opts(include)
                merge_dictionaries(result, parent)
        # to be sure the main options overwrite the parent options
        merge_dictionaries(result, options_yaml)
    result.pop('__include__', None)
    result = Dict(result)
    return result


def parse_loader_name(data_name):
    if '.' in data_name:
        name, lang = data_name.split('.')
        return name, lang
    else:
        return data_name, None


def load_pickle(path, encoding="ASCII"):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding)


def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
