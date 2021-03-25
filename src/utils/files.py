import json
import os
import pandas as pd
import numpy as np
import warnings
import argparse
import random
from pathlib import Path


def open_json(json_file_path, data_format=list):
    if data_format==dict or data_format=='dict':
        with open(json_file_path) as json_file:
            data = json.load(json_file)
    elif data_format==list or data_format=='list':
        data = []
        for line in open(json_file_path, encoding='utf-8'):
            data.append(json.loads(line))
    elif data_format==pd.DataFrame or data_format=='pd.DataFrame':
        data = pd.read_json(json_file_path, orient="records", lines=True)
    else:
        raise NotImplementedError
    return data


def save_json(ddict, json_file_path):
    with open(json_file_path, 'w') as fp:
        json.dump(ddict, fp)


def make_dir(save_path):
    path = Path(save_path)
    if not os.path.isdir(path.parent):
        os.makedirs(path.parent, exist_ok=True)


def set_seeds(seed):
    # for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)


def str2bool(v):
    """
    Code source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
