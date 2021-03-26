# packages
import numpy as np
from datetime import datetime
from collections import defaultdict
import os
import time
import argparse
import sys
import pandas as pd

# project
from src.utils.files import open_json, make_dir, set_seeds, set_warnings, str2bool
from src.utils.logger import get_logger, log_params, get_log_level, save_results_to_csv
from src.steps.process import preprocess
from src.steps.training import train, predict

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--run_type', type=str, default='full',
                    help='process to run | options: full, train, predict, process')
parser.add_argument('--model_name', type=str, default='lgb', 
                    help='name of the model | options: lgb, xgb, rf, svm')
parser.add_argument('--folds', type=int, default=5,
                    help='number of training folds')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of training epochs')
parser.add_argument('--val_size', type=float, default=0.2,
                    help='proportion of data to use for validation')

# general
parser.add_argument('--data_folder', type=str, default='data',
                    help='folder name where data is located')
parser.add_argument('--train_data_name', type=str,
                    default='train.csv', help='name of train data file')
parser.add_argument('--test_data_name', type=str,
                    default='test.csv', help='name of test data file')
parser.add_argument('--out_folder', type=str, default='outs',
                    help='folder name to save outputs into')
parser.add_argument('--log_file', type=str,
                    default='training.log', help='filename to save log')
parser.add_argument('--results_file', type=str,
                    default='results.csv', help='filename to save results summary')
parser.add_argument('--plot_save_name', type=str, default='results_plot.png',
                    help='filename to save results plot in png format')
parser.add_argument('--model_save_name', type=str, default='best_model_state.bin',
                    help='filename to save best model in bin format')
parser.add_argument('--train_save_name', type=str,
                    help='filename to save training results in csv format (e.g. train_res.csv)')
parser.add_argument('--val_save_name', type=str, default='val_res.csv',
                    help='filename to save validation results in csv format (e.g. val_res.csv)')

# backend
parser.add_argument('--build_mode', type=str2bool, nargs='?',
                    default=False, help='activate build mode (aka reduced data size)')
parser.add_argument('--use_cpu', type=str2bool, nargs='?', default=False,
                    help='overwrite and use cpu (even if gpu is available)')
parser.add_argument('--cuda_device', type=str, default='0',
                    help='set which cuda device to run on (0 or 1)')
parser.add_argument('--log_level', type=str, default='info',
                    help='set logging level to store and print statements')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed to set for reproducibility')

args = parser.parse_args()

# setting up
set_seeds(args.seed)
set_warnings()
log_save_path = f'{args.out_folder}/{args.model_name}/{args.log_file}'.lower()
make_dir(log_save_path)
logger = get_logger(log_save_path, no_stdout=False, set_level=args.log_level)


def full_run():
    """
    Full run, inclusive of feature engineering, appending frames, train/val/predict
    """
    train_df = preprocess(data_path=args.train_data_name, args=args)
    train(train_df, args)
    test_df = preprocess(data_path=args.test_data_name, args=args)
    predict(test_df, args)


def main(task):
    task_func = {
        'full': full_run,
        # 'run_one': run_one
    }
    task_func[task]()
    

if __name__ == "__main__":
    logger.info('-- starting process')
    log_params(args)
    main(args.run_type)
    save_results_to_csv(os.path.join(args.out_folder, args.results_file))
    logger.info('-- complete process')