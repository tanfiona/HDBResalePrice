import logging
import os
import tempfile
import pandas as pd
from itertools import product
from .files import make_dir, save_json, open_json
from datetime import datetime

# get unique tmp file per run
tmp_file_path = f'outs/tmp/{next(tempfile._get_candidate_names())}.json'
make_dir(tmp_file_path)


def get_log_level(set_level):
    if set_level.lower() == 'info':
        set_level = logging.INFO
    elif set_level.lower() == 'debug':
        set_level = logging.DEBUG
    elif set_level.lower() == 'warning':
        set_level = logging.WARNING
    elif set_level.lower() == 'critical':
        set_level = logging.CRITICAL
    else:
        raise NotImplementedError
    return set_level


def get_logger(logname, no_stdout=True, set_level='info', datefmt='%d/%m/%Y %H:%M:%S'):
    set_level = get_log_level(set_level)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt=datefmt, level=set_level)

    logger = logging.getLogger()
    logger.setLevel(set_level)
    handler = logging.StreamHandler(open(logname, "a"))
    handler.setLevel(set_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt=datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if no_stdout:
        logger.removeHandler(logger.handlers[0])

    return logger


def log_params(args, save_results=True, tmp_file_path=tmp_file_path, datefmt='%d/%m/%Y %H:%M:%S'):
    if save_results:
        res_summary = {
            'starttime': datetime.now().strftime(datefmt)}
        res_summary.update(vars(args))
        save_json(res_summary, tmp_file_path)

    # if args.run_type.lower() == 'run_one':
    #     logging.info(
    #         f'{args.run_type} --> classify_by: {args.classify_by} | model_name: {args.model_name} | ' +
    #         f'val_size: {args.val_size} |')
    # elif args.run_type.lower() == 'run_kfolds':
    #     logging.info(
    #         f'{args.run_type} --> classify_by: {args.classify_by} | model_name: {args.model_name} | ' +
    #         f'k_folds: {args.folds} |')
    # else:
    #     logging.warning('No such run_type is specified yet.')
    #     raise NotImplementedError


def extend_res_summary(additional_res, tmp_file_path=tmp_file_path):
    res_summary = open_json(tmp_file_path, data_format=dict)
    res_summary.update(additional_res)
    save_json(res_summary, tmp_file_path)


def get_average(df, filter_by):
    """
    df [pd.DataFrame]
    filter_by [list or tuples] : list of strings to filter column names for averaging across folds
    E.g. "Train_K3_Micro_F1" can be found via ["Train", "Micro_F1"]
    Does edits in place!
    """
    keep_cols = [col for col in df.columns if all(
        [fil in col for fil in filter_by])]
    df['AVG_'+'_'.join(filter_by)] = df[keep_cols].mean(axis=1)


def save_results_to_csv(save_file_path, append=True, tmp_file_path=tmp_file_path, datefmt='%d/%m/%Y %H:%M:%S'):
    """
    Takes res_summary of current run (in json format) and appends to main results frame (in csv format)
    """
    # load tmp results
    res_summary = open_json(tmp_file_path, data_format=pd.DataFrame)

    # calculate average scores
    combis = list(product(['Train', 'Val'], [
                  'ARI_Macro', 'ARI_Micro', 'F1_Macro', 'F1_Micro']))
    for combi in combis:
        get_average(res_summary, combi)

    # calculate end time
    end = datetime.now()
    res_summary['endtime'] = end.strftime(datefmt)
    res_summary['timetaken'] = end - \
        datetime.strptime(res_summary['starttime'][0], datefmt)

    if append and os.path.isfile(save_file_path):
        # load old file
        old_summary = pd.read_csv(save_file_path)
        # append below
        res_summary = pd.concat([old_summary, res_summary], axis=0)

    # save final and delete tmp file
    res_summary.to_csv(save_file_path, index=False)
    os.remove(tmp_file_path)
