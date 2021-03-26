import os
import pandas as pd
import numpy as np
import logging
from src.utils.refs import aux_paths, params


def read_csv(path, build_mode=False):
    _df = pd.read_csv(path)
    if build_mode:
        _df = _df[0:50]
    return _df


def preprocess(data_path, args, load_if_avail=True, save_fe=True, save_all=True):
    """
    Feature cleaning
    Feature engineering
    Feature selection (minimally)
    """

    t_name = data_path.split('.csv')[0]
    all_fe_path = f'{args.out_folder}/{t_name}_df_fe_all.csv'
    logging.info(f'-- starting {t_name} fe')

    if load_if_avail and os.path.exists(all_fe_path):
        logging.info(f'Opening fe data for all...')
        df = read_csv(all_fe_path, args.build_mode)
    else:
        # Load or generate main features
        fe_path = f'{args.out_folder}/{t_name}_df_fe.csv'
        if load_if_avail and os.path.exists(fe_path):
            logging.info(f'Opening fe data for main...')
            df = read_csv(fe_path, args.build_mode)
        else:
            logging.info(f'Generating fe data for main...')
            df = read_csv(os.path.join(args.data_folder, args.train_data_name), args.build_mode)
            df = generate_main_fe(df, fe_path, save_fe=save_fe)

        # Load or generate auxiliary features
        aux_df = pd.DataFrame()
        for aux in ['commercial', 'hawker', 'malls', 'station', 'demographics']:
            aux_fe_path = f'{args.out_folder}/{t_name}_df_fe_{aux}.csv'
            if load_if_avail and os.path.exists(aux_fe_path):
                logging.info(f'Opening fe auxiliary data for "{aux}"...')
                _aux_df = read_csv(aux_fe_path, args.build_mode)
            else:
                logging.info(f'Generating fe auxiliary data for "{aux}"...')
                # load all auxiliary data (no build mode here)
                _aux_df = read_csv(os.path.join(args.data_folder, aux_paths[aux]))
                _aux_df = generate_aux_fe(df, _aux_df, aux_fe_path, save_fe=save_fe)
            keep_columns = [i for i in _aux_df.columns if i in params['aux_cols']]
            _aux_df = _aux_df[keep_columns]
            # concat to frame
            aux_df = pd.concat([aux_df, _aux_df], axis=1)
        logging.info(f'Auxiliary data has {len(aux_df)} rows, {len(aux_df.columns)} cols')

        # Combine features
        df = pd.concat([df, aux_df], axis=1)
        logging.info(f'Final data has {len(df)} rows, {len(df.columns)} cols')

        # Save frame
        if save_all:
            df.to_csv(all_fe_path, index=False)
        logging.info(f'-- complete {t_name} fe')

    return df


def generate_main_fe(df, fe_path, save_fe=True):
    """
    Note that this section is manual, created by domain knowledge.
    """

    # resale timing
    df[['resale_year', 'resale_month']] = df['month'].str.split('-', 1, expand=True)
    df['flat_age'] = df['resale_year'].astype(int)-df['lease_commence_date'].astype(int)

    # flat type
    df.loc[df['flat_type'] == "1-room", 'flat_type'] = "1 room"
    df.loc[df['flat_type'] == "2-room", 'flat_type'] = "2 room"
    df.loc[df['flat_type'] == "3-room", 'flat_type'] = "3 room"
    df.loc[df['flat_type'] == "4-room", 'flat_type'] = "4 room"
    df.loc[df['flat_type'] == "5-room", 'flat_type'] = "5 room"

    # converting the block column to 1 if it has the number 4
    # converting the block column to 0 if it does not have the number 4
    df.loc[df['block'].str.contains('4'),'block'] = 1
    df.loc[df['block'].str.contains('4') == False, 'block'] = 0

    # convert to 01 to 06, 06 to 10, 10 to 15, 16 to 21, 21 to 25, 25 to 30, 
    # 31 to 36, 36 to 40, 40 to 45, 46 to 51
    # data is messy as it has lots of overlaps, so the partioning is to make
    # it more systematic
    # 01 to 06
    df.loc[df['storey_range'] == "01 to 03", 'storey_range'] = "01 to 06"
    df.loc[df['storey_range'] == "01 to 05", 'storey_range'] = "01 to 06"
    df.loc[df['storey_range'] == "04 to 06", 'storey_range'] = "01 to 06"
    # 06 to 10
    df.loc[df['storey_range'] == "07 to 09", 'storey_range'] = "06 to 10"
    # 10 to 15
    df.loc[df['storey_range'] == "10 to 12", 'storey_range'] = "10 to 15"
    df.loc[df['storey_range'] == "11 to 15", 'storey_range'] = "10 to 15"
    df.loc[df['storey_range'] == "13 to 15", 'storey_range'] = "10 to 15"
    # 16 to 21
    df.loc[df['storey_range'] == "16 to 18", 'storey_range'] = "16 to 21"
    df.loc[df['storey_range'] == "16 to 20", 'storey_range'] = "16 to 21"
    df.loc[df['storey_range'] == "19 to 21", 'storey_range'] = "16 to 21"
    # 21 to 25
    df.loc[df['storey_range'] == "22 to 24", 'storey_range'] = "21 to 25"
    # 25 to 30
    df.loc[df['storey_range'] == "25 to 27", 'storey_range'] = "25 to 30"
    df.loc[df['storey_range'] == "26 to 30", 'storey_range'] = "25 to 30"
    df.loc[df['storey_range'] == "28 to 30", 'storey_range'] = "25 to 30"
    # 31 to 36
    df.loc[df['storey_range'] == "31 to 33", 'storey_range'] = "31 to 36"
    df.loc[df['storey_range'] == "31 to 35", 'storey_range'] = "31 to 36"
    df.loc[df['storey_range'] == "34 to 36", 'storey_range'] = "31 to 36"
    # 36 to 40
    df.loc[df['storey_range'] == "37 to 39", 'storey_range'] = "36 to 40"
    # 40 to 45
    df.loc[df['storey_range'] == "40 to 42", 'storey_range'] = "40 to 45"
    df.loc[df['storey_range'] == "43 to 45", 'storey_range'] = "40 to 45"
    # 46 to 51
    df.loc[df['storey_range'] == "46 to 48", 'storey_range'] = "46 to 51"
    df.loc[df['storey_range'] == "49 to 51", 'storey_range'] = "46 to 51"

    # save frame if opted
    if save_fe:
        df.to_csv(fe_path, index=False)
    
    return df


def generate_aux_fe(df, aux_df, aux_fe_path, save_fe=True):
    # create features per aux type
    if 'demographics' in aux_fe_path:
        aux_df = generate_aux_demographic(df, aux_df)
    elif 'commercial' in aux_fe_path:
        aux_df = generate_aux_commercial(df, aux_df)
    elif 'hawker' in aux_fe_path:
        aux_df = generate_aux_hawker(df, aux_df)
    elif 'station' in aux_fe_path:
        aux_df = generate_aux_station(df, aux_df)
    elif 'malls' in aux_fe_path:
        aux_df = generate_aux_malls(df, aux_df)
    else:
        raise NotImplementedError
    
    # save frame if opted
    if save_fe:
        aux_df.to_csv(aux_fe_path, index=False)
    
    return aux_df


def generate_aux_demographic(df, aux_df):
    # only this column used
    df = df[['subzone']].copy()

    # population count across age in a particular subzone
    dicts = {}
    for area in np.unique(aux_df.subzone):
        area_count = aux_df[aux_df['subzone'] == area]['count'].sum()
        dicts[area] = area_count 
    df['popcount_subzone'] = df['subzone'].map(dicts)

    # 490 was derived from central subzone in the population demographics
    # dataset. However, there is no such subzone in the main dataset. After
    # verifying it, central subzone is inferred to be 'city hall' in main 
    # data set (beach road area)
    df.loc[df['subzone'] == "city hall", 'popcount_subzone'] = 490

    return df[['popcount_subzone']]

    
##### WIP #####
def generate_aux_hawker(df, aux_df):
    pass

def generate_aux_station(df, aux_df):
    pass

def generate_aux_malls(df, aux_df):
    pass