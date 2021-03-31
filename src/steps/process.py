import os
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from src.utils.files import save_json
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
            df = read_csv(os.path.join(args.data_folder, data_path), args.build_mode)
            df = generate_main_fe(df, fe_path, save_fe=save_fe)

        # Load or generate auxiliary features
        aux_df = pd.DataFrame()
        for aux in ['commercial', 'hawker', 'malls', 'station', 'demographics', 'prisch']:
            aux_fe_path = f'{args.out_folder}/{t_name}_df_fe_{aux}.csv'
            if load_if_avail and os.path.exists(aux_fe_path):
                logging.info(f'Opening fe auxiliary data for "{aux}"...')
                _aux_df = read_csv(aux_fe_path, args.build_mode)
            else:
                logging.info(f'Generating fe auxiliary data for "{aux}"...')
                # load all auxiliary data (no build mode here)
                _aux_df = read_csv(os.path.join(args.data_folder, aux_paths[aux]))
                _aux_df = generate_aux_fe(df, _aux_df, aux, aux_fe_path, save_fe=save_fe)
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

    # create alternative dep var
    if 'resale_price' in df.columns:
        df['resale_price_sqm'] = df['resale_price']/df['floor_area_sqm']

    # flat type
    df.loc[df['flat_type'] == "1-room", 'flat_type'] = "1 room"
    df.loc[df['flat_type'] == "2-room", 'flat_type'] = "2 room"
    df.loc[df['flat_type'] == "3-room", 'flat_type'] = "3 room"
    df.loc[df['flat_type'] == "4-room", 'flat_type'] = "4 room"
    df.loc[df['flat_type'] == "5-room", 'flat_type'] = "5 room"

    # count of 4 occurences in block no
    df['block'] = df['block'].apply(lambda x: x.count('4'))

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


def generate_aux_fe(df, aux_df, aux, aux_fe_path, save_fe=True):
    # create features per aux type
    if aux == 'demographics':
        aux_df, dnew_columns = generate_aux_demographic(df, aux_df, aux)
    elif aux == 'commercial':
        aux_df, dnew_columns = generate_aux_commercial(df, aux_df, aux)
    elif aux == 'hawker':
        aux_df, dnew_columns = generate_aux_hawker(df, aux_df, aux)
    elif aux == 'station':
        aux_df, dnew_columns = generate_aux_station(df, aux_df, aux)
    elif aux == 'malls':
        aux_df, dnew_columns = generate_aux_malls(df, aux_df, aux)
    elif aux == 'prisch':
        aux_df, dnew_columns = generate_aux_prisch(df, aux_df, aux)
    else:
        raise NotImplementedError

    # save frame if opted
    if save_fe:
        aux_df.to_csv(aux_fe_path, index=False)
        save_json(dnew_columns, aux_fe_path.split('.csv')[0]+'_cols.json')

    return aux_df


def generate_aux_demographic(df, aux_df, aux):
    dnew_columns = defaultdict(dict)
    conv_dict = {
        'kids': ['0-4', '5-9', '10-14'],                      # dependents
        'youth': ['15-19', '20-24'],                          # students/ part-timers
        'youngads': ['25-29', '30-34', '35-39'],              # young families
        'middle': ['40-44', '45-49', '50-54'],                # older families
        'older': ['55-59', '60-64'],                          # retirees
        'elderly': ['65-69', '70-74','75-79', '80-84', '85+'] # older group
    }
    rev_dict = {}
    for k,v in conv_dict.items():
        for i in v:
            rev_dict[i] = k
    aux_df['age_grp'] = aux_df['age_group'].apply(lambda x: rev_dict[x])
    aux_df = aux_df.groupby(['subzone', 'age_grp'])['count'].sum().unstack('age_grp').reset_index()
    df_x_aux = pd.merge(df, aux_df, how='left', on='subzone').iloc[:,-6:]
    df_x_aux.columns = [aux+'_'+i for i in df_x_aux.columns]
    dnew_columns[aux] = list(df_x_aux.columns)

    # 'city hall' and 'gali batu' does not have some information
    # we assume this is because none of a certain age group lives in that vicinity
    # thus we will fillna with 0
    df_x_aux = df_x_aux.fillna(0)

    return df_x_aux, dnew_columns


def generate_aux_commercial(df, aux_df, aux):
    dnew_columns = defaultdict(dict)
    # distance from each location
    df_x_aux, dnew_columns[aux] = create_main_aux_dist_cols(
        df.copy(), aux_df, aux)
    # distance from nearest type (grouped/min)
    grp_col_name = 'type'
    df_x_aux, dnew_columns[aux+'_'+grp_col_name] = create_grouped_cols(
        dnew_columns, df_x_aux, aux_df, aux, grp_col_name, new_frame=False)
    
    return df_x_aux, dnew_columns


def generate_aux_prisch(df, aux_df, aux):
    dnew_columns = defaultdict(dict)
    # distance from each location
    df_x_aux, dnew_columns[aux] = create_main_aux_dist_cols(
        df.copy(), aux_df, aux)
    # create top 50 variable
    aux_df['top50'] = [
        '' if i>0 else None 
        for i in aux_df[
            ['KiasuRank', '2020over', '2019over', '2018over','2017over']
            ].sum(axis=1)]
    # distance from nearest top school (grouped/min)
    grp_col_name = 'top50'
    df_x_aux, dnew_columns[aux+'_'+grp_col_name] = create_grouped_cols(
        dnew_columns, df_x_aux, aux_df, aux, grp_col_name, new_frame=False)
    # create dummies that permit phase applications for pri schools
    df_x_aux['prisch_top50_<=1km'] = df_x_aux['prisch_top50_'].apply(lambda x: 1 if x<=1 else 0)
    df_x_aux['prisch_top50_1to2km'] = df_x_aux['prisch_top50_'].apply(lambda x: 1 if (x>1 and x<=2) else 0)
    df_x_aux['prisch_top50_2to4km'] = df_x_aux['prisch_top50_'].apply(lambda x: 1 if (x>2 and x<=4) else 0)

    return df_x_aux, dnew_columns


def Haversine(lat1, lon1, lat2, lon2, roundoff=4):
    """
    Code Source: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude

    This uses the ‘haversine’ formula to calculate the great-circle distance between two points – that is, 
    the shortest distance over the earth’s surface – giving an ‘as-the-crow-flies’ distance between the points 
    (ignoring any hills they fly over, of course!).
    Haversine
    formula:    a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    c = 2 ⋅ atan2( √a, √(1−a) )
    d = R ⋅ c
    where   φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
    note that angles need to be in radians to pass to trig functions!
    """
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) ** 2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = R * c
    return round(d, roundoff)


def get_distance_between_main_aux(df_row, aux_row, verbose=False):
    distance = Haversine(
        df_row['latitude'], df_row['longitude'],
        aux_row['lat'], aux_row['lng']
    )
    if verbose:
        subzone = df_row['subzone']
        auxname = aux_row['name']
        print(f'Distance between "{subzone}" and "{auxname}": {distance}')
    return distance


def create_main_aux_dist_cols(df, _aux_df, aux='', aux_col_name='name', df_lat_name='latitude', df_lng_name='longitude',
                              aux_lat_name='lat', aux_lng_name='lng', verbose=False, new_frame=True):
    """
    Assumes the following column naming convensions:
    df: has columns ['latitude', 'longitude']
    aux: has columns ['name', 'lat', 'lng']
    """
    out_df = pd.DataFrame()
    dcol_conversion = defaultdict(str)
    for aux_ix, aux_row in _aux_df.iterrows():
        # generate new column names
        abrv_name = aux_row[aux_col_name]
        if ' ' in abrv_name:
            abrv_name = ''.join([s[0] if s[0].isupper() else (
                s if s.isnumeric() else '')for s in abrv_name.split(' ')])
            abrv_name = abrv_name.replace('_', '')
        col_name = aux + '_' + abrv_name
        # store column conversion
        if col_name in dcol_conversion.values():
            col_name += 'V' + str(aux_ix)  # create a new unique column
        dcol_conversion[aux_row[aux_col_name]] = col_name
        # generate columns
        out_df[col_name] = Haversine(
            df[df_lat_name], df[df_lng_name], aux_row[aux_lat_name], aux_row[aux_lng_name])
        # complete
        if verbose:
            print(f'Created new column "{col_name}"...')
    if new_frame:
        return out_df, dcol_conversion
    else:
        return pd.concat([df, out_df], axis=1), dcol_conversion


def mmin(df):
    return df.min(axis=1)


def create_grouped_cols(
    dnew_columns, df, _aux_df, aux='', grp_col_name='type', function=mmin, 
    verbose=False, new_frame=True):

    out_df = pd.DataFrame()
    dcol_conversion = defaultdict(str)
    for grp_ix, grp in enumerate(_aux_df[grp_col_name].unique()):
        # we do not create new cols for missings
        if grp is None:
            continue
        # generate new column names
        col_name = aux + '_' + grp_col_name + '_' + grp
        # store column conversion
        if col_name in dcol_conversion.values():
            col_name += '_' + str(grp_ix)  # create a new unique column
        dcol_conversion[grp] = col_name
        relevant_columns = [dnew_columns[aux][old]
                            for old in _aux_df[_aux_df[grp_col_name] == grp]['name']]
        out_df[col_name] = function(df[relevant_columns])
        # complete
        if verbose:
            print(f'Created new column "{col_name}"...')
    if new_frame:
        return out_df, dcol_conversion
    else:
        return pd.concat([df, out_df], axis=1), dcol_conversion


def label_rows_by_index(full_indexes, positive_indexes, positive_label, negative_label=None):
    """ create group tags """
    return [positive_label if i in positive_indexes else negative_label for i in full_indexes]


def generate_aux_hawker(df, aux_df, aux):
    dnew_columns = defaultdict(dict)
    aux_df[''] = ''
    grp_col_name = ''
    # distance from each hawker
    df_x_aux, dnew_columns[aux] = create_main_aux_dist_cols(
        df.copy(), aux_df, aux)
    # distance from nearest hawker
    df_x_aux, dnew_columns[aux+'_'+grp_col_name] = create_grouped_cols(
        dnew_columns, df_x_aux, aux_df, aux, grp_col_name, new_frame=False)

    # reviews
    crawled_path = f'data/auxiliary-data/google_search_{aux}.csv'
    aux_df2 = pd.read_csv(crawled_path)
    aux_df2 = aux_df2.rename(
        columns={'Unnamed: 0': 'aux_ix', 'name': 'crawled_name'})
    # construct local df version
    _aux_df = pd.concat([aux_df, aux_df2], axis=1)
    # distance from high ratings hawker
    grp_col_name = 'highrating'
    _aux_df[grp_col_name] = label_rows_by_index(
        full_indexes=_aux_df.index,
        positive_indexes=_aux_df[(_aux_df['fuzzy_score'] > 70) & (
            _aux_df['user_ratings_total'] > 5) & (_aux_df['rating'] > 4)].index,
        positive_label=''
    )
    df_x_aux, dnew_columns[aux+'_'+grp_col_name] = create_grouped_cols(
        dnew_columns, df_x_aux, _aux_df, aux, grp_col_name, new_frame=False)

    # distance from established hawker
    grp_col_name = 'established'
    _aux_df[grp_col_name] = label_rows_by_index(
        full_indexes=_aux_df.index,
        positive_indexes=_aux_df[(_aux_df['fuzzy_score'] > 70) & (
            _aux_df['user_ratings_total'] > 15)].index,
        positive_label=''
    )
    df_x_aux, dnew_columns[aux+'_'+grp_col_name] = create_grouped_cols(
        dnew_columns, df_x_aux, _aux_df, aux, grp_col_name, new_frame=False)

    return df_x_aux, dnew_columns


def generate_aux_malls(df, aux_df, aux):
    dnew_columns = defaultdict(dict)
    # manual fix loyang point 1.3670, 103.9644
    aux_ix = 94
    aux_df.loc[aux_ix, 'lat'] = 1.3670
    aux_df.loc[aux_ix, 'lng'] = 103.9644
    aux_df.loc[aux_ix]

    aux_df[''] = ''
    grp_col_name = ''
    # distance from each mall
    df_x_aux, dnew_columns[aux] = create_main_aux_dist_cols(
        df.copy(), aux_df, aux)
    # distance from nearest mall
    df_x_aux, dnew_columns[aux+'_'+grp_col_name] = create_grouped_cols(
        dnew_columns, df_x_aux, aux_df, aux, grp_col_name, new_frame=False)

    # reviews
    crawled_path = f'data/auxiliary-data/google_search_{aux}.csv'
    aux_df2 = pd.read_csv(crawled_path)
    aux_df2 = aux_df2.rename(
        columns={'Unnamed: 0': 'aux_ix', 'name': 'crawled_name'})
    # construct local df version
    _aux_df = pd.concat([aux_df, aux_df2], axis=1)

    # create grouping by ratings
    # rationale: malls differ in ranges 4.5-4.0 alot (Central to Local malls)
    grp_col_name = 'ratingsbin'
    _aux_df2 = _aux_df[(_aux_df['fuzzy_score'] > 70) & (
        _aux_df['user_ratings_total'] > 5)].copy()
    _aux_df2[grp_col_name] = _aux_df2['rating'].apply(
        lambda x: None if pd.isnull(x) else (
            '>=4.5' if x >= 4.5 else (
                '4.4' if x >= 4.4 else (
                    '4.3' if x >= 4.3 else (
                        '4.2' if x >= 4.2 else (
                            '4.1' if x >= 4.1 else (
                                '4.0' if x >= 4.0 else ">4.0"))))))
    )
    df_x_aux, dnew_columns[aux+'_'+grp_col_name] = create_grouped_cols(
        dnew_columns, df_x_aux, _aux_df2, aux, grp_col_name, new_frame=False)

    # distance from established mall
    grp_col_name = 'established'
    _aux_df[grp_col_name] = label_rows_by_index(
        full_indexes=_aux_df.index,
        positive_indexes=_aux_df[(_aux_df['fuzzy_score'] > 70) & (
            _aux_df['user_ratings_total'] > 15)].index,
        positive_label=''
    )
    df_x_aux, dnew_columns[aux+'_'+grp_col_name] = create_grouped_cols(
        dnew_columns, df_x_aux, _aux_df, aux, grp_col_name, new_frame=False)

    return df_x_aux, dnew_columns


def generate_aux_station(df, aux_df, aux):
    dnew_columns = defaultdict(dict)
    # manual fix botanic gardens is an mrt
    aux_ix = 139
    aux_df.loc[aux_ix, 'type'] = 'mrt'
    aux_df.loc[aux_ix, 'opening_year'] = 2011
    aux_df.loc[aux_ix]

    # fix for duplicate rows that exists in mrt data
    _aux_df = aux_df.copy()
    _aux_df = _aux_df.groupby(['name', 'type']).agg(
        {'codes': '/'.join, 'lat': np.mean, 'lng': np.mean, 'opening_year': np.min}).reset_index()

    # generate groupings
    _aux_df['numlines'] = _aux_df['codes'].apply(lambda x: x.count('/')+1)
    _aux_df['interchange'] = label_rows_by_index(
        full_indexes=_aux_df.index,
        positive_indexes=_aux_df[_aux_df['numlines'] > 1].index,
        positive_label=''
    )

    # group by main lines
    for line in ['EW', 'NS', 'NE', 'CC', 'DT']:
        _aux_df[line] = label_rows_by_index(
            full_indexes=_aux_df.index,
            positive_indexes=[ix for ix, code in enumerate(
                _aux_df['codes']) if line in code],
            positive_label=''
        )

    # distance from each mrt stn
    df_x_aux, dnew_columns[aux] = create_main_aux_dist_cols(
        df.copy(), _aux_df, aux)

    # overwrite with NaN if MRT not opened then
    # aux_row = _aux_df[_aux_df['opening_year']>=2004].iloc[aux_ix]
    dcol_conversion = dnew_columns[aux]
    for aux_ix, aux_row in _aux_df[_aux_df['opening_year'] >= 2004].iterrows():
        focus_yr, focus_name = aux_row['opening_year'], aux_row['name']
        focus_col = dcol_conversion[focus_name]
        # create new column
        df_x_aux[focus_col+'_open'] = [
            ds if yr > focus_yr-5 else np.nan for yr, ds in zip(df['resale_year'].astype(int), df_x_aux[focus_col].astype(float))]
        # new column naming
        dcol_conversion[focus_name] = focus_col+'_open'
    dnew_columns[aux] = dcol_conversion

    # distance from group type
    for grp_col_name in ['type', 'interchange', 'EW', 'NS', 'NE', 'CC', 'DT']:
        df_x_aux, dnew_columns[aux+'_'+grp_col_name] = create_grouped_cols(
            dnew_columns, df_x_aux, _aux_df, aux, grp_col_name, new_frame=False)

    return df_x_aux, dnew_columns
