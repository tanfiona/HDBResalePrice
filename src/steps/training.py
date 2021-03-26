import os
import pandas as pd
import numpy as np
import logging
import joblib
import lightgbm
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from src.utils.logger import extend_res_summary
from src.utils.refs import params
import warnings
warnings.filterwarnings("ignore")


def format_data(df, encode_categorical=False):
    """
    This function formats the columns of the dataframe into the right format before
    the dataframe is parsed for training or testing.
    In train mode, OneHotEncoder model is trained and saved, we reuse the model later. 
    In test mode, we reload the OneHotEncoder for using.
    """

    nums = params['num_cols']+params['aux_cols']
    if params['target'] in df.columns:
        train_mode = True
        nums += [params['target']]
    cates = params['cate_cols']
    cols = nums + cates

    # numericals
    for col in nums:
        df[col] = df[col].astype(float)

    # strings
    if encode_categorical:
        for col in cates:
            df[col] = df[col].astype(str)
        # usually most ml packages need dummy encoding e.g. via One-Hot
        ohe_path = 'outs/ohe_encoder.joblib'
        if train_mode:
            ohe = OneHotEncoder()
            ohe.fit(df[cates])
            joblib.dump(ohe, ohe_path)
        else:
            ohe = joblib.load(ohe_path) 
        df_dummies = pd.DataFrame(ohe.transform(df[cates]))
        df_dummies.columns = ohe.get_feature_names(cates)
        df = df.drop(columns=cates)
        df = pd.concat([df, df_dummies], axis=1)
    else:
        # lgb can read categories directly
        for col in cates:
            df[col] = df[col].astype('category')
    return df[cols]


def train(df, args):
    logging.info(f'-- training')
    # format dataset
    df = format_data(df, encode_categorical=False if args.model_name=='lgb' else True)
    # train-val split
    df, test_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)
    y, y_test = df[params['target']].copy(), test_df[params['target']].copy()
    X, X_test = df.drop(columns=params['target']), test_df.drop(columns=params['target'])
    # train folds
    train_folds(X, y, X_test, y_test, args)


def train_lgb(X_train, y_train, X_val, y_val, args):
    lgb_params = params['lgb']
    lgb_params['random_state'] = args.seed
    lgb = lightgbm.LGBMRegressor(**lgb_params)
    lgb.fit(
        X=X_train, 
        y=y_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)],
        categorical_feature=params['cate_cols'],
        eval_metric='l2',
        verbose=False
    )
    return lgb


def train_folds(X, y, X_test, y_test, args):

    pred_train = np.zeros(len(X))
    pred_test = defaultdict(list)
    kf = KFold(n_splits=args.folds, random_state=None, shuffle=False)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X,y)):
        # get fold split
        print(f'Conducting fold #{fold}...')
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        # train round
        if args.model_name == 'lgb':
            model = train_lgb(X, y, X_test, y_test, args)
        else:
            raise NotImplementedError
        
        # evaluate fold
        pred_train[val_idx] = model.predict(X_val)
        pred_test[fold] = model.predict(X_test)
        val_rmse = mean_squared_error(y_val, pred_train[val_idx], squared=False)
        test_rmse = mean_squared_error(y_test, pred_test[fold], squared=False)
        print(f'TRAIN: n={len(trn_idx)} | VAL: n={len(val_idx)}, rmse={val_rmse} | TEST: n={len(y_test)}, rmse={test_rmse}')
        extend_res_summary({f'Val_RMSE_{fold}': val_rmse, f'Test_RMSE_{fold}': test_rmse})

        # save model
        model_path = f'outs/{args.model_name}/fold{fold}.joblib'
        joblib.dump(model, model_path)

    # evaluate overall
    pred_test['mean'] = pd.DataFrame(pred_test).mean(axis=1)
    val_rmse = mean_squared_error(y, pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, pred_test['mean'], squared=False)
    print(f'OVERALL --> VAL: rmse={val_rmse} | TEST: rmse={test_rmse}')
    extend_res_summary({f'Val_RMSE': val_rmse, f'Test_RMSE': test_rmse})


def predict(df, args):
    logging.info(f'-- predicting')
    # format dataset
    df = format_data(df, encode_categorical=False if args.model_name=='lgb' else True)
    # predict folds
    pred_new = defaultdict(list)
    for fold in range(args.folds):
        model = joblib.load(f'outs/{args.model_name}/fold{fold}.joblib')
        pred_new[fold] = model.predict(df)
    pred_new['mean'] = pd.DataFrame(pred_new).mean(axis=1)
    pred_new.to_csv(f'outs/{args.model_name}/preds.csv', index=False)