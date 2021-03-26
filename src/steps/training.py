import os
import pandas as pd
import numpy as np
import logging
import joblib
import lightgbm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
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
            logging.info(f'Training OneHotEncoder...')
            ohe = OneHotEncoder()
            ohe.fit(df[cates])
            joblib.dump(ohe, ohe_path)
        else:
            logging.info(f'Loading OneHotEncoder...')
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


def x_y_split(df):
    y = df[params['target']].copy()
    X = df.drop(columns=params['target']).copy()
    del(df)
    return X,y


def train(df, args):
    logging.info(f'-- training')
    # format dataset
    df = format_data(df, encode_categorical=False if args.model_name=='lgb' else True)
    # train-val split
    df, test_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)
    X, y = x_y_split(df)
    X_test, y_test = x_y_split(test_df)
    # train folds
    train_folds(X, y, X_test, y_test, args)


def train_lgb(X_train, y_train, X_val, y_val, args):
    lgb_params = params['lgb']
    lgb_params['random_state'] = args.seed
    lgb = lightgbm.LGBMRegressor(**lgb_params)

    if args.tuning:
        search_params = {
            'learning_rate': [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1, 5, 10],
            'n_estimators': [15, 30, 50, 80, 100, 120, 150, 200],
            'min_split_gain': [0, 10, 50, 100, 150, 300, 500, 1000],
            'num_leaves': [15, 30, 50, 80, 100, 120, 150, 200]
        }
        model = GridSearchCV(
            lgb, 
            search_params,
            scoring = 'neg_root_mean_squared_error',
            cv = 5, 
            n_jobs = -1, 
            verbose=True
        )
        model.fit(X_train, y_train)
        best_params = model.best_params_
        extend_res_summary({'GridSearch': best_params})
        lgb = lightgbm.LGBMRegressor(**best_params)

    lgb.fit(
        X=X_train, 
        y=y_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)],
        categorical_feature=params['cate_cols'],
        eval_metric='l2',
        verbose=False
    )
    return lgb


def plot_feat_imp(model, save_path):
    ax = lightgbm.plot_importance(model, height=5, figsize=(5,10))
    ax.figure.tight_layout()
    ax.figure.savefig(save_path)
    plt.close()


def plot_rmse(model, save_path):
    ax = lightgbm.plot_metric(model, metric='rmse')
    ax.figure.tight_layout()
    ax.figure.savefig(save_path)
    plt.close()


def plot_scatters(actual, pred, save_path, ext=''):
    ax = sns.scatterplot(x=actual, y=pred)
    ax.set(xlabel='actual', ylabel='predicted', title='Plot of Predicted vs Actual'+ext)
    ax.figure.tight_layout()
    ax.figure.savefig(save_path)
    plt.close()


def train_folds(X, y, X_test, y_test, args):

    pred_train = np.zeros(len(X))
    pred_test = defaultdict(list)
    kf = KFold(n_splits=args.folds, random_state=args.seed, shuffle=True)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X,y)):
        # get fold split
        logging.info(f'Conducting fold #{fold}...')
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        # train round
        if args.model_name == 'lgb':
            model = train_lgb(X_train, y_train, X_val, y_val, args)
        else:
            raise NotImplementedError
        
        # evaluate fold
        pred_train[val_idx] = model.predict(X_val)
        pred_test[fold] = model.predict(X_test)
        val_rmse = mean_squared_error(y_val, pred_train[val_idx], squared=False)
        test_rmse = mean_squared_error(y_test, pred_test[fold], squared=False)
        logging.info(f'TRAIN: n={len(trn_idx)} | VAL: n={len(val_idx)}, rmse={val_rmse} | TEST: n={len(y_test)}, rmse={test_rmse}')
        extend_res_summary({f'Val_RMSE_{fold}': val_rmse, f'Test_RMSE_{fold}': test_rmse})

        # save model
        joblib.dump(model, f'outs/{args.model_name}/fold{fold}.joblib')

        # save plots
        plot_feat_imp(
            model=model, 
            save_path=f'outs/{args.model_name}/fold{fold}_featimp.png'
            )
        plot_rmse(            
            model=model, 
            save_path=f'outs/{args.model_name}/fold{fold}_iters.png'
            )
        plot_scatters(
            actual=y_val, 
            pred=pred_train[val_idx], 
            save_path=f'outs/{args.model_name}/fold{fold}_val.png', 
            ext=f' in Val Fold{fold}'
            )
        plot_scatters(
            actual=y_test, 
            pred=pred_test[fold], 
            save_path=f'outs/{args.model_name}/fold{fold}_test.png', 
            ext=f' in Test Fold{fold}'
            )

    # evaluate overall
    pred_test['mean'] = pd.DataFrame(pred_test).mean(axis=1)
    val_rmse = mean_squared_error(y, pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, pred_test['mean'], squared=False)
    logging.info(f'OVERALL --> VAL: rmse={val_rmse} | TEST: rmse={test_rmse}')
    extend_res_summary({f'Val_RMSE': val_rmse, f'Test_RMSE': test_rmse})

    # save plots
    plot_scatters(
        actual=y, 
        pred=pred_train, 
        save_path=f'outs/{args.model_name}/val.png', 
        ext=f' in Val Fold{fold}'
        )
    plot_scatters(
        actual=y_test, 
        pred=pred_test['mean'], 
        save_path=f'outs/{args.model_name}/test.png', 
        ext=f' in Test Fold{fold}'
        )


def predict(df, args):
    logging.info(f'-- predicting')
    # format dataset
    df = format_data(df, encode_categorical=False if args.model_name=='lgb' else True)
    # predict folds
    pred_new = defaultdict(list)
    for fold in range(args.folds):
        model = joblib.load(f'outs/{args.model_name}/fold{fold}.joblib')
        feats = model._Booster.feature_name()
        # print([i for i in df.columns if i not in feats])
        # print(df.columns)
        pred_new[fold] = model.predict(df)
    pred_new = pd.DataFrame(pred_new)
    pred_new['mean'] = pred_new.mean(axis=1)
    pred_new = pred_new.reset_index(drop=True)
    pred_new.to_csv(f'outs/{args.model_name}/preds.csv', index=False)
    # submission format
    submission = pd.DataFrame()
    submission['Id'] = pred_new.index
    submission['Predicted'] = pred_new['mean']
    submission.to_csv(f'outs/{args.model_name}/submission.csv', index=False)