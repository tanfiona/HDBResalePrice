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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from src.utils.logger import extend_res_summary
from src.utils.refs import params
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


def format_data(df, args, encode_categorical=False):
    """
    This function formats the columns of the dataframe into the right format before
    the dataframe is parsed for training or testing.
    In train mode, OneHotEncoder model is trained and saved, we reuse the model later. 
    In test mode, we reload the OneHotEncoder for using.
    """

    nums = params['num_cols']+params['aux_cols']
    if (args.target in df.columns) and args.run_type!='predict':
        train_mode = True
        nums += [args.target]
    else:
        train_mode = False
    cates = params['cate_cols']
    cols = nums + cates

    # numericals
    for col in nums:
        df[col] = df[col].astype(float)

    # strings
    for col in cates:
        df[col] = df[col].apply(lambda x: str(x).lower())
    if encode_categorical:
        # usually most ml packages need dummy encoding e.g. via One-Hot
        ohe_path = 'outs/ohe_encoder.joblib'
        if train_mode or not os.path.exists(ohe_path):
            logging.info(f'Training OneHotEncoder...')
            ohe = OneHotEncoder()
            ohe.fit(df[cates])
            joblib.dump(ohe, ohe_path)
        else:
            logging.info(f'Loading OneHotEncoder...')
            ohe = joblib.load(ohe_path)
        # generate dummies
        df_dummies_columns = ohe.get_feature_names(cates) 
        df_dummies = pd.DataFrame(
            data=ohe.transform(df[cates]).todense(), 
            columns=df_dummies_columns
            )
        # update main df
        df = df.drop(columns=cates)
        df = pd.concat([df, df_dummies], axis=1)
        cols = nums + list(df_dummies_columns)
    else:
        # lgb can read categories directly
        for col in cates:
            df[col] = df[col].astype('category')
    return df[cols]


def x_y_split(df, target):
    y = df[target].copy()
    X = df.drop(columns=target).copy()
    del(df)
    return X,y


def train(df, args):
    logging.info(f'-- training')
    extend_res_summary({'params': params})
    # format dataset
    df = format_data(df, args, encode_categorical=False if args.model_name=='lgb' else True)
    # train-val split
    df, test_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)
    X, y = x_y_split(df, args.target)
    X_test, y_test = x_y_split(test_df, args.target)
    # train folds
    train_folds(X, y, X_test, y_test, args)


def train_lgb(X_train, y_train, X_val, y_val, args):
    model_params = params['lgb']
    model_params['random_state'] = args.seed
    model = lightgbm.LGBMRegressor(**model_params)
    
    if args.tuning:
        search_params = {
            'learning_rate': [1e-2, 0.05, 0.1],
            'n_estimators': [200, 300, 500, 800, 1000],
            'min_split_gain': [0, 10, 100],
            'num_leaves': [150, 200, 300, 500, 800]
            # 'learning_rate': [1e-4, 1e-3, 1e-2, 0.05, 0.1],
            # 'n_estimators': [15, 30, 50, 100, 150, 200, 300, 500],
            # 'min_split_gain': [0, 10, 100],
            # 'num_leaves': [15, 30, 50, 100, 150, 200, 300, 500]
            # 'learning_rate': [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1, 5, 10],
            # 'n_estimators': [15, 30, 50, 80, 100, 120, 150, 200],
            # 'min_split_gain': [0, 10, 50, 100, 150, 300, 500, 1000],
            # 'num_leaves': [15, 30, 50, 80, 100, 120, 150, 200]
        }
        best_params = param_tuning(X_train, y_train, model, search_params, args, X_val, y_val)
        for k, v in best_params.items():
            model_params[k] = v
        model = lightgbm.LGBMRegressor(**model_params)
    
    model.fit(
        X=X_train, 
        y=y_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)],
        categorical_feature=params['cate_cols'],
        eval_metric='l2',
        verbose=False
    )
    return model


def train_knn(X_train, y_train, X_val, y_val, args):
    model_params = params['knn']
    model = KNeighborsRegressor(**model_params)

    if args.tuning:
        search_params = {
            'p': [1, 2],
            'leaf_size': [5, 10, 15, 20, 30, 50, 100],
            'n_neighbors': [5, 15, 45, 100, 200]
        }
        best_params = param_tuning(X_train, y_train, model, search_params, args)
        for k, v in best_params.items():
            model_params[k] = v
        model = KNeighborsRegressor(**model_params)

    model.fit(X=X_train, y=y_train)
    return model


def train_adaboost(X_train, y_train, X_val, y_val, args):
    base_params = params['decisiontree']
    base_params['random_state'] = args.seed
    model_params = params['adaboost']
    model_params['random_state'] = args.seed
    model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(**base_params), **model_params)

    if args.tuning:
        search_params = {
            'learning_rate': [1e-3, 1e-2, 0.05, 0.1, 1],
            'n_estimators': [15, 30, 50, 80, 100, 150, 200],
            'loss': ['linear', 'square', 'exponential'],
            'base_estimator__criterion' : ['mse', 'mae'],
            'base_estimator__splitter' : ['best', 'random'],
            'base_estimator__max_depth' : [3, 5, 10, 15, 20],
        }
        best_params = param_tuning(X_train, y_train, model, search_params, args)
        for k, v in best_params.items():
            if 'base_estimator' in k:
                base_params[k] = v
            model_params[k] = v
        model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(**base_params), **model_params)

    model.fit(X=X_train, y=y_train)
    return model


def train_svm(X_train, y_train, X_val, y_val, args):
    model_params = params['svm']
    model = SVR(**model_params)

    if args.tuning:
        search_params = {
            'C': [0.05, 0.1, 1, 2, 5],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'],
            'shrinking' : [True, False]
        }
        best_params = param_tuning(X_train, y_train, model, search_params, args)
        for k, v in best_params.items():
            model_params[k] = v
        model = SVC(**model_params)

    model.fit(X=X_train, y=y_train)
    return model


def param_tuning(X_train, y_train, clf, search_params, args, X_val=None, y_val=None):
    """
    Not working for lgb yet
    """
    logging.info(f'Tuning parameters...')
    # model = GridSearchCV(
    #     lgb,
    #     search_params,
    #     scoring = 'neg_root_mean_squared_error',
    #     cv = 3,
    #     n_jobs = -1,
    #     verbose=False
    # )
    model = RandomizedSearchCV(
        estimator=clf, 
        param_distributions=search_params, 
        n_iter=500,
        scoring='neg_root_mean_squared_error',
        cv=3, n_jobs = 12,
        random_state=args.seed,
        verbose=False
        )

    if args.model_name == 'lgb':
        model.fit(X=X_train, y=y_train, eval_set = (X_val, y_val))
    else:
        model.fit(X_train, y_train)
    best_params = model.best_params_
    extend_res_summary({'GridSearch': best_params})
    logging.info(f'Best parameters: {best_params}')
    return best_params


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


def plot_binscatter(x, y, save_path, nbins = 10, xlabel='actual', ylabel='predicted', ext=''):
    """
    Adapted code from 
    https://stackoverflow.com/questions/15556930/turn-scatter-data-into-binned-data-with-errors-bars-equal-to-standard-deviation
    """
    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)

    plt.plot(x, y, 'bo')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Binscatter Plot of Predicted vs Actual'+ext)
    plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='r-')
    plt.tight_layout()
    plt.savefig(save_path)
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
        elif args.model_name == 'knn':
            model = train_knn(X_train, y_train, X_val, y_val, args)
        elif args.model_name == 'adaboost':
            model = train_adaboost(X_train, y_train, X_val, y_val, args)
        elif args.model_name == 'svm':
            model = train_svm(X_train, y_train, X_val, y_val, args)
        else:
            raise NotImplementedError
        
        # evaluate fold
        pred_train[val_idx] = model.predict(X_val)
        pred_test[fold] = model.predict(X_test)

        if args.target=='resale_price_sqm':
            y_val = [a*b for a,b in zip(y_val,X_val['floor_area_sqm'])]
            pred_train[val_idx] = [a*b for a,b in zip(pred_train[val_idx],X_val['floor_area_sqm'])]
            y_test = [a*b for a,b in zip(y_test,X_test['floor_area_sqm'])]
            pred_test[fold] = [a*b for a,b in zip(pred_test[fold],X_test['floor_area_sqm'])]

        val_rmse = mean_squared_error(y_val, pred_train[val_idx], squared=False)
        test_rmse = mean_squared_error(y_test, pred_test[fold], squared=False)
        logging.info(f'TRAIN: n={len(trn_idx)} | VAL: n={len(val_idx)}, rmse={val_rmse} | TEST: n={len(y_test)}, rmse={test_rmse}')
        extend_res_summary({f'Val_RMSE_{fold}': val_rmse, f'Test_RMSE_{fold}': test_rmse})

        # save model
        joblib.dump(model, f'outs/{args.model_name}/fold{fold}.joblib')

        # save plots
        if args.model_name in ['lgb', 'rf']:
            plot_feat_imp(
                model=model, 
                save_path=f'outs/{args.model_name}/fold{fold}_featimp.png'
                )
        if args.model_name=='lgb':
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
    
    # save frames
    pd.DataFrame({'index': y.index, 'pred': pred_train}).to_csv(f'outs/{args.model_name}/pred_train.csv', index=False)
    pd.DataFrame({'index': y_test.index, 'pred': pred_test['mean']}).to_csv(f'outs/{args.model_name}/pred_test.csv', index=False)

    # save plots
    plot_scatters(
        actual=y, 
        pred=pred_train, 
        save_path=f'outs/{args.model_name}/val.png', 
        ext=f' in Val OVERALL'
        )
    plot_scatters(
        actual=y_test, 
        pred=pred_test['mean'], 
        save_path=f'outs/{args.model_name}/test.png', 
        ext=f' in Test OVERALL'
        )
    plot_binscatter(
        x=y, 
        y=pred_train, 
        nbins= 10000,
        save_path=f'outs/{args.model_name}/val_binned.png', 
        ext=f' in Val OVERALL'
        )
    plot_binscatter(
        x=y_test, 
        y=pred_test['mean'], 
        nbins= 10000,
        save_path=f'outs/{args.model_name}/test_binned.png', 
        ext=f' in Test OVERALL'
        )


def predict(df, args):
    logging.info(f'-- predicting')
    # format dataset
    df = format_data(df, args, encode_categorical=False if args.model_name=='lgb' else True)
    # predict folds
    pred_new = defaultdict(list)
    for fold in range(args.folds):
        model = joblib.load(f'outs/{args.model_name}/fold{fold}.joblib')
        # feats = model._Booster.feature_name()
        # print([i for i in df.columns if i not in feats])
        # print(df.columns)
        pred_new[fold] = model.predict(df)
        if args.target=='resale_price_sqm':
            pred_new[fold] = [a*b for a,b in zip(pred_new[fold],df['floor_area_sqm'])]
    pred_new = pd.DataFrame(pred_new)
    pred_new['mean'] = pred_new.mean(axis=1)
    pred_new = pred_new.reset_index(drop=True)
    pred_new.to_csv(f'outs/{args.model_name}/preds.csv', index=False)
    # submission format
    submission = pd.DataFrame()
    submission['Id'] = pred_new.index
    submission['Predicted'] = pred_new['mean']
    submission.to_csv(f'outs/{args.model_name}/submission.csv', index=False)