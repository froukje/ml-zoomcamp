#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

import optuna

import pickle

seed = 42
np.random.seed(seed)

# read the data
df = pd.read_csv("../data/avocado.csv")

# data preparation
## sort by date
df = df.sort_values(by="Date").reset_index(drop=True)
## delete column "Unnamed"
del df["Unnamed: 0"]
## make column names consistent: lower case and no spaces
df.columns = df.columns.str.lower().str.replace(" ","_")
## define categorical and numerical columns
categorical = ["type", "region", "year"]
numerical = ["total_volume", "4046", "4225", "4770", "total_bags", "small_bags", "xlarge_bags"]

## train, test split (consider time)
df_train_full, df_test = train_test_split(df, test_size=0.2, shuffle=False, random_state=seed)

df_train_full = df_train_full.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

## define y
y_train_full = df_train_full["averageprice"].values
y_test = df_test["averageprice"].values

## delete "averageprice" from the dataframe
del df_train_full["averageprice"]
del df_test["averageprice"]

train_vars = ["total_bags", "total_volume"] + categorical

## scaling
scaler = StandardScaler()
df_train_full_num = pd.DataFrame(scaler.fit_transform(df_train_full[["total_bags", "total_volume"]]),
                            columns=["total_bags", "total_volume"])
df_test_num = pd.DataFrame(scaler.transform(df_test[["total_bags", "total_volume"]]),
                            columns=["total_bags", "total_volume"])

df_train_full_new = pd.concat([df_train_full[categorical], df_train_full_num], axis=1)
df_test_new = pd.concat([df_test[categorical], df_test_num], axis=1)
df_train_full = df_train_full_new
df_test = df_test_new

train_full_dicts = df_train_full[train_vars].to_dict(orient="records")
test_dicts = df_test[train_vars].to_dict(orient="records")
dv = DictVectorizer(sparse=False)
X_train_full = dv.fit_transform(train_full_dicts)
X_test = dv.transform(test_dicts)


def objective(trial):

    reg_name = trial.suggest_categorical("regressor", ["RandomForest", "xgb"])

    if reg_name == "RandomForest":
        n_estimators = trial.suggest_int('n_estimators', 50, 300, step=10)
        max_depth = trial.suggest_int('max_depth', 5, 100, step=5)
        min_samples_split= trial.suggest_int('min_samples_split', 2, 10)
        model = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      n_jobs=-1)
    else:
        n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
        eta = trial.suggest_float('eta', 0.001, 0.03)
        gamma = trial.suggest_float('gamma', 0.01, .1, log=True)
        alpha = trial.suggest_float('alpha', 1e-8, 0.5, log=True)
        max_depth = trial.suggest_int('max_depth', 10, 50, step=5)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 15)
        
        model = xgb.XGBRegressor(n_estimators=n_estimators,
                             eta=eta,
                             gamma=gamma,
                             alpha=alpha,
                             max_depth=max_depth,
                             min_child_weight=min_child_weight,
                             nthread=8,
                             verbosity=1
                            )
        
    
    score = cross_val_score(model, X_train_full, y_train_full, 
                            n_jobs=-1, cv=5, 
                            scoring='neg_root_mean_squared_error')
    rmse = score.mean()
    return abs(rmse)

study = optuna.create_study()
study.optimize(objective, n_trials=100)

## Final Model
# Create an instance with tuned hyperparameters
if study.best_params['regressor'] == 'RandomForest':
    optimised_model = RandomForestRegressor(max_depth = study.best_params['max_depth'],
                                     min_samples_split = study.best_params['min_samples_split'],
                                     n_estimators = study.best_params['n_estimators'],
                                     n_jobs=-1)

else:
    optimised_model = xgb.XGBRegressor(n_estimators=study.best_params['n_estimators'],
                             eta=study.best_params['eta'],
                             gamma=study.best_params['gamma'],
                             alpha=study.best_params['alpha'],
                             max_depth=study.best_params['max_depth'],
                             min_child_weight=study.best_params['min_child_weight'],
                             nthread=8,
                             verbosity=1
                            )

optimised_model.fit(X_train_full ,y_train_full)

y_pred_train = optimised_model.predict(X_train_full)
y_pred = optimised_model.predict(X_test)
rmse_train = mean_squared_error(y_pred_train, y_train_full, squared=False)
rmse_val = mean_squared_error(y_pred, y_test, squared=False)
print(f"rmse training: {rmse_train:.3f}\t rmse test: {rmse_val:.3f}")

# save the model
reg = study.best_params['regressor'] 
if reg=='RandomForest':
    output_file=f"{reg}__n_est={study.best_params['n_estimators']}"\
                f"_max_depth={study.best_params['max_depth']}" \
                f"_min_sample_weight={study.best_params['min_samples_split']:.3f}.bin"
else:
    output_file=f"{reg}_n_est={study.best_params['n_estimators']}_eta={study.best_params['eta']:.3f}" \
                f"_gamma={study.best_params['gamma']:.3f}_alpha={study.best_params['alpha']:.3f}" \
                f"_max_depth={study.best_params['max_depth']}" \
                f"_min_child_weight={study.best_params['min_child_weight']:.3f}.bin"


with open(output_file, "wb") as f_out:
    # save model, scaler and vectorization
    pickle.dump((scaler, dv, optimised_model), f_out)
