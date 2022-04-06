#! /usr/bin/env python3

import pickle
import datetime
import pandas as pd

model_file= 'xgb_n_est=250_eta=0.004_gamma=0.015_alpha=0.000_max_depth=35_min_child_weight=11.000.bin'

with open(model_file, "rb") as f_in:
    scaler, dv, model = pickle.load(f_in)

avocado = {"Unnamed: 0": [11],
           "Date": ["2018-01-07"],
           "Total Volume": [17489.58],
           4046: [2894.77],
           4225: [2356.13],
           4770: [224.53],
           "Total Bags": [12014.15],
           "Small Bags": [11988.14],
           "Large Bags": [26.01],
           "XLarge Bags": [0.0],
           "type": ["organic"],
           "year": [2018],
           "region": ["WestTexNewMexico"]}

# save as dataframe
df = pd.DataFrame.from_dict(avocado)
print(df)

y_true = 1.62

# preprocessing

def preprocess(df, scaler):
    """ Assume incoming data has the form as given in the csv file
        1. delete "Unnamed 0"
        2. make column names consistent
        3. define used columns
        4. normalize 
        5. turn into dictionary
    """

    del df["Unnamed: 0"] 
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    train_vars = ["total_bags", "total_volume", "type", "region", "year"]
    df_scaled = pd.DataFrame(scaler.transform(df[["total_bags","total_volume"]]),
                         columns=["total_bags","total_volume"])

    df_preprocessed = pd.concat([df[["type", "region", "year"]], df_scaled], axis=1)
    # turn into dict
    preprocessed_data = df_preprocessed[train_vars].to_dict(orient="records")
    return preprocessed_data


# turn avocado into a feature matrix
preprocessed_data = preprocess(df, scaler)
print(preprocessed_data)
X = dv.transform(preprocessed_data)

# price of this avocado
y_pred = model.predict(X)
print(f"input: {avocado}")
print(f"predicted price: {y_pred[0]}")
