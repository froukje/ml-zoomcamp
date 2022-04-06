#! /usr/bin/env python3

import pickle
import pandas as pd
from flask import Flask
from flask import request, jsonify

model_file= 'xgb_n_est=250_eta=0.004_gamma=0.015_alpha=0.000_max_depth=35_min_child_weight=11.000.bin'

with open(model_file, "rb") as f_in:
    scaler, dv, model = pickle.load(f_in)

app = Flask("avocado_price")

def preprocess(json_data, scaler):
    """ Assume incoming data has the form as given in the csv file
        1. delete "Unnamed 0"
        2. make column names consistent
        3. define used columns
        4. normalize 
        5. turn into dictionary
    """

    df = pd.DataFrame.from_dict(json_data)
    del df["Unnamed: 0"] 
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    train_vars = ["total_bags", "total_volume", "type", "region", "year"]
    df_scaled = pd.DataFrame(scaler.transform(df[["total_bags","total_volume"]]),
                         columns=["total_bags","total_volume"])

    df_preprocessed = pd.concat([df[["type", "region", "year"]], df_scaled], axis=1)
    # turn into dict
    preprocessed_data = df_preprocessed[train_vars].to_dict(orient="records")
    return preprocessed_data



# we want to send information, need "POST" method
@app.route("/predict", methods=['POST'])
def predict():
    # input is expected as JSON, convert to python dict
    avocado = request.get_json()
    print("avocado", avocado)
    
    # preprocessing
    preprocessed_data = preprocess(avocado, scaler)
    X = dv.transform(preprocessed_data)

    # price of this avocado
    y_pred = model.predict(X)
    result = {"predicted_price": float(y_pred[0])}
    
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
