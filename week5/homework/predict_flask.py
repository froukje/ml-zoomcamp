import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = "model1.bin"
dv_file = "dv.bin"

with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)

with open(dv_file, "rb") as dv_in:
    dv = pickle.load(dv_in)

app = Flask("churn")

@app.route("/predict", methods=["POST"])
def predict():
    # input is JSON, convert it to python dict
    customer = request.get_json()

    # convert to feature matrix
    X = dv.transform([customer])

    y_pred = model.predict_proba(X)[0, 1]
    churn = (y_pred >= 0.5)

    result = {
            "churn_propability": float(y_pred),
            "churn": bool(churn)
            }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
