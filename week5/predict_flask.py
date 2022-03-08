import pickle
from flask import Flask
from flask import request
from flask import jsonify 

model_file = "model_C=1.0.bin"

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("churn")

# we want to send information about the customer, so we need the "POST" method
@app.route("/predict", methods=["POST"])
def predict():
    # input is expected as JSON and converted in Python dictionary
    customer = request.get_json()
    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5

    result = {
            "churn_probability": float(y_pred),
            "churn": bool(churn)
            }

    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
