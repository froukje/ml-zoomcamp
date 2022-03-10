import pickle


model_file = "../../model/model1.bin"
dv_file = "../../model/dv.bin"

with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)

with open(dv_file, "rb") as dv_in:
    dv = pickle.load(dv_in)

customer = {
    "contract": "two_year",
    "tenure": 12,
    "monthlycharges": 19.7,
}

# turn this customer into a feature matrix
X = dv.transform([customer])

# propability that this customer churns
y_pred = model.predict_proba(X)[0,1]

print(f"input: {customer}")
print(f"churn probability: {y_pred}")

