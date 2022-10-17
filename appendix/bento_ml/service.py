import bentoml
from bentoml.io import JSON

# reference a model, the tag 'latest' will pull the latest model
model_ref = bentoml.xgboost.get("credit_risk_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

# access the model
model_runner = model_ref.to_runner()

# create the service
svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

# the decoration allows us to call this endpoint using 'rest' and 'curl'
@svc.api(input=JSON(), output=JSON())
def classify(application_data):
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    print(prediction)

    result = prediction[0]
    if result > 0.5:
        return { "status": "DECLINED" }
    elif result > 0.25:
        return { "status": "MAYBE" }
    else
        return { "status": "APPROVED" }
