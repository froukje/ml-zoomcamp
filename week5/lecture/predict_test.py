#!/usr/bin/env python
# coding: utf-8

import requests

host = "churn-serving-env.eba-zkrywmjr.eu-west-1.elasticbeanstalk.com"
url = f"http://{host}/predict"
#url = "http://localhost:9696/predict"

customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}


# send this customer in a post request
response = requests.post(url, json=customer).json()


id = "xyz-123"
if response["churn"] == True:
    print(response)
    print(f"sending promo e-mail to {id}")

