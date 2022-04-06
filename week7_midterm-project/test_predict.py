import requests

url = "http://10.193.167.24:9696/predict"

avocado = {
        "Unnamed: 0": [11],
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
        "region": ["WestTexNewMexico"]
}

print(f"Predicted Price {requests.post(url, json=avocado).json()['predicted_price']:.3f}")
