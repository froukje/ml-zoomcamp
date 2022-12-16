import requests
#url = 'http://localhost:9696/predict'
url = 'http://localhost:8080/predict'

data = {'url': 'http://bit.ly/mlbookcamp-pants'}

result = requests.post(url, json=data).json()
print(result)
