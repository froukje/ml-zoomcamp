import requests
#url = 'http://localhost:9696/predict'
#url = 'http://localhost:8080/predict'
url = 'http://af6a03b14e6834fa9b77454bfbdf400a-1093699068.eu-west-1.elb.amazonaws.com/predict'

data = {'url': 'http://bit.ly/mlbookcamp-pants'}

result = requests.post(url, json=data).json()
print(result)
