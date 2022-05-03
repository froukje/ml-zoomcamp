import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://upload.wikimedia.org/wikipedia/commons/1/18/Vombatus_ursinus_-Maria_Island_National_Park.jpg'}

print(requests.post(url, json=data).text)
try:
    result = requests.post(url, json=data).json()
    print(result)
except ValueError:
    print("Response content is not valid JSON")
