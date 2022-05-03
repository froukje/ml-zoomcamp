import tflite_runtime.interpreter as tflite
#import tensorflow.lite as tflite
import numpy as np
import os
from io import BytesIO
from urllib import request
from PIL import Image

#cats_dogs_model = "cats-vs-dogs-v2.tflite"
MODEL_NAME = os.getenv('MODEL_NAME', "cats-vs-dogs.tflite")

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    # normalize images
    x = np.array(x, dtype="float32")
    x = x/255.
    return x


# load the model
interpreter = tflite.Interpreter(model_path=MODEL_NAME)
# load the weights (in keras this is automatically)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# url = 'https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg'
def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))

    x = preprocess_input(img)
    X = np.array([x]) # batch size = 1

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return float(preds[0, 0])

def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    result = {'prediction': pred}

    return result
