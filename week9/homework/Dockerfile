FROM agrigorev/zoomcamp-cats-dogs-lambda:v2

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

COPY Lecture_9_homework.py .
ENV MODEL_NAME=cats-vs-dogs-v2.tflite

# this command is to tell lambda, where to find the lambda function
CMD ["Lecture_9_homework.lambda_handler"]

