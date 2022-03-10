FROM agrigorev/zoomcamp-model:3.8.12-slim

# Install all the dependencies from the Pipenv file:
RUN pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

# Copy your Flask script
COPY ["predict_flask_q6.py", "./"]

# Run it with gunicorn
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_flask_q6:app"]
