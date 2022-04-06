FROM continuumio/miniconda3 
#FROM python:3.8.12-slim

RUN apt-get update

WORKDIR /app

# Make RUN commands use 'bash --login'
SHELL ["/bin/bash", "--login", "-c"]

# Create the environment
COPY ml-zoomcamp.yml .
RUN conda env create -f ml-zoomcamp.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "ml-zoomcamp", "/bin/bash", "-c"]

# Demonstrate the evironment is activated
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# Copy the python script
COPY ["predict_flask.py", "xgb_n_est=250_eta=0.004_gamma=0.015_alpha=0.000_max_depth=35_min_child_weight=11.000.bin", "./"]

# Expose the port
EXPOSE 9696

# Define entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ml-zoomcamp", "gunicorn", "--bind=0.0.0.0:9696", "predict_flask:app"]
