# Use the official Bitnami MLflow image as the base image
FROM bitnami/mlflow:latest

# Set the working directory
WORKDIR /opt/mlflow

# Switch to root user to install system dependencies
USER root

# Ensure the /var/lib/apt/lists/partial directory exists, clean the apt lists, update and install dependencies for h5py and build-essential
RUN mkdir -p /var/lib/apt/lists/partial && apt-get clean && apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libhdf5-dev \
    build-essential \
    supervisor

# Upgrade pip
RUN pip install --upgrade pip

# Switch back to the default user
USER 1001

# Copy the requirements.txt file into the container
COPY model_dir/model/requirements.txt /opt/mlflow/requirements.txt

# Install Python dependencies from the requirements.txt file
RUN pip install -r /opt/mlflow/requirements.txt

# Ensure gevent is installed
RUN pip install gevent

# Copy the model directory
COPY model_dir/model /opt/ml/model

# Copy the supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy the mlflow_test.py script
COPY mlflow_test.py /opt/mlflow/mlflow_test.py

# Set environment variables for MLflow
ENV MLFLOW_DISABLE_ENV_CREATION=True
ENV ENABLE_MLSERVER=False
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"

# Expose the ports for MLflow and the UI
EXPOSE 8000
EXPOSE 5000

# Entry point for the Docker container
ENTRYPOINT ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
