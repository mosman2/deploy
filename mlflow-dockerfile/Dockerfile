# Use a slim version of Python as the base image
FROM python:3.11.6-slim

# Install system dependencies
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget curl nginx ca-certificates bzip2 build-essential cmake git-core \
    pkg-config libhdf5-dev

# Install pyenv dependencies
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

# Set environment variables for pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

# Install pyenv
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT
RUN $PYENV_ROOT/bin/pyenv install 3.11.6
RUN $PYENV_ROOT/bin/pyenv global 3.11.6

# Install virtualenv and gevent
RUN pip install virtualenv gevent

# Set the working directory
WORKDIR /opt/mlflow

# Install MLflow
RUN pip install mlflow==2.13.1

# Copy the model directory
COPY model_dir/model /opt/ml/model

# Install model dependencies
RUN python -c "from mlflow.models import container as C; C._install_pyfunc_deps('/opt/ml/model', install_mlflow=False, enable_mlserver=False, env_manager='local');"

# Set environment variables for MLflow
ENV MLFLOW_DISABLE_ENV_CREATION=True
ENV ENABLE_MLSERVER=False
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"

# Clean up apt cache to reduce image size
RUN rm -rf /var/lib/apt/lists/*

# Entrypoint for the Docker container
ENTRYPOINT ["mlflow", "models", "serve", "-m", "/opt/ml/model", "-p", "8000", "--host", "0.0.0.0"]
