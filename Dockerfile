FROM tensorflow/tensorflow:2.11.0-gpu

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN python -m pip install "setuptools==65.5.0" wheel

# Install all Python deps (we will re-pin TF/TFP to tested versions)
RUN pip install --no-cache-dir -r requirements.txt

# Ensure tested TensorFlow / TFP versions (matching README)
RUN pip install --no-cache-dir \
    "tensorflow==2.11.0" \
    "tensorflow-probability==0.19.0"

# Copy project code
COPY . .

# Default command: drop into shell; Slurm/Apptainer or docker run
# will override with the actual script (e.g., train_sota_models.py).
CMD ["/bin/bash"]

