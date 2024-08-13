# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for CUDA architecture
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Install PyTorch and other dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Uvicorn and FastAPI
RUN pip install uvicorn fastapi

# Install Detectron2
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'


# Install python-multipart
RUN pip install python-multipart

# Install OpenCV
RUN pip install opencv-python

# Set the working directory
WORKDIR /app

# Copy your application code into the container
COPY ./app /app/app

# Ensure the model directory exists and copy the model files
RUN mkdir -p /app/model
COPY ./app/model/model_final.pth /app/model/model_final.pth
COPY ./app/model/config.yaml /app/model/config.yaml

# Command to run your FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
