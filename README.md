🚀 Missile Part Detection API (FastAPI + Detectron2 + Docker)
-This project hosts a custom-trained Detectron2 Mask R-CNN model for missile part detection as a Dockerized FastAPI service on AWS EC2.

🧠 What it does
-Serves a trained instance segmentation model (Mask R-CNN via Detectron2) through a FastAPI REST endpoint
-Accepts an image of a missile via POST request
-Returns a JSON containing:
-Masked image (base64-encoded)
-Detected class names
-Pixel-wise segmentation data

☁️ Deployment Details
-The entire app (FastAPI backend, model loading, prediction logic) runs inside a Docker container
-The .pth model file (from custom training) should be placed in the model/ directory before starting the container
-When deployed on AWS EC2, the container exposes a public API port (e.g., 0.0.0.0:8000) that can be connected to a frontend GUI for real-time inference
-Note: The .pth model file is excluded from this repository due to size constraints. You must supply your own trained weights.

🛠 Stack
-FastAPI for API
-Detectron2 for instance segmentation (Mask R-CNN)
-Docker for containerization
-AWS EC2 for cloud hosting

