import torch
from PIL import Image
import io
import numpy as np
import base64
from io import BytesIO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

__version__ = "1.0.0"

# Load Detectron2 model configuration and weights
cfg = get_cfg()
cfg.merge_from_file("/app/model/config.yaml")  # Path to your model config file
cfg.MODEL.WEIGHTS = "/app/model/model_final.pth"  # Path to your model weights file
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

predictor = DefaultPredictor(cfg)

async def segment_image(file: UploadFile) -> str:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    
    outputs = predictor(image_np)
    
    # Use Visualizer to draw the segmentation mask on the image
    v = Visualizer(image_np[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Convert the image to base64
    mask_image = Image.fromarray(out.get_image()[:, :, ::-1])
    buffered = BytesIO()
    mask_image.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return mask_base64
