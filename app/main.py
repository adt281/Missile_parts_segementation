from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import base64
import logging
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

# Define the device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a transformation pipeline for the input image
transform = transforms.Compose([
    transforms.ToTensor(),
])

class PredictionOut(BaseModel):
    prediction: str
    image_with_prediction: str  # Base64 encoded image

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
def home():
    return {"health_check": "OK"}

# Define custom classes
class_names = ["missile-pach", "body", "fin", "missile", "nose", "tail", "wing"]

# Register custom dataset with unique name and metadata
DatasetCatalog.register("custom_dataset", lambda: [])
MetadataCatalog.get("custom_dataset").thing_classes = class_names

@app.post("/segment_image")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        logger.info(f"Segment image shape: {image_np.shape}")

        # Configure the predictor
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "/app/model/model_final.pth"  # Path to your trained model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = str(device)  # Ensure the device is set correctly as a string
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Set this to the number of classes your model was trained on

        predictor = DefaultPredictor(cfg)

        # Run inference
        outputs = predictor(image_np)
        logger.info(f"Model outputs: {outputs}")

        # Extract instances and class IDs
        instances = outputs["instances"].to("cpu")
        class_ids = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        masks = instances.pred_masks.numpy()

        # Map class IDs to class names
        detected_classes = [MetadataCatalog.get("custom_dataset").thing_classes[int(class_id)] for class_id in class_ids]
        logger.info(f"Detected classes: {detected_classes}")

        # Initialize a dictionary to store the maximum instance for each class and lists for "fin" and "wing"
        max_instances = {}
        fin_instances = []
        wing_instances = []

        # Loop through each instance
        for i in range(len(instances)):
            pred_class = instances.pred_classes[i].item()
            score = instances.scores[i].item()

            class_name = class_names[pred_class]

            # Separate instances of "fin" and "wing"
            if class_name == "fin":
                fin_instances.append(instances[i:i+1])
            elif class_name == "wing":
                wing_instances.append(instances[i:i+1])
            else:
                # Check if we already have an instance for this class
                if pred_class in max_instances:
                    # If current instance has a higher score, update it
                    if score > max_instances[pred_class]['score']:
                        max_instances[pred_class] = {
                            'instance': instances[i:i+1],
                            'score': score
                        }
                else:
                    max_instances[pred_class] = {
                        'instance': instances[i:i+1],
                        'score': score
                    }

        # Visualize and display the maximum instance for each class
        v = Visualizer(image_np[:, :, ::-1], MetadataCatalog.get("custom_dataset"), scale=1.2)

        combined_image = v.output.get_image()

        # Visualize and display all instances of "fin"
        for fin_instance in fin_instances:
            out = v.draw_instance_predictions(fin_instance)
            combined_image = out.get_image()

        # Visualize and display all instances of "wing"
        for wing_instance in wing_instances:
            out = v.draw_instance_predictions(wing_instance)
            combined_image = out.get_image()

        for key in max_instances:
            max_instance = max_instances[key]['instance']
            out = v.draw_instance_predictions(max_instance)
            combined_image = out.get_image()

        # Create a pixel-wise class data array
        class_map = np.zeros(image_np.shape[:2], dtype=np.uint8)  # Assuming class IDs are small integers

        # Assign pixels their classes based on the max instances
        for key, value in max_instances.items():
            mask = value['instance'].pred_masks[0].numpy()  # Assuming only one mask per instance
            class_map[mask] = key + 1  # Offset by 1 to avoid background being labeled as 0

        # Assign pixels for all "fin" instances
        for fin_instance in fin_instances:
            mask = fin_instance.pred_masks[0].numpy()  # Assuming only one mask per instance
            class_map[mask] = class_names.index("fin") + 1  # Get the index for "fin" and offset by 1

        # Assign pixels for all "wing" instances
        for wing_instance in wing_instances:
            mask = wing_instance.pred_masks[0].numpy()  # Assuming only one mask per instance
            class_map[mask] = class_names.index("wing") + 1  # Get the index for "wing" and offset by 1

        # Convert the image to base64
        mask_image = Image.fromarray(combined_image[:, :, ::-1])
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Convert class map to base64
        class_map_image = Image.fromarray(class_map)
        buffered_class_map = io.BytesIO()
        class_map_image.save(buffered_class_map, format="PNG")
        class_map_base64 = base64.b64encode(buffered_class_map.getvalue()).decode('utf-8')

        # Populate detected_classes using max_instances, fin_instances, and wing_instances
        detected_classes_names = set()
        for key in max_instances:
            detected_classes_names.add(class_names[key])
        if fin_instances:
            detected_classes_names.add("fin")
        if wing_instances:
            detected_classes_names.add("wing")
        detected_classes_names = list(detected_classes_names)
        return {"segmented_image": mask_base64, "classes": detected_classes_names, "class_map": class_map_base64}

    except Exception as e:
        logger.error(f"Error in segment_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
