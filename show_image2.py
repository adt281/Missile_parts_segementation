import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import json

# Load the JSON file containing the segmentation data
with open("C:\\Users\\adity\\Downloads\\response_1722679693254.json", 'r') as file:
    data = json.load(file)

# Decode the base64 image with segmentation mask
image_data = base64.b64decode(data["segmented_image"])
image_with_mask = Image.open(BytesIO(image_data))

# Load the original image
file_path = "C:\\Users\\adity\\Downloads\\brahmos-new.jpg"
original_image = Image.open(file_path)

# Display the original image and the image with mask side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(original_image)
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(image_with_mask)
ax2.set_title('Image with Segmentation Mask')
ax2.axis('off')

plt.tight_layout()
plt.show()
