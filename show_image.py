import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import json
import numpy as np

# Load the JSON file containing the segmentation data
with open("C:\\Users\\adity\\Downloads\\response_1722071568179.json", 'r') as file:
    data = json.load(file)

# Decode the base64 image with segmentation mask
image_data = base64.b64decode(data["segmented_image"])
image_with_mask = Image.open(BytesIO(image_data))

# Load the original image
file_path = "C:\\Users\\adity\\Downloads\\brahmos-new.jpg"
original_image = Image.open(file_path)

# Decode the base64 class map
class_map_data = base64.b64decode(data["class_map"])
class_map_image = Image.open(BytesIO(class_map_data))
class_map = np.array(class_map_image)

# Resize class_map to match the dimensions of the original image
class_map = np.array(class_map_image.resize(original_image.size, resample=Image.NEAREST))

# Define your class names
class_names = ["missile-pach", "body", "fin", "missile", "nose", "tail", "wing"]

# Create figures with subplots for general instances
num_classes = len(class_names) - 2  # Exclude "fin" and "wing" from the count
fig, axs = plt.subplots(1, num_classes + 2, figsize=(5 * (num_classes + 2), 6))

# Display the image with mask
axs[0].imshow(image_with_mask)
axs[0].axis('off')
axs[0].set_title('Image with Mask')

# Display the original image
axs[1].imshow(original_image)
axs[1].axis('off')
axs[1].set_title('Original Image')

# Convert the original image to numpy array
original_image_np = np.array(original_image)

# Initialize subplot index for general instances
plot_index = 2

# Loop through each class and display the isolated instance
for i, class_name in enumerate(class_names):
    # Skip "fin" and "wing" classes in this loop
    if class_name in ["fin", "wing"]:
        continue
    
    # Get the index of the target class
    target_class_index = i + 1  # Assuming class_map indices start from 1

    # Create a mask for the target class
    class_mask = (class_map == target_class_index)

    # Apply the mask to the original image
    class_image_np = np.zeros_like(original_image_np)
    class_image_np[class_mask] = original_image_np[class_mask]

    # Convert the result back to an Image
    class_image = Image.fromarray(class_image_np)

    # Display the isolated class image
    axs[plot_index].imshow(class_image)
    axs[plot_index].axis('off')
    axs[plot_index].set_title(f'Isolated "{class_name}" Instance')
    
    # Increment the plot index
    plot_index += 1

# Show the plot
plt.tight_layout()
plt.show()

# Function to display all instances of a given class
def display_all_instances(class_name, axs):
    class_index = class_names.index(class_name) + 1
    class_mask = (class_map == class_index)
    class_image_np = np.zeros_like(original_image_np)
    class_image_np[class_mask] = original_image_np[class_mask]
    class_image = Image.fromarray(class_image_np)
    axs.imshow(class_image)
    axs.axis('off')
    axs.set_title(f'Isolated "{class_name}" Instances')

# Create a figure for "fin" instances
fig_fin, axs_fin = plt.subplots(1, 1, figsize=(5, 6))
display_all_instances("fin", axs_fin)
plt.tight_layout()
plt.show()

# Create a figure for "wing" instances
fig_wing, axs_wing = plt.subplots(1, 1, figsize=(5, 6))
display_all_instances("wing", axs_wing)
plt.tight_layout()
plt.show()
