import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import os
import shutil

from common import (
    transform,
    ROOT_TRAINING_DIR,
    get_leaf_node_classes,
    OUTPUT_FOLDER,
    INPUT_FOLDER,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 625)  # Adjust the final layer
model.to(device)


model.load_state_dict(torch.load())
model.eval()  # Set model to evaluation mode

# Automatically get class names from the training directory
# Root directory used during training
class_names = get_leaf_node_classes(ROOT_TRAINING_DIR)


def classify_and_sort_image(image_path, model, target_folder, class_names):
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    img = img.to(next(model.parameters()).device)

    # Make prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    # Move the image to the appropriate folder
    destination = os.path.join(target_folder, predicted_class)
    os.makedirs(destination, exist_ok=True)
    shutil.move(image_path, destination)

    print(f"Moved {image_path} to {destination}")


def watch_and_sort(folder_to_watch, model, target_folder, class_names):
    for filename in os.listdir(folder_to_watch):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Check if file is an image
            classify_and_sort_image(
                os.path.join(folder_to_watch, filename),
                model,
                target_folder,
                class_names,
            )


# Example usage:
watch_and_sort(INPUT_FOLDER, model, OUTPUT_FOLDER, class_names)
