import os

from PIL.Image import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

ROOT_TRAINING_DIR = ""
MODEL_PATH = ""

INPUT_FOLDER = ""
OUTPUT_FOLDER = ""


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Helper function to find leaf node folders
def find_leaf_node_classes(root_dir):
    leaf_node_classes = []
    class_to_idx = {}

    for root, dirs, files in os.walk(root_dir):
        if (
            files and not dirs
        ):  # This is a leaf node if it has files but no subdirectories
            relative_path = os.path.relpath(root, root_dir)
            leaf_node_classes.append(relative_path)

    # Assign indices to each detected class
    leaf_node_classes = sorted(leaf_node_classes)
    class_to_idx = {leaf_class: idx for idx, leaf_class in enumerate(leaf_node_classes)}

    return class_to_idx


# Custom dataset to work with nested structure
class NestedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = find_leaf_node_classes(root_dir)
        self._prepare_dataset()

    def _prepare_dataset(self):
        # Traverse the directory and prepare a list of (image_path, class_index) tuples
        for class_path, class_idx in self.class_to_idx.items():
            full_class_path = os.path.join(self.root_dir, class_path)
            for file_name in os.listdir(full_class_path):
                if file_name.endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(full_class_path, file_name), class_idx)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, class_idx


# Load a pre-trained model (e.g., ResNet) and modify it
def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    # Replace the final layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Function to identify leaf node folders (folders that contain images directly)
def get_leaf_node_classes(root_dir):
    leaf_node_classes = []
    for root, dirs, files in os.walk(root_dir):
        if (
            files and not dirs
        ):  # Check if it's a leaf node (contains files but no subdirectories)
            relative_path = os.path.relpath(root, root_dir)
            leaf_node_classes.append(relative_path)
    return sorted(leaf_node_classes)  # Sort to ensure consistent class order
