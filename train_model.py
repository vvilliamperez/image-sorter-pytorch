from torch import nn
from torch.utils.data import DataLoader
import torch
from torchvision import models

from common import (
    transform,
    ROOT_TRAINING_DIR,
    find_leaf_node_classes,
    NestedImageDataset,
    MODEL_PATH,
)

print(torch.cuda.get_device_name(0))

train_transform = transform

class_to_idx = find_leaf_node_classes(ROOT_TRAINING_DIR)

# Create the custom dataset and dataloader
train_dataset = NestedImageDataset(
    root_dir=ROOT_TRAINING_DIR, transform=train_transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Automatically determine the number of classes
num_classes = len(class_to_idx)
print(f"Number of classes: {num_classes}")

# Set up the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# Save the trained model
torch.save(model.state_dict(), MODEL_PATH)
