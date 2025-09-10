import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from PIL import Image

# --- U-Net Model Definition ---
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 1, 1) # Output 1 channel for binary mask

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# --- Custom Dataset Class ---
class OilSpillDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.mask_filenames = [f for f in os.listdir(masks_dir) if f.endswith('.png')]

        # Ensure image and mask lists are sorted to maintain correspondence
        self.image_filenames.sort()
        self.mask_filenames.sort()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        # Load image (RGB)
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Load mask (Grayscale)
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Ensure mask is binary (0 or 1)
        mask[mask > 0] = 1.0

        return image, mask

# --- Training Loop ---
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters (you can tune these)
    NUM_EPOCHS = 100
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-3
    IMAGE_SIZE = (256, 256)

    # Define paths
    DATASET_IMAGES_DIR = "dataset/train/images"
    DATASET_MASKS_DIR = "dataset/train/binary_masks"
    MODEL_SAVE_PATH = "model.pth"

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    train_dataset = OilSpillDataset(DATASET_IMAGES_DIR, DATASET_MASKS_DIR, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

    print("Training finished. Saving model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    train_model()