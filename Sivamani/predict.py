import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import os
from PIL import Image

# --- U-Net Model Definition (Must be the same as in train.py) ---
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# --- Prediction Function ---
def predict(image_path, model_path, output_path):
    """
    Loads a model, predicts a mask for an image, and saves the result.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model and load the trained weights
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded from '{model_path}'")

    # Load and preprocess the input image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at '{image_path}'")
        return

    # Resize image to the same size as the training images
    IMAGE_SIZE = (256, 256)
    transform = T.Compose([
        T.Resize(IMAGE_SIZE), 
        T.ToTensor(),
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension
    
    print(f"Predicting mask for '{image_path}'...")
    
    with torch.no_grad():
        output = model(input_tensor)
        
    # Apply sigmoid to get probabilities between 0 and 1
    probabilities = torch.sigmoid(output)
    
    # Threshold the probabilities to get a binary mask
    predicted_mask = (probabilities > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
    
    # Resize the predicted mask back to the original image size for easier viewing
    original_size = image.size
    predicted_mask_uint8 = (predicted_mask * 255).astype(np.uint8)
    final_mask = cv2.resize(predicted_mask_uint8, original_size, interpolation=cv2.INTER_NEAREST)

    # Save the predicted mask
    cv2.imwrite(output_path, final_mask)
    print(f"Prediction complete. Mask saved to '{output_path}'")

    # --- New code to check for a detected spill based on pixel count ---
    MIN_SPILL_PIXELS = 100 # Adjust this value as needed
    white_pixels = np.count_nonzero(final_mask)
    
    if white_pixels > MIN_SPILL_PIXELS:
        print(f"\n*** Spill Detected! ({white_pixels} pixels) ***")
    else:
        print(f"\n*** No Spill Detected. ({white_pixels} pixels) ***")
    # --- End of new code ---

if __name__ == "__main__":
    # Define paths
    IMAGE_TO_PREDICT_PATH = "dataset/test/images/Oil (1220).jpg" # Example test image
    TRAINED_MODEL_PATH = "model.pth"
    OUTPUT_MASK_PATH = "predicted/predicted_mask.png"
    
    # Run the prediction
    predict(IMAGE_TO_PREDICT_PATH, TRAINED_MODEL_PATH, OUTPUT_MASK_PATH)