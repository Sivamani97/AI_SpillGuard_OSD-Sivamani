import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import os
from PIL import Image

# Assuming model.py is in the same directory or accessible
from model import UNet

# --- Prediction Function ---
def predict(image_path, model_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model and load the trained weights
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded from '{model_path}'")

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at '{image_path}'")
        return

    IMAGE_SIZE = (256, 256)
    transform = T.Compose([
        T.Resize(IMAGE_SIZE), 
        T.ToTensor(),
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    print(f"Predicting mask for '{image_path}'...")
    
    with torch.no_grad():
        output = model(input_tensor)
        
    probabilities = torch.sigmoid(output)
    predicted_mask = (probabilities > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
    
    original_size = image.size
    predicted_mask_uint8 = (predicted_mask * 255).astype(np.uint8)
    final_mask = cv2.resize(predicted_mask_uint8, original_size, interpolation=cv2.INTER_NEAREST)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_mask)
    print(f"Prediction complete. Mask saved to '{output_path}'")

    MIN_SPILL_PIXELS = 100 
    white_pixels = np.count_nonzero(final_mask)
    
    if white_pixels > MIN_SPILL_PIXELS:
        print(f"\n*** Spill Detected! ({white_pixels} pixels) ***")
    else:
        print(f"\n*** No Spill Detected. ({white_pixels} pixels) ***")

if __name__ == "__main__":
    IMAGE_TO_PREDICT_PATH = "dataset/test/images/Oil (1220).jpg"
    TRAINED_MODEL_PATH = "model.pth"
    OUTPUT_MASK_PATH = "predicted/predicted_mask.png"
    
    predict(IMAGE_TO_PREDICT_PATH, TRAINED_MODEL_PATH, OUTPUT_MASK_PATH)