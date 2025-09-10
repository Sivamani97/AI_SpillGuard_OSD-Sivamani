import os
import cv2
import numpy as np

# === CONFIG ===
BASE_DIR = "dataset"
SPLITS = ["train", "val", "test"]
OUT_FOLDER = "binary_masks"
OIL_RGB = np.array([255, 0, 124], dtype=np.int16)
TOLERANCE = 15 # Increased tolerance for better color matching

def convert_mask_to_binary(mask_path, out_path, oil_rgb=OIL_RGB, tol=TOLERANCE):
    img = cv2.imread(mask_path)
    if img is None:
        raise FileNotFoundError(mask_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.int16)
    dist = np.linalg.norm(img_rgb - oil_rgb[None, None, :], axis=-1)
    oil_mask = dist <= tol
    binary = (oil_mask.astype(np.uint8)) * 255
    cv2.imwrite(out_path, binary)

def process_split(split):
    masks_dir = os.path.join(BASE_DIR, split, "masks")
    out_dir = os.path.join(BASE_DIR, split, OUT_FOLDER)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if len(files) == 0:
        print(f"[WARN] No mask files found in {masks_dir}")
        return
    for fname in files:
        in_path = os.path.join(masks_dir, fname)
        out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".png")
        convert_mask_to_binary(in_path, out_path)
    print(f"[OK] Processed {len(files)} masks -> {out_dir}")

if __name__ == "__main__":
    for s in SPLITS:
        print("Processing split:", s)
        process_split(s)
    print("All done. Binary masks created.")