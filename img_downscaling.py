import os
import glob
from PIL import Image

# Paths to High-Resolution Images
BASE_DIR = "./DIV2K"  # Change this to your dataset root directory
HR_TRAIN_PATH = os.path.join(BASE_DIR, "DIV2K_train_HR")
HR_VALID_PATH = os.path.join(BASE_DIR, "DIV2K_valid_HR")

# Low-Resolution Save Paths
LR_TRAIN_PATH = os.path.join(BASE_DIR, "DIV2K_train_LR")
LR_VALID_PATH = os.path.join(BASE_DIR, "DIV2K_valid_LR")

# Scaling Factors
UPSCALE_FACTORS = [2, 4, 8]

# Ensure directories exist for each factor
for scale in UPSCALE_FACTORS:
    os.makedirs(os.path.join(LR_TRAIN_PATH, f"X{scale}"), exist_ok=True)
    os.makedirs(os.path.join(LR_VALID_PATH, f"X{scale}"), exist_ok=True)

# Function to generate low-resolution images
def generate_low_res_images(hr_dir, lr_base_dir):
    """Downscale high-resolution images for multiple scales."""
    hr_images = glob.glob(os.path.join(hr_dir, "*.png"))
    
    for img_path in hr_images:
        img = Image.open(img_path).convert("RGB")  # Open image
        w, h = img.size
        img_name = os.path.basename(img_path)

        for scale in UPSCALE_FACTORS:
            lr_img = img.resize((w // scale, h // scale), Image.BICUBIC)
            save_path = os.path.join(lr_base_dir, f"X{scale}", img_name)
            lr_img.save(save_path)

# Generate Low-Resolution Images for Training and Validation
generate_low_res_images(HR_TRAIN_PATH, LR_TRAIN_PATH)
generate_low_res_images(HR_VALID_PATH, LR_VALID_PATH)