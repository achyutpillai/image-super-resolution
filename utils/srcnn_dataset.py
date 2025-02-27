import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random


# Custom dataset to load LR and HR image pairs
class ImagePairDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None, crop_size=96, scaling_factor=2):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.image_filenames = sorted(os.listdir(hr_dir))  # Ensure matching filenames

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.image_filenames[idx])
        lr_path = os.path.join(self.lr_dir, self.image_filenames[idx])

        hr_img = Image.open(hr_path).convert("RGB")  # Load as RGB
        lr_img = Image.open(lr_path).convert("RGB")
        
        # Apply random crop
        lr_img, hr_img = random_crop(hr_img, lr_img, self.crop_size, self.scaling_factor)
        
        # Resize LR image to match HR size
        lr_img = lr_img.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        
        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)

        return lr_img, hr_img

# Function to randomly crop images to ensure all data samples are of equal size
def random_crop(hr_img, lr_img, crop_size=96, scaling_factor=2):
    hr_width, hr_height = hr_img.size
    lr_width, lr_height = lr_img.size
    
    # Ensure crop coordinates are within valid range
    hr_x = random.randint(0, hr_width - crop_size)
    hr_y = random.randint(0, hr_height - crop_size)
    lr_x = hr_x // scaling_factor  # Adjusting for low-res size
    lr_y = hr_y // scaling_factor
    
    hr_crop = hr_img.crop((hr_x, hr_y, hr_x + crop_size, hr_y + crop_size))
    lr_crop = lr_img.crop((lr_x, lr_y, lr_x + crop_size // scaling_factor, lr_y + crop_size // scaling_factor))
    
    return lr_crop, hr_crop


# Function to generate low-resolution images
def generate_low_res(image, scaling_factor=2):
    width, height = image.size
    low_res = image.resize((width // scaling_factor, height // scaling_factor), Image.BICUBIC)
    return low_res