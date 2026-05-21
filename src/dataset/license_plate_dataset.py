import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import random
import pytesseract

class LicensePlateDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, transform=None, img_size=(128, 256)):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.transform = transform
        self.img_size = img_size

        self.blur_images = sorted([f for f in os.listdir(blur_dir) if f.endswith(('.jpg', '.png'))])
        self.sharp_images = sorted([f for f in os.listdir(sharp_dir) if f.endswith(('.jpg', '.png'))])

        assert len(self.blur_images) == len(self.sharp_images), "Mismatch in number of blur and sharp images"

    def __len__(self):
        return len(self.blur_images)
    
    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_dir, self.blur_images[idx])
        sharp_path = os.path.join(self.sharp_dir, self.sharp_images[idx])

        blur_img = Image.open(blur_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')

        try:
            text = pytesseract.image_to_string(sharp_img, config='--psm 8')
            valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            is_readable = len(text.strip()) > 0 and all(c in valid_chars for c in text.strip().upper())
        except:
            is_readable = False

        if not is_readable:
            print(f"Unreadable sharp image: {self.sharp_images[idx]}")
            blur_img = blur_img.transpose(Image.FLIP_LEFT_RIGHT)
            sharp_img = sharp_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        blur_img = blur_img.resize(self.img_size, Image.LANCZOS)
        sharp_img = sharp_img.resize(self.img_size, Image.LANCZOS)

        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)

        return blur_img, sharp_img