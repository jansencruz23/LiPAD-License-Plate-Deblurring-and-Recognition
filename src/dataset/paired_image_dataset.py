import os
from PIL import Image
from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    def __init__(self, blur_dir, normal_dir, transform=None):
        self.blur_dir = blur_dir
        self.normal_dir = normal_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(blur_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        blur_path = os.path.join(self.blur_dir, filename)
        normal_path = os.path.join(self.normal_dir, filename)

        blur_img = Image.open(blur_path).convert('RGB')
        normal_img = Image.open(normal_path).convert('RGB')

        if self.transform:
            blur_img = self.transform(blur_img)
            normal_img = self.transform(normal_img)

        return blur_img, normal_img