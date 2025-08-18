import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json

class MyShapesDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(root) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.root, img_name)
        json_path = img_path.replace('.jpg', '.json')

        image = Image.open(img_path).convert('L')
        with open(json_path, 'r') as f:
            data = json.load(f)
        label = torch.tensor([
            data["counts"]["circle"],
            data["counts"]["triangle"],
            data["counts"]["square"]
        ], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label