import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torch

class ImageDataset(Dataset):
    
    def __init__(self, dir_pos: str, dir_neg: str):
        self.dir_pos = dir_pos
        self.dir_neg = dir_neg
        self.files_pos = os.listdir(dir_pos)
        self.files_neg = os.listdir(dir_neg)
        self.files = self.files_pos + self.files_neg
        self.labels = [1.0] * len(self.files_pos) + [0.0] * len(self.files_neg)
        self.labels = torch.tensor(self.labels, dtype = torch.float32)
    
    def image2tensor(self, image_path) -> torch.Tensor:
        image = read_image(image_path, ImageReadMode.RGB)
        #Change dtype to float32
        image = image.to(torch.float32)

        # Normalize in [0, 1]
        image = image / image.max()

        return image
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx):
        if idx < len(self.files_pos):
            image_path = os.path.join(self.dir_pos, self.files[idx])
        else:
            image_path = os.path.join(self.dir_neg, self.files[idx])
        image = self.image2tensor(image_path)
        
        return image, self.labels[idx], image_path
        
class PredImageDataset(Dataset):
    
    def __init__(self, dirs: list[str]):
        self.files = []
        for d in dirs:
            self.files.extend([os.path.join(d, f) for f in os.listdir(d)])
    
    # TO DO: Move this function out of the class
    # in order to avoid repetition with ImageDataset class
    def image2tensor(self, image_path) -> torch.Tensor:
        image = read_image(image_path, ImageReadMode.RGB)
        #Change dtype to float32
        image = image.to(torch.float32)

        # Normalize in [0, 1]
        image = image / image.max()

        return image
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx):
        image = self.image2tensor(self.files[idx])
        return image, self.files[idx]
        