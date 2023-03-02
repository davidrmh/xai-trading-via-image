import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torch

class ImageDataset(Dataset):
    
    def __init__(self, dir_pos: str, dir_neg: str, bool_norm: bool = True):
        self.dir_pos = dir_pos
        self.dir_neg = dir_neg
        self.bool_norm = bool_norm
        self.files = os.listdir(dir_pos) + os.listdir(dir_neg)
        self.labels = [1] * int(len(self.files) / 2) + [0] * int(len(self.files) / 2)
    
    def image2tensor(self, image_path, bool_norm = True) -> torch.Tensor:
        image = read_image(image_path, ImageReadMode.RGB)
        #Change dtype to float32
        image = image.to(torch.float32)

        # Normalize in [0, 1]
        if bool_norm:
            image = image / image.max()

        # Permute is used for plotting using matplotlib imshow
        # a tensor with shape (channels, heigth, width)
        # is reshaped to (height, width, channels)
        # curiosly reshape method does not work
        image = image.permute([1, 2, 0])
        return image
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.labels[idx] == 1:
            image_path = os.path.join(self.dir_pos, self.files[idx])
        else:
            image_path = os.path.join(self.dir_neg, self.files[idx])
        image = self.image2tensor(image_path, self.bool_norm)
        
        return image, self.labels[idx]
        
        