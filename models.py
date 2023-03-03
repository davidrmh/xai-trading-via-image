import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import datasets as ds


class JPClassifier(nn.Module):
    """
    Classifier from paper Trading via image classification
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 32, (3, 3))
        self.maxpool1 = nn.MaxPool2d((2, 2))
        self.maxpool2 = nn.MaxPool2d((2, 2))
        self.fc = nn.Linear(32 * 52 * 52, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)
        x = self.sig(x)
        return x.flatten()
        
dir_pos = './images_2010_2017/BB_Buy/'
dir_neg = './images_2010_2017/no_BB_Buy/'
im_ds = ds.ImageDataset(dir_pos, dir_neg)
dsload = DataLoader(im_ds, batch_size = 5, shuffle=True)
im_b, lab_b = next(iter(dsload))
model = JPClassifier()
y = model(im_b)
bce = nn.BCELoss()
loss = bce(y, lab_b)
