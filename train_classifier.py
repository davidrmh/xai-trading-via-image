import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import datasets as ds
import numpy as np
from torch import nn
from torch.utils.data import DataLoader


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

# Input configuration
train_dir_pos = './images_2010_2017/BB_Buy/'
train_dir_neg = './images_2010_2017/no_BB_Buy/'
test_dir_pos = './images_2018/BB_Buy/'
test_dir_neg = './images_2018/no_BB_Buy/'
out_path = './jpclassifier'
out_file = 'jpclassifier'
batch_size = 32
epochs = 3
checkpoint_iter = 5
loss_metric = nn.BCELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = JPClassifier().to(device)
opt = optim.Adam(model.parameters())

if not os.path.exists(out_path):
    os.mkdir(out_path)

# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(19900802)

# For reproducibility
torch.backends.cudnn.determinstic = True 
torch.backends.cudnn.benchmark = False

# Data loaders
train_ds = ds.ImageDataset(train_dir_pos, train_dir_neg)
train_load = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
test_ds = ds.ImageDataset(test_dir_pos, test_dir_neg)
test_load = DataLoader(test_ds, batch_size = batch_size, shuffle = False)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_load, start = 0):
        batch_im, batch_lab = data
        batch_im, batch_lab = batch_im.to(device), batch_lab.to(device)
        
        # Zero gradients
        opt.zero_grad()
        
        # Predictions
        pred_lab = model(batch_im)
        
        # Loss
        loss = loss_metric(pred_lab, batch_lab)
        
        # Backpropagation
        loss.backward()
        
        # Update parameters
        opt.step()
        
        running_loss = running_loss + loss.item()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Average loss: {running_loss / (i + 1):.3f}')
    
    if (epoch + 1) % checkpoint_iter == 0:
        chk_name = f'{os.path.join(out_path, out_file)}.pth'
        torch.save(model.state_dict(), chk_name)
print(" ===== Model Trained ===== \n")
        





