import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    """
    Classifier from paper Trading via image classification
    """
    def __init__(self, image_size = (70, 70)):
        super().__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 32, (3, 3))
        self.maxpool1 = nn.MaxPool2d((2, 2))
        self.maxpool2 = nn.MaxPool2d((2, 2))
        self.linear_size = self.get_linear_input_size()
        self.fc = nn.Linear(self.linear_size, 1)
        self.sig = nn.Sigmoid()
        
    # Helper function for making possible to work
    # with images of arbitrary size
    @torch.no_grad()
    def get_linear_input_size(self):
        x = torch.zeros((1, 3, *self.image_size))
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim = 1)
        return x.shape[1]

        
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

    def test_acc(self, test_load, accept_lev = 0.5):
        self.train(False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        count_correct, total_count = 0.0, 0.0
        for batch_im, batch_lab, _ in test_load:
            batch_im, batch_lab = batch_im.to(device), batch_lab.to(device)
            with torch.no_grad():
                pred_lab = self(batch_im)
                count_correct += ((pred_lab >= accept_lev) == batch_lab).sum().item()
                total_count += batch_lab.shape[0]
        test_accuracy = count_correct / total_count
        self.train(True)
        return test_accuracy

class ClassifierEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 32, (3, 3))
        # return_indices is set to true in order to use MaxUnpool2d
        # in the decoder part
        self.maxpool1 = nn.MaxPool2d((2, 2), return_indices = True)
        self.maxpool2 = nn.MaxPool2d((2, 2), return_indices = True)
        
    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x, idx1 = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x, idx2 = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        return x, [idx1, idx2]

class ClassifierDecoder(nn.Module):
    
    def __init__(self, image_size):
        super().__init__()
        self.inv_conv1 = nn.ConvTranspose2d(32, 32, (3, 3))
        self.inv_conv2 = nn.ConvTranspose2d(32, 32, (3, 3))
        self.inv_conv3 = nn.ConvTranspose2d(32, 3, (3, 3))
        self.inv_maxpool1 = nn.MaxUnpool2d((2, 2))
        self.inv_maxpool2 = nn.MaxUnpool2d((2, 2))
        self.upsample = nn.Upsample(image_size)
        
    def forward(self, x: torch.Tensor, indices: list) -> torch.Tensor:
        x_recons = F.relu(self.inv_conv1(x))
        # IMPORTANT: Notice the order of indices
        # See forward method of ClassifierEncoder
        x_recons = self.inv_maxpool1(x_recons, indices[1])
        x_recons = F.relu(self.inv_conv2(x_recons))
        x_recons = self.inv_maxpool2(x_recons, indices[0])
        x_recons = F.relu(self.inv_conv3(x_recons))
        x_recons = self.upsample(x_recons)
        
        return x_recons

class ClassifierAutoEncoder(nn.Module):
    """
    Classifier from paper Trading via image classification + Autoencoder
    """
    def __init__(self, image_size = (70, 70)):
        super().__init__()
        self.image_size = image_size
        
        # Encoder
        self.encoder = ClassifierEncoder()
        
        # Decoder
        self.decoder = ClassifierDecoder(self.image_size)
        
        # Fully connected for classifier
        self.linear_size = self.get_linear_input_size()
        self.fc = nn.Linear(self.linear_size, 1)
        self.sig = nn.Sigmoid()
        
    # Helper function for making possible to work
    # with images of arbitrary size
    @torch.no_grad()
    def get_linear_input_size(self):
        x = torch.zeros((1, 3, *self.image_size))
        x, _ = self.encoder(x)
        x = torch.flatten(x, start_dim = 1)
        return x.shape[1]

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, indices = self.encoder(x)
        
        # Transformations related to classifier
        x_class = torch.flatten(x, start_dim = 1)
        x_class = self.fc(x_class)
        x_class = self.sig(x_class)
        
        # Transformations related to reconstruction
        x_recons = self.decoder(x, indices)
        
        return x_class.flatten(), x_recons

    def test_acc(self, test_load, accept_lev = 0.5):
        self.train(False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        count_correct, total_count = 0.0, 0.0
        for batch_im, batch_lab, _ in test_load:
            batch_im, batch_lab = batch_im.to(device), batch_lab.to(device)
            with torch.no_grad():
                pred_lab = self(batch_im)
                count_correct += ((pred_lab >= accept_lev) == batch_lab).sum().item()
                total_count += batch_lab.shape[0]
        test_accuracy = count_correct / total_count
        self.train(True)
        return test_accuracy
