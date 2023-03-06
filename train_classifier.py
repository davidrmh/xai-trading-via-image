import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import datasets as ds
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the JSON file with the input arguments',
                   type = str)
args = parser.parse_args()

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

def main(config) -> None:
    train_dir_pos = config['train_dir_pos']
    train_dir_neg = config['train_dir_neg']
    test_dir_pos = config['test_dir_pos']
    test_dir_neg = config['test_dir_neg']
    out_path = config['out_path']
    out_file = config['out_file']
    adam_par = config['adam_par'] if 'adam_par' in config else {}
    batch_size = config['batch_size']
    epochs = config['epochs']
    accept_lev = config['accept_lev']
    early_stop = config['early']

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    loss_metric = nn.BCELoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
    for i in range(len(train_dir_pos)):
        chk_name = f'{os.path.join(out_path, out_file[i])}.pth'
        model = JPClassifier().to(device)
        opt = optim.Adam(model.parameters(), **adam_par)

        # Data loaders
        train_ds = ds.ImageDataset(train_dir_pos[i], train_dir_neg[i])
        train_load = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
        test_ds = ds.ImageDataset(test_dir_pos[i], test_dir_neg[i])
        test_load = DataLoader(test_ds, batch_size = batch_size, shuffle = False)
        
        count_stop = 0
        prev_test_accuracy = 0.0
        test_accuracy = 0.0
        print(f' {"*" * 25} Training Classifier {out_file[i]} {"*" * 25} \n')
        # Training loop
        for epoch in range(epochs):
            model.train(True)
            running_loss = 0.0
            for j, data in enumerate(train_load):
                batch_im, batch_lab, _ = data
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

                if (j + 1) % 100 == 0:
                    print(f'Model: {out_file[i]}. Epoch: {epoch + 1}/{epochs}, Batch: {j + 1}/{len(train_load)}, Average train loss: {running_loss / (j + 1):.4f} \n')
                    
            test_accuracy = model.test_acc(test_load, accept_lev)
            if test_accuracy > prev_test_accuracy:
                print(f' {"@"*20} Improvement in test accuracy from {prev_test_accuracy:.4f} to {test_accuracy:.4f}. Saving model {"@"*20} \n')
                torch.save(model.state_dict(), chk_name)
                prev_test_accuracy = test_accuracy
                count_stop = 0
            else:
                count_stop = count_stop + 1
            
            # If no improvement in consecutive epochs
            # move to the next model
            if count_stop >= early_stop:
                print(f' {"X" * 20} No improvement in test accuracy for {count_stop} consecutive epochs {"X" * 20}. Training next model. \n')
                break
            print(f' {"="*10} By the end of epoch {epoch + 1}/{epochs}, the test accuracy is: {test_accuracy:.4f} {"="*10} \n')

if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config)
    print(" ===== Model(s) Trained ===== \n")

        





