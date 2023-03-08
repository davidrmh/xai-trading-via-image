import os
import argparse
import json
import torch
import resnet
import time
import datasets as ds
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from utils import set_seed


dict_optim = {'adam':optim.Adam, 'sgd': optim.SGD}

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the JSON file with the input arguments',
                   type = str)
args = parser.parse_args()

def main(config: dict) -> None:
    train_images = config['train_images']
    test_images = config['test_images']
    out_dir = config['out_dir']
    out_files = config['out_files']
    img_shape = config['img_shape']
    batch_size = config['batch_size']
    epochs = config['epochs']
    early_iter = config['early_iter']
    early_tol = config['early_tol']
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    loss_metric = nn.MSELoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    for i in range(len(out_files)):
        model_init_time = time.perf_counter()
        # name of the output file storing model's state_dict
        out_f = f'{os.path.join(out_dir, out_files[i])}.pth'
        
        # Initialize model or continue training
        model = resnet.ResNetAutoEncoder(img_shape)
        if config['continue'].lower() == 'yes' and os.path.exists(out_f):
            checkpoint = torch.load(out_f)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            optim = dict_optim[config['optim']](model.parameters())
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f" {'*'*25} Resuming traning of {out_f} {'*'*25} \n")
        else:
            model = model.to(device)
            optim = dict_optim[config['optim']](model.parameters(), **config['optim_par'])
        
        # Dataloaders
        train_ds = ds.PredImageDataset(train_images[i])
        train_load = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
        test_ds = ds.PredImageDataset(test_images[i])
        test_load = DataLoader(test_ds, batch_size = batch_size, shuffle = False)
        
        # For reproducibility
        seed = config['seed'] if 'seed' in config else 19900802
        set_seed(seed)
        torch.backends.cudnn.determinstic = True 
        torch.backends.cudnn.benchmark = False
        
        # Helpers for early stopping
        start_epoch = checkpoint['epoch'] + 1 if config['continue'] == "yes" else 0
        count_stop = 0
        prev_test_loss = checkpoint['prev_test_loss'] if config['continue'] == "yes" else torch.inf
        test_loss = torch.inf
        for epoch in range(start_epoch, epochs):
            model.train(True)
            running_loss = 0.0
            for j, data in enumerate(train_load):
                batch_im, batch_f = data
                batch_im = batch_im.to(device)
                
                # Zero gradients
                optim.zero_grad()
                
                # Forward pass (reconstructions)
                output = model(batch_im)
                
                # Loss
                loss = loss_metric(batch_im, output)
                
                # Backpropagation
                loss.backward()
                
                # Update parameters
                optim.step()
                
                # Update running loss for the current epoch
                running_loss = running_loss + loss.item()
                
                if (j + 1) % 100 == 0:
                    print(f'Model: {out_files[i]}. Epoch: {epoch + 1}/{epochs}, Batch: {j + 1}/{len(train_load)}, Average train loss: {running_loss / (j + 1):.7f} \n')
                    
            # Test loss after finishing the epoch
            test_loss = model.test_loss(test_load, loss_metric)
            print(f' {"="*10} By the end of epoch {epoch + 1}/{epochs}, the test loss is: {test_loss:.7f} {"="*10} \n')
            
            # Save if there is a significant improvement
            if prev_test_loss - test_loss > early_tol:
                print(f' {"@"*20} Improvement in test loss from {prev_test_loss:.7f} to {test_loss:.7f}. Saving model {"@"*20} \n')
                prev_test_loss = test_loss
                torch.save({
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'epoch': epoch,
                    'prev_test_loss': prev_test_loss
                }, out_f)
                count_stop = 0
            else:
                count_stop = count_stop + 1
            
            # If no improvement in  early_iter consecutive epochs
            # move to the next model
            if count_stop >= early_iter:
                print(f' {"X" * 20} No improvement in test loss for {count_stop} consecutive epochs {"X" * 20}. Training next model. \n')
                break
        model_end_time = time.perf_counter()
        print(f' ===== Finishing with model {out_files[i]}. Elapsed time: {int( (model_end_time - model_init_time) / 60 )} minutes ===== \n')


if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config)
    print(" ===== Model(s) Trained ===== \n")