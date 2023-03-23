'''
RECONSTRUCTION DOMINATES PREDICTION
'''
import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import datasets as ds
import numpy as np
from torch import nn
from utils import set_seed
from torch.utils.data import DataLoader
from classifiers import ClassifierAutoEncoder
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the JSON file with the input arguments',
                   type = str)
args = parser.parse_args()

def penalize_separated(batch_latent, batch_lab):
    '''
    This function is used to penalize latent representations
    with the same label that are far apart. The objective
    is to somehow put together latent representations with
    the same label
    '''
    # Little trick to identify observations with the
    # same label when using encoding 1-0. (1 -> positive class)
    # (0 -> negative class).
    # If aux_mask[i, j] is 2, then observations i and j have different label
    # and thus they should not be penalized if they are separated in the latent space
    aux_mask = (batch_lab.view((-1,1)) + 1) @ (batch_lab.view((1,-1)) + 1)
    
    mask = torch.ones_like(aux_mask)
    mask[aux_mask == 2] = 0.0
    
    # Distance matrix
    dist_mat = torch.cdist(batch_latent.flatten(start_dim = 1),
                           batch_latent.flatten(start_dim = 1),
                           p = 2.0)
    
    return (mask * dist_mat).sum()

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
    image_size = config['image_size']
    seed = config['seed']

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    loss_metric_classif = nn.BCELoss()
    loss_metric_recons = nn.MSELoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # For reproducibility
    set_seed(seed)
    torch.backends.cudnn.determinstic = True 
    torch.backends.cudnn.benchmark = False
    
    for i in range(len(train_dir_pos)):
        chk_name = f'{os.path.join(out_path, out_file[i])}.pth'
        model = ClassifierAutoEncoder(image_size).to(device)
        opt = optim.Adam(model.parameters(), **adam_par)

        # Data loaders
        train_ds = ds.ImageDataset(train_dir_pos[i], train_dir_neg[i])
        train_load = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
        test_ds = ds.ImageDataset(test_dir_pos[i], test_dir_neg[i])
        test_load = DataLoader(test_ds, batch_size = batch_size, shuffle = False)
        
        count_stop = 0
        prev_test_accuracy = 0.0
        test_accuracy = 0.0
        prev_test_recons = torch.inf
        test_recons = 0.0
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
                pred_lab, batch_latent, batch_recons = model(batch_im)

                # Loss
                loss_separated = penalize_separated(batch_latent, batch_lab)
                loss = loss_metric_classif(pred_lab, batch_lab) + loss_metric_recons(batch_recons, batch_im) + loss_separated * 0.001

                # Backpropagation
                loss.backward()

                # Update parameters
                opt.step()

                running_loss = running_loss + loss.item()

                if (j + 1) % 100 == 0:
                    print(f'Model: {out_file[i]}. Epoch: {epoch + 1}/{epochs}, Batch: {j + 1}/{len(train_load)}, Average Train Loss: {running_loss / (j + 1):.4f} \n')
                    
            test_accuracy, test_recons = model.test_performance(test_load, accept_lev)
            if test_accuracy > prev_test_accuracy or test_recons < prev_test_recons:
                if test_accuracy > prev_test_accuracy:
                    print(f' {"=" * 25}>> Improvement in Accuracy \n')
                if test_recons < prev_test_recons:
                    print(f' {"=" * 25}>> Improvement in Reconstruction \n')
                print(f' {"@"*20} Improvement in test metrics from. Saving model {"@"*20} \n')
                print(f' {"@"*20} Previous Best Test Accuracy: {prev_test_accuracy:.4f}. Current Test Accuracy {test_accuracy:.4f} \n')
                print(f' {"@"*20} Previous Best Test MSE: {prev_test_recons:.7f}. Current Test MSE {test_recons:.7f} \n')
                torch.save(model.state_dict(), chk_name)
                prev_test_accuracy = test_accuracy
                prev_test_recons = test_recons
                count_stop = 0
            else:
                count_stop = count_stop + 1
            
            # If no improvement in consecutive epochs
            # move to the next model
            if count_stop >= early_stop:
                print(f' {"X" * 20} No improvement in test accuracy for {count_stop} consecutive epochs {"X" * 20}. Training next model. \n')
                break
            print(f' {"=" * 20} End of Epoch {epoch + 1}/{epochs} {"=" * 20} \n')
            print(f' {"@"*20} Previous Best Test Accuracy: {prev_test_accuracy:.4f}. Current Test Accuracy {test_accuracy:.4f} \n')
            print(f' {"@"*20} Previous Best Test MSE: {prev_test_recons:.7f}. Current Test MSE {test_recons:.7f} \n')

if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config)
    print(" ===== Model(s) Trained ===== \n")