import os
import json
import shutil
import argparse
import pandas as pd
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the JSON file with the input arguments',
                   type = str)
args = parser.parse_args()

    # Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def separate_correct(config: dict) -> None:
    """
    Given a CSV (created with pred_classfier.py) with a classifier's predictions
    separate the classified images in correctly/uncorrectly classified.
    """
    file_pred = config['file_pred']
    out_dir = config['out_dir']
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    for i in range(len(file_pred)):
        df = pd.read_csv(file_pred[i])
        
        subdir = file_pred[i].split('/')[-1].replace('.csv', '')
        subdir_correct = os.path.join(out_dir, f'{subdir}/correct')
        subdir_incorrect = os.path.join(out_dir, f'{subdir}/incorrect')
        
        if not os.path.exists(os.path.join(out_dir, subdir)):
            os.mkdir(os.path.join(out_dir, subdir))
        os.mkdir(subdir_correct)
        os.mkdir(subdir_incorrect)
            
        correct = df[df['is_correct?'] == True]['file']
        incorrect = df[df['is_correct?'] == False]['file']
        
        # Files must be accesible from the directory where this file
        # is stored
        for f in correct:
            filepath = os.path.join(subdir_correct, f.split('/')[-1])
            shutil.copyfile(f, filepath)
        for f in incorrect:
            filepath = os.path.join(subdir_incorrect, f.split('/')[-1])
            shutil.copyfile(f, filepath)
        print(f'{"*" * 25} Finished with file {file_pred[i]} {"*" * 25}\n')
        
    print(f"{'=='*25} Files copied {'=='*25}")

def separate_class(config: dict) -> None:
    """
    Given a CSV (created with pred_classfier.py) with a classifier's predictions
    separate the classified images in positive/negative class.
    """
    file_pred = config['file_pred']
    out_dir = config['out_dir']
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    for i in range(len(file_pred)):
        df = pd.read_csv(file_pred[i])
        
        subdir = file_pred[i].split('/')[-1].replace('.csv', '')
        subdir_positive = os.path.join(out_dir, f'{subdir}/positive')
        subdir_negative = os.path.join(out_dir, f'{subdir}/negative')
        
        if not os.path.exists(os.path.join(out_dir, subdir)):
            os.mkdir(os.path.join(out_dir, subdir))
        os.mkdir(subdir_positive)
        os.mkdir(subdir_negative)
            
        positive = df[df['prediction'] == 1.0]['file']
        negative = df[df['prediction'] == 0.0]['file']
        
        # Files must be accesible from the directory where this file
        # is stored
        for f in positive:
            filepath = os.path.join(subdir_positive, f.split('/')[-1])
            shutil.copyfile(f, filepath)
        for f in negative:
            filepath = os.path.join(subdir_negative, f.split('/')[-1])
            shutil.copyfile(f, filepath)
        print(f'{"*" * 25} Finished with file {file_pred[i]} {"*" * 25}\n')
        
    print(f"{'=='*25} Files copied {'=='*25}")

if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    if config['function'].lower() == 'separate_correct':
        separate_correct(config)
    if config['function'].lower() == 'separate_class':
        separate_class(config)
    # Here is possible to add more functions to be called
