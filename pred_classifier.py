import os
import json
import torch
import argparse
import pandas as pd
from train_classifier import JPClassifier
from torch.utils.data import DataLoader
from datasets import ImageDataset, PredImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the JSON file with the input arguments',
                   type = str)
args = parser.parse_args()

def main(config: dict) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    path_model = config['path_model']

    # Nested list with paths of the images to use.
    # The length must be equal to the length of path_model.
    # The i-th inner corresponds to the i-th classifier in path_model
    # In validation mode the first element of each inner list
    # is the path of the images with positive class whereas the
    # second element is tha path of the images with negative class.
    path_img = config['path_img']

    # Validation or prediction mode (val or pred)
    mode = config['mode'].lower()
    
    if not mode in ("val", "pred"):
        raise Exception("Values for mode are 'val' or 'pred', but {mode} was given.")
    
    batch_size = config['batch_size']
    
    accept_lev = config['accept_lev']
    
    out_dir = config['out_dir']
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    for i in range(len(path_model)):
        print(f' {"*" * 25} Predicting with classifier {path_model[i]} {"*" * 25}\n')
        # Load model
        model = JPClassifier()
        model.load_state_dict(torch.load(path_model[i]))
        model.train(False)
        model.to(device)
        
        # Create dataloader
        if mode == 'val':
            dataset = ImageDataset(path_img[i][0], path_img[i][1])
        else:
            dataset = PredImageDataset(path_img[i])
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
        
        # Make predictions
        list_pred = []
        list_true = []
        list_agree = []
        list_f = []
        for j, data in enumerate(dataloader):
            if mode == 'val':
                batch_im, batch_lab, batch_f = data
            else:
                batch_im, batch_f = data
            batch_im = batch_im.to(device)
            
            with torch.no_grad():
                pred = model(batch_im)
            pred[pred >= accept_lev] = 1.0
            pred[pred < accept_lev] = 0.0
            pred = pred.cpu().detach()
            
            # Update lists for creating the pandas DataFrame
            list_pred.extend(pred.numpy())
            list_f.extend(batch_f)
            if mode == 'val':
                agree = pred == batch_lab
                list_true.extend(batch_lab.numpy())
                list_agree.extend(agree.numpy())
                
        # Save results in a pandas DataFrame
        filename = f"pred_{path_model[i].split('/')[-1].replace('.pth', '')}.csv"
        if mode == 'val':
            df_data = {"file":list_f, "prediction":list_pred, 'truth': list_true, 'is_correct?':list_agree}
        else:
            df_data = {"file":list_f, "prediction":list_pred}
        df = pd.DataFrame(data = df_data)
        df.to_csv(os.path.join(out_dir, filename), index = False)
            
if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config)
    print(f" {'='*25} Predictions saved {'='*25}\n")
