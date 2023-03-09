import os
import json
import torch
import pickle
import resnet
import argparse
import pandas as pd
from utils import set_seed
from datasets import PredImageDataset
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the JSON file with the input arguments',
                   type = str)
args = parser.parse_args()

@torch.no_grad()
def join_feat_maps(model: resnet.ResNetAutoEncoder,
                  dataloader: DataLoader):
    """
    Returns a matrix with the (flattened) feature maps
    and a list with the corresponding files
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f_maps = torch.tensor([], dtype = torch.float32)
    list_files = []
    for batch_im, batch_f in dataloader:
        batch_im = batch_im.to(device)
        feat_map = model.encoder(batch_im).cpu().detach()
        feat_map = feat_map.flatten(start_dim = 1)
        f_maps = torch.concat((f_maps, feat_map))
        list_files.extend(batch_f)
    return f_maps.numpy(), list_files

def main(config: dict) -> None:
    """
    Creates pickle files.
    - pca_*.pkl: Contains the PCA object fitted
      with the feature maps of the training data
      
    - scaler_*.pkl: Contains the StandardScaler object
    fitted with the feature maps of the training data
      
    - train_pca_*.pkl: DataFrame with the latent representation
    of the feature maps from the training data
    
    - test_pca_*.pkl: DataFrame with the latent representation of the
    feature maps from the test data
    """
    path_auto = config['path_auto']
    path_train = config['path_train']
    path_test = config['path_test']
    out_dir = config['out_dir']
    img_shape = config['img_shape']
    num_comp = config['num_comp']
    batch_size = config['batch_size']
    seed = config['seed'] if 'seed' in config else 19900802
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # For reproducibility
    set_seed(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    
    for i in range(len(path_auto)):
        
        # Load model
        model = resnet.ResNetAutoEncoder(img_shape)
        checkpoint = torch.load(path_auto[i])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Create data loaders
        train_ds = PredImageDataset(path_train[i])
        train_load = DataLoader(train_ds, batch_size = batch_size, shuffle = False)
        test_ds = PredImageDataset(path_test[i])
        test_load = DataLoader(test_ds, batch_size = batch_size, shuffle = False)
        
        # Join feature maps for training set
        train_feat_map, train_files = join_feat_maps(model, train_load)
        
        # Join feature maps for test set
        test_feat_map, test_files = join_feat_maps(model, test_load)
        
        # Fit standardizer to train feature maps
        # and scales it
        scaler = StandardScaler()
        scaler.fit(train_feat_map)
        train_feat_map = scaler.transform(train_feat_map)
        
        # Fit pca to (scaled) train feature maps
        pca = PCA(n_components = num_comp)
        pca.fit(train_feat_map)
        
        # Obtain pca latent representation
        # for (scaled) training feature maps
        pca_train = pca.transform(train_feat_map)
        
        # scale test feature maps using the
        # mean and std from the train maps
        test_feat_map = scaler.transform(test_feat_map)
        
        # Obtain pca latent representation
        # for (scaled) test feature maps
        pca_test = pca.transform(test_feat_map)
        
        # Create DataFrames storing the latent
        # representations
        df_pca_train = pd.concat((pd.Series(train_files, name = 'file'),
                                 pd.DataFrame(pca_train)), axis = 1)
        
        df_pca_test = pd.concat((pd.Series(test_files, name = 'file'),
                                 pd.DataFrame(pca_test)), axis = 1)
        
        # Save objects
        aux = path_auto[i].split('/')[-1].replace('pth', 'pkl')
        f_pca = os.path.join(out_dir, f'pca_{aux}')
        f_scaler = os.path.join(out_dir, f'scaler_{aux}')
        f_train_pca = os.path.join(out_dir, f'train_pca_{aux}')
        f_test_pca = os.path.join(out_dir, f'test_pca_{aux}')
        
        with open(f_pca, 'wb') as f:
            pickle.dump(pca, f)
            
        with open(f_scaler, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(f_train_pca, 'wb') as f:
            pickle.dump(df_pca_train, f)
        
        with open(f_test_pca, 'wb') as f:
            pickle.dump(df_pca_test, f)
            
        print(f" {'*' * 20} Finishing with model {aux} {'*' * 20}\n")

if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config)
    print(f" {'=' * 25} PCA Finished {'=' * 25} \n")