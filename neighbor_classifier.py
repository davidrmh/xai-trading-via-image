import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the JSON file with the input arguments',
                   type = str)
args = parser.parse_args()

def label_files(files: pd.Series, path_positive: str):
    files_pos_class = os.listdir(path_positive)
    labels = np.zeros(files.shape)
    for i, f in enumerate(files):
        f = f.replace('\\', '/')
        if f.split('/')[-1] in files_pos_class:
            labels[i] = 1.0
    return labels

def get_average_accuracy(dist:  np.ndarray,
                         tau: float,
                         source_labels: np.ndarray,
                         neighbors_labels: np.ndarray,
                        train_vs_train: bool = False):
    # number of samples
    n = source_labels.shape[0]
    avg = 0.0
    for i in range(dist.shape[0]):
        idx = np.where(dist[i, :] <= tau)[0]
        # Considers the fact that, when using
        # only train data, the nearest neighbor to a point
        # is the point itself
        if train_vs_train:
            agree = sum(neighbors_labels[idx] == source_labels[i]) - 1
        else:
            agree = sum(neighbors_labels[idx] == source_labels[i])
            
        avg = avg +  agree / idx.shape[0]
    return avg / n

def find_opt_tau(train_dist: np.ndarray,
                 train_labels: np.ndarray,
                tau_step: float = 0.5) -> float:
    min_tau = tau_step
    max_tau = np.quantile(train_dist, 0.5)
    opt_tau = 0.0
    best_avg_train_acc = 0.0
    for tau in np.arange(min_tau, max_tau + tau_step, tau_step):
        # Average accuracy between points in training data
        avg_train_acc = get_average_accuracy(train_dist,
                                        tau,
                                        train_labels,
                                        train_labels,
                                        True)
        if best_avg_train_acc < avg_train_acc:
            opt_tau = tau
            best_avg_train_acc = avg_train_acc
    
    return opt_tau, best_avg_train_acc

def make_predictions(dist_mat: np.ndarray,
                     neighbors_labels: np.ndarray,
                    tau: float) -> np.ndarray:
    
    # To store the predictions
    pred = np.zeros(dist_mat.shape[0])
    
    for i in range(dist_mat.shape[0]):
        idx = np.where(dist_mat[i] <= tau)[0]
        # If there are no neighbors at distance tau
        # then the negative class is predicted (zero)
        if len(idx) > 0:
            closest_lab = neighbors_labels[idx]
            lab, count = np.unique(closest_lab, return_counts = True)
            pred[i] = lab[count.argmax()]

    return pred
        

def main(config: dict) -> None:
    
    path_train_latent = config['path_train_latent']
    path_test_latent = config['path_test_latent']
    path_train_positive = config['path_train_positive']
    path_test_positive = config['path_test_positive']
    out_dir = config['out_dir']
    out_file = config['out_file']
    tau_step = config['tau_step']
    
    for i in range(len(path_train_latent)):
        
        print(f' {"=" * 5} Working with file {path_test_latent[i]} {"=" * 5}\n')

        # Read file with latent representations
        # of the training data
        with open(path_train_latent[i], 'rb') as f:
            latent_train = pickle.load(f)
            train_files = latent_train['file']
            latent_train = latent_train.iloc[:, 1:].to_numpy()
        
        # Read file with latent representations
        # of the test data
        with open(path_test_latent[i], 'rb') as f:
            latent_test = pickle.load(f)
            test_files = latent_test['file']
            latent_test = latent_test.iloc[:, 1:].to_numpy()

        # Compute Euclidean distances between points in the trainig/test set
        print(f' {"=" * 20} Computing distance matrices {"=" * 20}\n')
        train_dist = cdist(latent_train, latent_train)
        test_dist = cdist(latent_test, latent_train)

        # Get labels for train/test dataset
        train_labels = label_files(train_files, path_train_positive[i])
        test_labels = label_files(test_files, path_test_positive[i])

        # Find optimal tau using training data
        print(f' {"=" * 20} Finding optimal value for tau {"=" * 20}\n')
        opt_tau, avg_opt_tau = find_opt_tau(train_dist, train_labels, tau_step)

        # Make predictions
        pred = make_predictions(test_dist, train_labels, opt_tau)

        # Make output dataframe
        df_output = pd.DataFrame({'file':test_files,
                                  'prediction': pred,
                                  'truth': test_labels,
                                 'is_correct?': pred == test_labels})
        # Save csv file
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_name = os.path.join(out_dir, f'{out_file[i]}.csv')
        df_output.to_csv(out_name, index = False)
        
        test_acc = sum(pred == test_labels) / len(pred)
        print(f' {"=" * 20} For {out_file[i]} the test accuracy is {test_acc:.4f} {"=" * 20}\n')

if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config)
    print(f" {'='*25} Predictions saved {'='*25}\n")