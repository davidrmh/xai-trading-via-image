import os
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from utils import image2tensor
from classifiers import ClassifierAutoEncoder
from sklearn.neighbors import NearestNeighbors
from interpret_latent_classif_autoencoder import get_sd_map, overall_level


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the JSON file with the input arguments',
                   type = str)
args = parser.parse_args()

def get_labels(files: pd.Series, path_images: list[str, str]):
    files_pos = os.listdir(path_images[0])
    files_neg = os.listdir(path_images[1])
    
    if files.shape[0] != len(files_pos) + len(files_neg):
        raise Exception('The number of files is not the same as in path_images')
    labels = [0] * files.shape[0]
    
    for i, f in enumerate(files):
        # For compatibility with windows path
        f = f.replace('\\', '/')
        f = f.split('/')[-1]
        if f in files_pos:
            labels[i] = 1
    return np.array(labels)

def apply_rules(train_labels: np.ndarray,
               idx_neighbors: np.ndarray,
               overall_sim_pos_class: float,
               overall_sim_neg_class: float) -> int:
    
    # If neighbors have positive label
    # and the average similarity is greater than the average disimilarity
    # then predict the positive label
    #if np.all(train_labels[idx_neighbors] == 1) and overall_sim_pos_class > 0:
    #    return 1
    
    # If neighbors have positive label
    # but in average there the disimilarity level is greater than the similarity
    # then predict the negative label
    #elif np.all(train_labels[idx_neighbors] == 1) and overall_sim_pos_class < 0:
    #    return 0
    
    # Similar logic as above
    #elif np.all(train_labels[idx_neighbors] == 0) and overall_sim_neg_class > 0:
    #    return 0
    
    # Similar logic as above
    #elif np.all(train_labels[idx_neighbors] == 0) and overall_sim_neg_class < 0:
    #    return 1
    
    # If neighbors have mixed labels then consider the label of the most similar group
    #else:
    #    if overall_sim_pos_class > overall_sim_neg_class:
    #        return 1
    #    else:
    #        return 0

    if np.all(train_labels[idx_neighbors] == 1):
        return 1
    elif np.all(train_labels[idx_neighbors] == 0):
        return 0
    elif overall_sim_pos_class > overall_sim_neg_class:
        return 1
    else:
        return 0

def main(config: dict):

    # Path of the train images (positve/negatice class)
    path_train_images = config['path_train_images']

    # Path of the test images (positve/negatice class)
    path_test_images = config['path_test_images']

    # File containing the latent representations of the images in the train set
    train_lat_file = config['train_lat_file']

    # File containing the latent representations of the images in the test set
    # Ues the train set to analyze the decisions over the trained data
    test_lat_file = config['test_lat_file']

    # File containing the path of a trained autoencoder
    path_autoencoder = config['path_autoencoder']

    # Number of neighbors to compare with (neighbors come from training data)
    num_neighbors = config['num_neighbors']

    # Input image size
    image_size = config['image_size']

    # Mask size
    mask_size = config['mask_size']

    # stride size
    stride = config['stride']

    # Path where to store the results
    out_path = config['out_path']

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(len(path_autoencoder)):
        print(f' {"=" * 20} Working With Model {path_autoencoder[i]} {"=" * 20}\n')

        # Latent representations of the train images
        with open(train_lat_file[i], 'rb') as f:
            train_latent = pickle.load(f)

        # Labels of train files
        train_labels = get_labels(train_latent['file'], path_train_images[i])

        # Latent representations of the test images
        with open(test_lat_file[i], 'rb') as f:
            test_latent = pickle.load(f)

        # Labels of test files
        test_labels = get_labels(test_latent['file'], path_test_images[i])

        # Model
        model = ClassifierAutoEncoder(image_size)
        checkpoint = torch.load(path_autoencoder[i])
        model.load_state_dict(checkpoint)
        model.train(False)
        model.to(device)

        # Nearest neighbors object using train data
        # Considers the case when test set is equal as train set
        # and, trivially, one of the neighbors is a point itself
        if train_lat_file[i] == test_lat_file[i]:
            neighbors = NearestNeighbors(n_neighbors = num_neighbors + 1)
        else:
            neighbors = NearestNeighbors(n_neighbors = num_neighbors)
        neighbors.fit(train_latent.iloc[:, 1:].to_numpy(dtype = np.float64))
        
        pred_labels = np.zeros(test_latent.shape[0])
        for j in range(test_latent.shape[0]):
            latent_query = test_latent.iloc[j, 1:].to_numpy(dtype = np.float64)

            # Get neighbors from the training set
            dist_neighbors, idx_neighbors = neighbors.kneighbors(latent_query.reshape((1, -1)), return_distance = True)
            dist_neighbors = dist_neighbors.reshape((-1,))
            idx_neighbors = idx_neighbors.reshape((-1, ))

            avg_sim_pos_class = []
            avg_sim_neg_class = []
            avg_dis_pos_class = []
            avg_dis_neg_class = []
            for idx in idx_neighbors:
                # Latent representation of the neighbor
                latent_neighbor = train_latent.iloc[idx, 1:].to_numpy(dtype = np.float64)

                # Image of the neighbor
                image_neighbor = image2tensor(train_latent['file'][idx])

                # Distance (in latent space) between query and neighbor
                unmask_dist = np.sqrt( sum((latent_query - latent_neighbor)**2) )

                # S/D Maps
                sim_map, dis_map, sim_map_focused, dis_map_focused = get_sd_map(image_neighbor,
                                                                               latent_query,
                                                                               unmask_dist,
                                                                               mask_size,
                                                                               stride,
                                                                               model)
                # Positive class
                if train_labels[idx] == 1:
                    avg_sim_pos_class.append(sim_map_focused[sim_map_focused > 0].mean())
                    avg_dis_pos_class.append(dis_map_focused[dis_map_focused > 0].mean())
                # Negative class
                elif train_labels[idx] == 0:
                    avg_sim_neg_class.append(sim_map_focused[sim_map_focused > 0].mean())
                    avg_dis_neg_class.append(dis_map_focused[dis_map_focused > 0].mean())

            overall_sim_pos_class = overall_level(avg_sim_pos_class, avg_dis_pos_class, num_neighbors)
            overall_sim_neg_class = overall_level(avg_sim_neg_class, avg_dis_neg_class, num_neighbors)

            pred_labels[j] = apply_rules(train_labels,
                                        idx_neighbors,
                                        overall_sim_pos_class,
                                        overall_sim_neg_class)
            if (j + 1) % 100 == 0:
                print(f' {"@" * 20} Finished With {j + 1}/{test_latent.shape[0]} Latent Queries {"@" * 20}\n')
        # Create output dataframe
        df_output = pd.DataFrame({'file': test_latent['file'],
                                 'prediction': pred_labels,
                                 'truth': test_labels,
                                 'is_corrrect?': pred_labels == test_labels})

        # Save File
        aux = path_autoencoder[i].split('/')[-1]
        aux = aux.replace('.pth', '.csv')
        aux = f'pred_neigh_latent_{aux}'
        out_file = os.path.join(out_path, aux)
        print(f' {"=" * 20} Saving file {out_file} {"=" * 20}\n')
        df_output.to_csv(out_file, index = False)

if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config)
    print(f' {"=" * 25} Predictions Saved {"=" * 25}')