import os
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from functools import partial
from utils import image2tensor
from multiprocessing import Pool
from interpret_image_distance import get_sd_map, overall_level, distance_uqi


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the JSON file with the input arguments',
                   type = str)
args = parser.parse_args()

def get_labels(path_images: list[str, str]):
    files_pos = os.listdir(path_images[0])
    files_neg = os.listdir(path_images[1])
    
    labels = []
    files = []
    
    # Images with positive class
    for f in files_pos:
        labels.append(1)
        files.append(os.path.join(path_images[0], f))
    
    # Images with negative class
    for f in files_neg:
        labels.append(1)
        files.append(os.path.join(path_images[1], f))
    
    return np.array(labels), np.array(files)

def apply_rules1(train_labels: np.ndarray,
               idx_neighbors: np.ndarray,
               overall_sim_pos_class: float,
               overall_sim_neg_class: float) -> int:
    
    if np.all(train_labels[idx_neighbors] == 1):
        return 1
    elif np.all(train_labels[idx_neighbors] == 0):
        return 0
    elif overall_sim_pos_class > overall_sim_neg_class:
        return 1
    else:
        return 0

def apply_rules2(train_labels: np.ndarray,
               idx_neighbors: np.ndarray,
               overall_sim_pos_class: float,
               overall_sim_neg_class: float) -> int:
    
    # If neighbors have positive label
    # and the average similarity is greater than the average disimilarity
    # then predict the positive label
    if np.all(train_labels[idx_neighbors] == 1) and overall_sim_pos_class > 0:
        return 1
    
    # If neighbors have positive label
    # but in average there the disimilarity level is greater than the similarity
    # then predict the negative label
    elif np.all(train_labels[idx_neighbors] == 1) and overall_sim_pos_class < 0:
        return 0
    
    # Similar logic as above for negative label
    elif np.all(train_labels[idx_neighbors] == 0) and overall_sim_neg_class > 0:
        return 0
    
    # Similar logic as above for negative label
    elif np.all(train_labels[idx_neighbors] == 0) and overall_sim_neg_class < 0:
        return 1
    
    # If neighbors have mixed labels then consider the label of the most similar group
    else:
        if overall_sim_pos_class > overall_sim_neg_class:
            return 1
        else:
            return 0

def main(config: dict):

    # Path of the train images (positve/negatice class)
    path_train_images = config['path_train_images']

    # Path of the test images (positve/negatice class)
    path_test_images = config['path_test_images']

    # Number of neighbors to compare with (neighbors come from training data)
    num_neighbors = config['num_neighbors']

    # Mask size
    mask_size = config['mask_size']

    # stride size
    stride = config['stride']
    
    # Type of rule to apply
    type_rule = config['type_rule']

    # Path where to store the results
    out_path = config['out_path']

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    pool = Pool(processes = 6)
    for i in range(len(path_test_images)):
        print(f' {"=" * 20} Working With Images {path_test_images[i]} {"=" * 20}\n')

        # Labels of train files
        train_labels, train_files = get_labels(path_train_images[i])
        train_images = [image2tensor(file).permute([1, 2, 0]).numpy() for file in train_files]

        # Labels of test files
        test_labels, test_files = get_labels(path_test_images[i])
        
        pred_labels = np.zeros(test_labels.shape[0])
        for j in range(test_files.shape[0]):
            # Get query image
            # torch tensor with shape (C, H, W)
            img_query = image2tensor(test_files[j])

            # Get neighbors from the training set
            # The closeness criterion is UQI function
            #distance_neighbors = np.zeros(train_files.shape[0])
            partial_uqi = partial(distance_uqi, img_query.permute([1, 2, 0]).numpy(), ws = mask_size)
            distance_neighbors = np.array(pool.map(partial_uqi, train_images))
            #for k, file in enumerate(train_files):
            #    img_neighbor = image2tensor(file).permute([1, 2, 0]).numpy()
                #d = distance_uqi(img_query.permute([1, 2, 0]).numpy(), img_neighbor, ws = mask_size)
                #distance_neighbors[k] = d
            
            # Sorts in decreasing order since the UQI index
            # lies in [-1, 1] taking the value of 1 only
            # when both arguments are the same image
            idx_neighbors = distance_neighbors.argsort()[::-1]
            
            # Get Nearest neighbors
            # The if is used to make the analysis over the training set
            # When the test set is set equal to the training set we have
            # to discard the query image as neighbor (distance 0)
            if path_test_images[i] == path_train_images[i]:
                idx_neighbors = idx_neighbors[1:num_neighbors + 1]
            else:
                idx_neighbors = idx_neighbors[0:num_neighbors]
            
            avg_sim_pos_class = []
            avg_sim_neg_class = []
            avg_dis_pos_class = []
            avg_dis_neg_class = []
            for idx in idx_neighbors:
                # Image of the neighbor
                img_neighbor = image2tensor(train_files[idx])

                # "Distance" between query and neighbor
                unmask_dist = distance_uqi(img_query.permute([1, 2, 0]).numpy(), img_neighbor.permute([1, 2, 0]).numpy())

                # S/D Maps
                sim_map, dis_map, sim_map_focused, dis_map_focused = get_sd_map(img_query,
                                                                                img_neighbor,
                                                                                unmask_dist,
                                                                                mask_size,
                                                                                stride)
                # Positive class
                if train_labels[idx] == 1:
                    if len(sim_map_focused[sim_map_focused > 0]) == 0:
                        avg_sim_pos_class.append(0)
                    else:
                        avg_sim_pos_class.append(sim_map_focused[sim_map_focused > 0].mean())
                    if len(dis_map_focused[dis_map_focused > 0]) == 0:
                        avg_dis_pos_class.append(0)
                    else:
                        avg_dis_pos_class.append(dis_map_focused[dis_map_focused > 0].mean())
                # Negative class
                elif train_labels[idx] == 0:
                    if len(sim_map_focused[sim_map_focused > 0]) == 0:
                        avg_sim_neg_class.append(0)
                    else:
                        avg_sim_neg_class.append(sim_map_focused[sim_map_focused > 0].mean())
                    if len(dis_map_focused[dis_map_focused > 0]) == 0:
                        avg_dis_neg_class.append(0)
                    else:
                        avg_dis_neg_class.append(dis_map_focused[dis_map_focused > 0].mean())

            overall_sim_pos_class = overall_level(avg_sim_pos_class, avg_dis_pos_class, num_neighbors)
            overall_sim_neg_class = overall_level(avg_sim_neg_class, avg_dis_neg_class, num_neighbors)
            
            if type_rule == 1:
                pred_labels[j] = apply_rules1(train_labels,
                                              idx_neighbors,
                                              overall_sim_pos_class,
                                              overall_sim_neg_class)
            elif type_rule == 2:
                pred_labels[j] = apply_rules2(train_labels,
                                              idx_neighbors,
                                              overall_sim_pos_class,
                                              overall_sim_neg_class)
                
            if (j + 1) % 100 == 0:
                print(f' {"@" * 20} Finished With {j + 1}/{test_files.shape[0]} Queries {"@" * 20}\n')
        # Create output dataframe
        df_output = pd.DataFrame({'file': test_files,
                                 'prediction': pred_labels,
                                 'truth': test_labels,
                                 'is_correct?': pred_labels == test_labels})

        # Save File
        out_file = path_test_images[i][0].split('/')[-1] + f'_rule_{type_rule}.csv'
        out_file = os.path.join(out_path, out_file)
        print(f' {"=" * 20} Saving file {out_file} {"=" * 20}\n')
        df_output.to_csv(out_file, index = False)

if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config)
    print(f' {"=" * 25} Predictions Saved {"=" * 25}')