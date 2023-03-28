import os
import json
import pickle
import torch
import argparse
import copy as cp
import numpy as np
import pandas as pd
import sewar.full_ref as sw
import matplotlib.pyplot as plt
from utils import image2tensor, set_seed


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the JSON file with the input arguments',
                   type = str)
args = parser.parse_args()

def distance_uqi(img1: np.ndarray, img2: np.ndarray, **kwargs):
    """
    img1 and img2 must be numpy ndarrays with shape (H, W, C)
    """
    return sw.uqi(img1, img2, **kwargs)

def overall_level(avg_sim: list[float], avg_dis: list[float], num_neigh: int) -> float:
    if len(avg_sim) == 0:
        return 0.0
    overall_lev = 0.0
    for i in range(len(avg_sim)):
        overall_lev += avg_sim[i] - avg_dis[i]
    overall_lev = num_neigh * overall_lev / len(avg_sim)
    return overall_lev

def mask_image(source_img: torch.Tensor, mask_size: int, stride: int):

    input_image_size = source_img.shape[1:]
    factor_f = torch.zeros(input_image_size)
    masked_neighbor_image = torch.Tensor()
    indices = []
    for i in range(0, input_image_size[0] - mask_size + 1, stride):
        for j in range(0, input_image_size[1] - mask_size + 1, stride):
            mask = torch.ones(input_image_size)
            mask[i:min(i + mask_size, input_image_size[0]), j:min(j + mask_size, input_image_size[1])] = 0

            # Only consider masked images that do modify the source image
            if not torch.all(mask * source_img == source_img):
                # [(start_row, end_row), (start_col, end_col)]
                indices.append([(i, min(i + mask_size, input_image_size[0])), (j,min(j + mask_size, input_image_size[1]))])
                masked_neighbor_image = torch.cat((masked_neighbor_image, (source_img * mask).unsqueeze(0)), 0)
                factor_f[i:min(i + mask_size, input_image_size[0]), j:min(j + mask_size, input_image_size[1])] += 1

    return masked_neighbor_image, factor_f, indices

def get_regions(unmask_dist: float,
                masked_distances: np.ndarray,
                idx_mask: list):
    sim_regions = []
    dis_regions = []
    sim_contrib = []
    dis_contrib = []
    
    # Find similarity regions
    for i, d in enumerate(masked_distances):
        
        # Similarity region
        # Since UQI is in [-1, 1] and UQI(x, x) = 1
        # then, if the UQI of the masked image decreases
        # this means that the masked region is a similar region
        # with respect the query image
        if d <= unmask_dist:
            sim_regions.append(idx_mask[i])
            sim_contrib.append(unmask_dist - d)
        
        # Dissimilarity region
        else:
            dis_regions.append(idx_mask[i])
            dis_contrib.append(d - unmask_dist)
    
    # Compute contributions
    if len(sim_contrib) > 0:
        sim_contrib = np.array(sim_contrib)
        range_sim = sim_contrib.max() - sim_contrib.min()
        sim_contrib = (sim_contrib - sim_contrib.min()) / range_sim

    if len(dis_contrib) > 0:
        dis_contrib = np.array(dis_contrib)
        range_dis = dis_contrib.max() - dis_contrib.min()
        dis_contrib = (dis_contrib - dis_contrib.min()) / range_dis
    
    return sim_regions, sim_contrib, dis_regions, dis_contrib

def get_sd_map(img_query: torch.Tensor,
               img_neighbor: torch.Tensor,
               unmask_dist: float,
               mask_size: int,
               stride: int):
    
    # Make masked images
    masked_image, factor_f, idx_mask = mask_image(img_neighbor, mask_size, stride)
    
    # Compute UQI between the query image and the masked image
    masked_distances = np.zeros(masked_image.shape[0])
    for i in range(masked_image.shape[0]):
        masked_distances[i] = distance_uqi(img_query.permute([1, 2, 0]).numpy(),
                                          masked_image[i].permute([1, 2, 0]).numpy())
    
    # Obtain (dis)similarity regions and their contributions
    sim_regions, sim_contrib, dis_regions, dis_contrib = get_regions(unmask_dist, masked_distances, idx_mask)
    
    # Saliency Maps
    sim_map = torch.zeros_like(img_neighbor)
    sim_map_focused = torch.zeros_like(img_neighbor)
    dis_map = torch.zeros_like(img_neighbor)
    dis_map_focused = torch.zeros_like(img_neighbor)
    
    # Similarity map
    for i, e in enumerate(sim_regions):
        start_row = e[0][0]
        end_row = e[0][1]
        start_col = e[1][0]
        end_col = e[1][1]
        
        # aux_focused is used to only consider part of img_neighbor that is not background
        img_neighbor_region = img_neighbor[:, start_row:end_row, start_col:end_col]
        aux_focused = torch.zeros_like(img_neighbor_region)
        aux_focused[img_neighbor_region > 0] = 1.0
        aux = torch.ones_like(img_neighbor_region) # To obtain the unfocused version (the one that is plotted)
        
        # Factor to consider overlapping masks
        # factor = np.exp(- factor_f[start_row:end_row, start_col:end_col] / factor_f[start_row:end_row, start_col:end_col].max())
        factor = 1 / factor_f[start_row:end_row, start_col:end_col]
        factor = factor.reshape((1, *factor.shape))
        sim_map[:, start_row:end_row, start_col:end_col] = sim_contrib[i] * factor * aux + sim_map[:, start_row:end_row, start_col:end_col]
        sim_map_focused[:, start_row:end_row, start_col:end_col] = sim_contrib[i] * factor * aux_focused + sim_map_focused[:, start_row:end_row, start_col:end_col]
        
    sim_map = sim_map.permute([1,2,0]).numpy()
    sim_map = sim_map.mean(axis = 2) # To use Colormaps the shape must be (H, W)
    sim_map_focused = sim_map_focused.permute([1,2,0]).numpy()
    sim_map_focused = sim_map_focused.mean(axis = 2)
    
    # Dissimilarity map
    for i, e in enumerate(dis_regions):
        start_row = e[0][0]
        end_row = e[0][1]
        start_col = e[1][0]
        end_col = e[1][1]
        
        # aux is used to only consider part of img_neighbor that is not background
        img_neighbor_region = img_neighbor[:, start_row:end_row, start_col:end_col]
        aux_focused = torch.zeros_like(img_neighbor_region)
        aux_focused[img_neighbor_region > 0] = 1.0
        aux = torch.ones_like(img_neighbor_region) # To obtain the unfocused version (the one that is plotted)
        
        # Factor to consider overlapping masks
        # factor = np.exp(- factor_f[start_row:end_row, start_col:end_col] / factor_f[start_row:end_row, start_col:end_col].max())
        factor = 1 / factor_f[start_row:end_row, start_col:end_col]
        factor = factor.reshape((1, *factor.shape))
        dis_map[:, start_row:end_row, start_col:end_col] = dis_contrib[i] * factor * aux + dis_map[:, start_row:end_row, start_col:end_col]
        dis_map_focused[:, start_row:end_row, start_col:end_col] = dis_contrib[i] * factor * aux_focused + dis_map_focused[:, start_row:end_row, start_col:end_col]
        
    dis_map = dis_map.permute([1,2,0]).numpy()
    dis_map = dis_map.mean(axis = 2) # To use Colormaps the shape must be (H, W)
    dis_map_focused = dis_map_focused.permute([1,2,0]).numpy()
    dis_map_focused = dis_map_focused.mean(axis = 2)
    
    return sim_map, dis_map, sim_map_focused, dis_map_focused

def main(config: dict):
    # File containing the predictions from a classifier over the training set
    train_pred_file = config['train_pred_file']
    
    # File containing the predictions from a classifier over the test set
    test_pred_file = config['test_pred_file']

    # Number of query predictions from the test set to analyze
    num_query = config['num_query']
    
    # Type of prediction (True => right prediction, False => Wrong prediction)
    bool_type = True if config['bool_type'].lower() == 'true' else False
    
    # Number of neighbors to compare with (neighbors come from training data)
    num_neighbors = config['num_neighbors']
    
    # Output Figure size
    figsize = config['figsize']
    
    # DPI
    dpi = config['dpi']
    
    # Mask size
    mask_size = config['mask_size']
    
    # stride size
    stride = config['stride']
    
    # Labels for positive/negative class
    lab_pos = config['lab_pos']
    lab_neg = config['lab_neg']
    
    # Colormaps
    sim_col_map = config['sim_col_map']
    dis_col_map = config['dis_col_map']
    
    # Transparency
    alpha = config['alpha']
    
    # Path where to store the results
    out_path = config['out_path']
    
    # name of the file with the output
    out_file = config['out_file']
    if bool_type:
        out_file = f'{out_file}_right'
    else:
        out_file = f'{out_file}_wrong'
    
    # For reproducibility
    seed = config['seed']

    # DataFrames with the predictions from a classifier
    df_test_pred = pd.read_csv(test_pred_file)
    df_train_pred = pd.read_csv(train_pred_file)

    # Choose randomly some query images from the test set
    query_idx = np.random.choice(df_test_pred[df_test_pred['is_correct?'] == bool_type].index,
                                 size = num_query,
                                 replace = False)

    plt.ioff()
    fig, axs = plt.subplots(2 * num_query, num_neighbors + 1, figsize = figsize)

    # Color maps
    cmap_sim = plt.cm.get_cmap(sim_col_map).copy()
    cmap_dis = plt.cm.get_cmap(dis_col_map).copy()

    # Set seed (for reproducibility)
    set_seed(seed)

    # Create output directory
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    i = 0
    for idx in query_idx:
        # For debbuging
        print(f' {"=" * 20} Query image {idx} {"=" * 20}')
        
        # Get query image from test set
        path_img_query = df_test_pred.iloc[idx, 0]
        img_query = image2tensor(path_img_query)
        pred_query = lab_pos if df_test_pred.iloc[idx]['prediction'] == 1 else lab_neg
        true_query = lab_pos if df_test_pred.iloc[idx]['truth'] == 1 else lab_neg
        
        distance_neighbors = np.zeros(df_train_pred.shape[0])
        for k, file in enumerate(df_train_pred['file']):
            img_neighbor = image2tensor(file).permute([1, 2, 0]).numpy()
            d = distance_uqi(img_query.permute([1, 2, 0]).numpy(), img_neighbor, ws = mask_size)
            distance_neighbors[k] = d
        
        # Sorts in decreasing order since the UQI index
        # lies in [-1, 1] taking the value of 1 only
        # when both arguments are the same image
        idx_neighbors = distance_neighbors.argsort()[::-1]

        # Get Nearest neighbors
        # The if is used to make the analysis over the training set
        # When the test set is set equal to the training set we have
        # to discard the query image as neighbor (distance 0)
        if train_pred_file == test_pred_file:
            idx_neighbors = idx_neighbors[1:num_neighbors + 1]
        else:
            idx_neighbors = idx_neighbors[0:num_neighbors]

        # To compute overall level of (dis)agreement
        # according to neighbors and their predictions
        list_sim_same = []
        list_sim_dif = []
        list_dis_same = []
        list_dis_dif = []

        # To store overall level of (dis)agreement
        ola_same = 0.0 # Overall level of agreement for neighbors with same predicted label as query image
        ola_dif = 0.0 # Overall level of agreement for neighbors with different predicted label as query image

        # Plot row
        for j, idx_n in enumerate(idx_neighbors):
            print(f' {"=" * 20} Neighbor image {idx_n} {"=" * 20}')

            pred_neighbor = lab_pos if df_train_pred.iloc[idx_n]['prediction'] == 1 else lab_neg
            true_neighbor = lab_pos if df_train_pred.iloc[idx_n]['truth'] == 1 else lab_neg

            img_neighbor = image2tensor(df_train_pred['file'][idx_n])
            unmask_dist = distance_uqi(img_query.permute([1, 2, 0]).numpy(), img_neighbor.permute([1, 2, 0]).numpy())

            # Compute S/D Maps
            sim_map, dis_map, sim_map_focused, dis_map_focused = get_sd_map(img_query,
                                                                            img_neighbor,
                                                                            unmask_dist,
                                                                            mask_size,
                                                                            stride)
            avg_indiv_sim = sim_map_focused[sim_map_focused > 0].mean()
            avg_indiv_dis = dis_map_focused[dis_map_focused > 0].mean()
            
            # Store in corresponding lists in order to compute the
            # overall level of (dis)agreement.
            if pred_query == pred_neighbor:
                list_sim_same.append(avg_indiv_sim)
                list_dis_same.append(avg_indiv_dis)
            else:
                list_sim_dif.append(avg_indiv_sim)
                list_dis_dif.append(avg_indiv_dis)

            # Make plots for neighbors
            sim_map = torch.from_numpy(sim_map)
            sim_map[sim_map == 0.0] = torch.nan

            dis_map = torch.from_numpy(dis_map)
            dis_map[dis_map == 0.0] = torch.nan

            # Plot neighbor image with similarity map
            img_neighbor = img_neighbor.permute([1, 2, 0])
            img_neighbor = img_neighbor.mean(axis = 2)
            img_neighbor[img_neighbor == 0.0] = torch.nan
            axs[i, j + 1].imshow(img_neighbor, cmap = 'RdYlGn_r')
            axs[i, j + 1].imshow(sim_map, cmap = cmap_sim, alpha = alpha)
            axs[i, j + 1].set_title(f'Pred: {pred_neighbor}\n True: {true_neighbor}', fontsize = 6)
            axs[i, j + 1].set_xlabel(f'Avg. Sim: {avg_indiv_sim:.4f}', fontsize = 6)
            axs[i, j + 1].set_xticks([])
            axs[i, j + 1].set_yticks([])

            # Plot neighbor image with dissimilarity map
            axs[i + 1, j + 1].imshow(img_neighbor, cmap = 'RdYlGn_r')
            axs[i + 1, j + 1].imshow(dis_map, cmap = cmap_dis, alpha = alpha)
            axs[i + 1, j + 1].set_title(f'Pred: {pred_neighbor}\n True: {true_neighbor}', fontsize = 6)
            axs[i + 1, j + 1].set_xlabel(f'Avg. Dis: {avg_indiv_dis:.4f}', fontsize = 6)
            axs[i + 1, j + 1].set_xticks([])
            axs[i + 1, j + 1].set_yticks([])

        # Compute overall level of (dis)agreement
        ola_same = overall_level(list_sim_same, list_dis_same, num_neighbors)
        ola_dif = overall_level(list_sim_dif, list_dis_dif, num_neighbors)

        # Plot query image with aggregated similarity maps
        img_query = img_query.permute([1, 2, 0])
        img_query = img_query.mean(axis = 2)
        img_query[img_query == 0.0] = torch.nan
        axs[i, 0].imshow(img_query, cmap = 'RdYlGn_r')
        axs[i, 0].set_title(f'Pred: {pred_query}\n True: {true_query}', fontsize = 6)
        axs[i, 0].set_xlabel(f'Ag_Same: {ola_same:.4f}\n Ag_Dif: {ola_dif:.4f}', fontsize = 6)
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])

        # Plot query image with aggregated dissimilarity maps
        axs[i + 1, 0].imshow(img_query, cmap = 'RdYlGn_r')
        axs[i + 1, 0].set_title(f'Pred: {pred_query}\n True: {true_query}', fontsize = 5)
        axs[i + 1, 0].set_xticks([])
        axs[i + 1, 0].set_yticks([])

        i = i + 2
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(os.path.join(out_path, f'{out_file}.png'),
               backend = 'Agg',
               facecolor = 'white',
               dpi = dpi,
               transparent = True)
    plt.tight_layout()
    plt.close()
    plt.ion()

if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config)
    print(f' {"=" * 25} Image created {"=" * 25}')