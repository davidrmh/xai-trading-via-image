import os
import pickle
import torch
import sklearn
import copy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import MaskedImageDataset
from scipy.spatial.distance import cdist
from utils import image2tensor, set_seed
from classifiers import ClassifierAutoEncoder
from sklearn.neighbors import NearestNeighbors


# ----------------------- CONFIG FILE -------------------------------------

# File containing the predictions from a classifier over the training set
train_pred_file = "./70_by_70_pred_classif_autoencoder_images_2010_2017/pred_BB_Buy_classif.csv"

# File containing the predictions from a classifier over the test set
test_pred_file = "./70_by_70_pred_classif_autoencoder_images_2010_2017/pred_BB_Buy_classif.csv"

# File containing the latent representations of the images in the train set
train_lat_file = "./70_by_70_pca_classif_autoencoder/train_pca_BB_Buy_classif.pkl"

# File containing the latent representations of the images in the test set
# Ues the train set to analyze the decisions over the trained data
test_lat_file = "./70_by_70_pca_classif_autoencoder/train_pca_BB_Buy_classif.pkl"

# File containing the path of a trained autoencoder
path_autoencoder = './70_by_70_trained_classif_autoencoder/BB_Buy_classif.pth'

# File containing a PCA object trained with the training images
path_pca = './70_by_70_pca_classif_autoencoder/pca_BB_Buy_classif.pkl'

# File containing the scaler used to train the PCA
path_scaler = './70_by_70_pca_classif_autoencoder/scaler_BB_Buy_classif.pkl'

# Number of query predictions from the test set to analyze
num_query = 2

# Type of prediction (True => right prediction, False => Wrong prediction)
bool_type = False

# Number of neighbors to compare with (neighbors come from training data)
num_neighbors = 5

# Output Figure size
figsize = (12, 12)

# DPI
dpi = 300

# Input image size
image_size = (70, 70)

# Mask size
mask_size = 8 # Need to think how to select this parameter

# stride size
stride = 4 # Need to think how to select this parameter

# Labels for positive/negative class
lab_pos = 'Buy'
lab_neg = 'No Buy'

# Colormaps
sim_col_map = 'Greens'
dis_col_map = 'Reds'

# Transparency
alpha = 0.4

# Path where to store the results
out_path = './70_by_70_sd_maps_classif_autoencoder_correct_training'

# name of the file with the output
out_file = 'sd_map1_wrong'

# For reproducibility
seed = 20
# ---------------------END OF CONFIG FILE ------------------------


def overall_level(avg_sim: list[float], avg_dis: list[float]) -> float:
    if len(avg_sim) == 0:
        return 0.0
    overall_lev = 0.0
    for i in range(len(avg_sim)):
        overall_lev += avg_sim[i] - avg_dis[i]
    overall_lev /= len(avg_sim)
    return overall_lev

def mask_image(source_img: torch.Tensor, mask_size: int, stride: int):

    image_size = source_img.shape[1:]
    factor_f = torch.zeros(image_size)
    masked_neighbor_image = torch.Tensor()
    indices = []
    for i in range(0, image_size[0] - mask_size + 1, stride):
        for j in range(0, image_size[1] - mask_size + 1, stride):
            mask = torch.ones(image_size)
            mask[i:min(i + mask_size, image_size[0]), j:min(j + mask_size, image_size[1])] = 0

            # Only consider masked images that do modify the source image
            if not torch.all(mask * source_img == source_img):
                # [(start_row, end_row), (start_col, end_col)]
                indices.append([(i, min(i + mask_size, image_size[0])), (j,min(j + mask_size, image_size[1]))])
                masked_neighbor_image = torch.cat((masked_neighbor_image, (source_img * mask).unsqueeze(0)), 0)
                factor_f[i:min(i + mask_size, image_size[0]), j:min(j + mask_size, image_size[1])] += 1

    return masked_neighbor_image, factor_f, indices

def get_latent_masks(model: ClassifierAutoEncoder,
                     masked_neighbor_image: torch.Tensor,
                     pca: sklearn.decomposition.PCA,
                     scaler: sklearn.preprocessing.StandardScaler) -> torch.Tensor:
    
    dataset = MaskedImageDataset(masked_neighbor_image)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f_maps = torch.tensor([], dtype = torch.float32)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            feat_map, _ = model.encoder(batch)
            feat_map = feat_map.cpu().detach()
            feat_map = feat_map.flatten(start_dim = 1)
            f_maps = torch.concat((f_maps, feat_map))

    # Scale feature maps
    f_maps = scaler.transform(f_maps)

    # Apply PCA (reduce dimensionality)
    f_maps = pca.transform(f_maps)

    return f_maps

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
        if d >= unmask_dist:
            sim_regions.append(idx_mask[i])
            sim_contrib.append(d - unmask_dist)
        
        # Dissimilarity region
        else:
            dis_regions.append(idx_mask[i])
            dis_contrib.append(unmask_dist - d)
    
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

def get_sd_map(img_neighbor: torch.Tensor,
               latent_query: np.ndarray,
               unmask_dist: float,
               mask_size: int,
               stride: int,
               model: ClassifierAutoEncoder,
               pca: sklearn.decomposition.PCA,
               scaler: sklearn.preprocessing.StandardScaler) -> (torch.Tensor, torch.Tensor):
    
    # Make masked images
    masked_image, factor_f, idx_mask = mask_image(img_neighbor, mask_size, stride)
    
    # Get latent representations of each masked image
    latent_masks = get_latent_masks(model, masked_image, pca, scaler)
    
    # Compute distances between the latent representation of
    # query image and the latent representation
    # of masked versions of the neighbor image
    masked_distances = cdist(latent_query[np.newaxis, :], latent_masks, metric = 'euclidean')[0]
    
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# DataFrames with the predictions from a classifier
df_test_pred = pd.read_csv(test_pred_file)
df_train_pred = pd.read_csv(train_pred_file)

# Latent representations of train images
with open(train_lat_file, 'rb') as f:
    latent_train = pickle.load(f)

# Latent representations of test images
with open(test_lat_file, 'rb') as f:
    latent_test = pickle.load(f)

# Trained autoencoder
model = ClassifierAutoEncoder(image_size)
checkpoint = torch.load(path_autoencoder)
model.load_state_dict(checkpoint)
model.train(False)
model.to(device)

# PCA object
with open(path_pca, 'rb') as f:
    pca = pickle.load(f)

# Scaler object
with open(path_scaler, 'rb') as f:
    scaler = pickle.load(f)

# Choose randomly some query images from the test set
query_idx = np.random.choice(df_test_pred[df_test_pred['is_correct?'] == bool_type].index,
                             size = num_query,
                             replace = False)

plt.ioff()
fig, axs = plt.subplots(2 * num_query, num_neighbors + 1, figsize = figsize)

# This if is used to make the analysis over the training set
# When the test set is set equal to the training set we have
# to discard the query image as neighbor (distance 0)
if train_lat_file == test_lat_file:
    knn = NearestNeighbors(n_neighbors = num_neighbors + 1)
else:
    knn = NearestNeighbors(n_neighbors = num_neighbors)
    
# Fit nearest neighbors object
knn.fit(latent_train.iloc[:, 1:].to_numpy())


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
    # Get query image from test set
    path_img_query = df_test_pred.iloc[idx, 0]
    str_aux_query = df_test_pred.iloc[idx, 0].split('/')[-1]
    pred_query = lab_pos if df_test_pred.iloc[idx]['prediction'] == 1 else lab_neg
    true_query = lab_pos if df_test_pred.iloc[idx]['truth'] == 1 else lab_neg

    # TO DO: There must be an easier way of doing this
    for j, f in zip(latent_test['file'].index, latent_test['file']):
        if str_aux_query in f:
            latent_query = latent_test.iloc[j, 1:].to_numpy(dtype = np.float64)
            break
    img_query = image2tensor(path_img_query)

    # Get neighbors of the latent representation of the query image
    dist, idx_neighbors = knn.kneighbors(latent_query.reshape((1, -1)), return_distance = True)
    dist = dist.reshape((dist.shape[1], ))
    idx_neighbors = idx_neighbors.reshape((idx_neighbors.shape[1], ))
    # This if is used to make the analysis over the training set
    # When the test set is set equal to the training set we have
    # to discard the query image as neighbor (distance 0)
    if train_lat_file == test_lat_file:
        idx_neighbors = idx_neighbors[1:]
    
    # To aggregate (dis)similarities of neighbors
    sim_agg = torch.zeros(image_size)
    dis_agg = torch.zeros(image_size)
    
    # To compute overall level of (dis)agreement
    # according to neighbors and their predictions
    list_sim_same = []
    list_sim_dif = []
    list_dis_same = []
    list_dis_dif = []
    
    # To store overall level of (dis)agreement
    ola_same = 0.0 # Overall level of agreement for neighbors with same predicted label as query image
    old_same = 0.0 # Overall level of disagreement for neighbors with same predicted label as query image
    ola_dif = 0.0 # Overall level of agreement for neighbors with different predicted label as query image
    old_dif = 0.0 # Overall level of disagreement for neighbors with different predicted label as query image
    
    # For debbuging
    print(f' {"=" * 20} Query image {idx} {"=" * 20}')

    # Plot row
    for j, idx_n in enumerate(idx_neighbors):
        print(f' {"=" * 20} Neighbor image {idx_n} {"=" * 20}')
        
        # Find labels for neighbor
        str_aux_neighbor = latent_train.iloc[idx_n]['file']
        str_aux_neighbor = str_aux_neighbor.split('\\')[-1]
        
        for k, f_str in enumerate(df_train_pred['file']):
            if str_aux_neighbor in f_str:
                pred_neighbor = lab_pos if df_train_pred.iloc[k]['prediction'] == 1 else lab_neg
                true_neighbor = lab_pos if df_train_pred.iloc[k]['truth'] == 1 else lab_neg

        img_neighbor = image2tensor(latent_train['file'][idx_n])
        latent_neighbor = latent_train.iloc[idx_n, 1:].to_numpy(dtype = np.float64)
        unmask_dist = np.sqrt(sum((latent_query - latent_neighbor)**2))
        
        # Compute S/D Maps
        sim_map, dis_map, sim_map_focused, dis_map_focused = get_sd_map(img_neighbor,
                                                                        latent_query,
                                                                        unmask_dist,
                                                                        mask_size,
                                                                        stride,
                                                                        model,
                                                                        pca,
                                                                        scaler)
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
        #sim_agg += sim_map
        sim_map[sim_map == 0.0] = torch.nan
        
        dis_map = torch.from_numpy(dis_map)
        #dis_agg += dis_map
        dis_map[dis_map == 0.0] = torch.nan

        # Plot neighbor image with similarity map
        img_neighbor = img_neighbor.permute([1, 2, 0])
        img_neighbor = img_neighbor.mean(axis = 2)
        img_neighbor[img_neighbor == 0.0] = torch.nan
        axs[i, j + 1].imshow(img_neighbor, cmap = 'PiYG_r')
        axs[i, j + 1].imshow(sim_map, cmap = cmap_sim, alpha = alpha)
        axs[i, j + 1].set_title(f'Pred: {pred_neighbor}\n True: {true_neighbor}', fontsize = 6)
        axs[i, j + 1].set_xlabel(f'Avg. Sim: {avg_indiv_sim:.4f}', fontsize = 6)
        axs[i, j + 1].set_xticks([])
        axs[i, j + 1].set_yticks([])
        
        # Plot neighbor image with dissimilarity map
        axs[i + 1, j + 1].imshow(img_neighbor, cmap = 'PiYG_r')
        axs[i + 1, j + 1].imshow(dis_map, cmap = cmap_dis, alpha = alpha)
        axs[i + 1, j + 1].set_title(f'Pred: {pred_neighbor}\n True: {true_neighbor}', fontsize = 6)
        axs[i + 1, j + 1].set_xlabel(f'Avg. Dis: {avg_indiv_dis:.4f}', fontsize = 6)
        axs[i + 1, j + 1].set_xticks([])
        axs[i + 1, j + 1].set_yticks([])
    
    # Compute overall level of (dis)agreement
    ola_same = overall_level(list_sim_same, list_dis_same)
    ola_dif = overall_level(list_sim_dif, list_dis_dif)
    #old_same = overall_level(list_dis_same)
    #ola_dif = overall_level(list_sim_dif)
    #old_dif = overall_level(list_dis_dif)
    #total_ola = ola_same + ola_dif
    #total_old = old_same + old_dif
    
    # Plot query image with aggregated similarity maps
    #sim_agg /= sim_agg.max()
    #sim_agg[sim_agg == 0.0] = torch.nan
    img_query = img_query.permute([1, 2, 0])
    img_query = img_query.mean(axis = 2)
    img_query[img_query == 0.0] = torch.nan
    axs[i, 0].imshow(img_query, cmap = 'PiYG_r')
    #axs[i, 0].imshow(sim_agg / num_neighbors, alpha = alpha, cmap = cmap_sim,
    #                vmin = np.nanmin(sim_agg / num_neighbors),
    #                vmax = np.nanmax(sim_agg / num_neighbors))
    axs[i, 0].set_title(f'Pred: {pred_query}\n True: {true_query}', fontsize = 6)
    axs[i, 0].set_xlabel(f'Ag_Same: {ola_same:.4f}\n Ag_Dif: {ola_dif:.4f}', fontsize = 6)
    axs[i, 0].set_xticks([])
    axs[i, 0].set_yticks([])
    
    # Plot query image with aggregated dissimilarity maps
    #dis_agg /= dis_agg.max()
    #dis_agg[dis_agg == 0.0] = torch.nan
    axs[i + 1, 0].imshow(img_query, cmap = 'PiYG_r')
    #axs[i + 1, 0].imshow(dis_agg / num_neighbors, alpha = alpha, cmap = cmap_dis,
    #                    vmin = np.nanmin(dis_agg / num_neighbors),
    #                    vmax = np.nanmin(dis_agg / num_neighbors))
    axs[i + 1, 0].set_title(f'Pred: {pred_query}\n True: {true_query}', fontsize = 5)
    #axs[i + 1, 0].set_xlabel(f'OLD_D: {old_dif:.4f} \n OLD_S: {old_same:.4f}\n Tot. OLD {total_old:.4f}', fontsize = 5)
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
print(f' {"=" * 25} Image created {"=" * 25}')
