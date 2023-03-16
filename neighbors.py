import os
import pickle
import torch
import resnet
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import image2tensor
from torch.utils.data import DataLoader
from datasets import MaskedImageDataset
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def mask_image(source_img: torch.Tensor, mask_size: int, stride: int):

    image_size = source_img.shape[1:]
    non_normed_masks = torch.zeros(image_size) # Change name to norm_factor
    masked_neighbor_image = torch.Tensor() # Change name to masked_images
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
                non_normed_masks += 1 - mask # invert to get area that is masked, e.g. from [1, 1, 0, 1, 1] to [0, 0, 1, 0, 0]

    return masked_neighbor_image, non_normed_masks, indices

def get_latent_masks(model: resnet.ResNetAutoEncoder,
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
            feat_map = model.encoder(batch).cpu().detach()
            feat_map = feat_map.flatten(start_dim = 1)
            f_maps = torch.concat((f_maps, feat_map))

    # Scale feature maps
    f_maps = scaler.transform(f_maps)

    # Apply PCA (reduce dimensionality)
    f_maps = pca.transform(f_maps)

    return f_maps

def get_sd_map(img_neighbor: torch.Tensor,
               latent_query: np.ndarray,
               unmask_dist: float,
               mask_size: int,
               stride: int,
               model: resnet.ResNetAutoEncoder,
               pca: sklearn.decomposition.PCA,
               scaler: sklearn.preprocessing.StandardScaler) -> (torch.Tensor, torch.Tensor):
    
    image_size = img_neighbor.shape[1:]
    
    # Make masked images
    masked_image, norm_factor, idx_mask = mask_image(img_neighbor, mask_size, stride)
    
    # Get latent representations of each masked image
    latent_masks = get_latent_masks(model, masked_image, pca, scaler)
    
    # Compute distances between the latent representation of
    # query image and the latent representation
    # of masked versions of the neighbor image
    masked_distances = cdist(latent_query[np.newaxis, :], latent_masks, metric = 'euclidean')[0]
    
    # Compute importance of masked regions
    sim_importance_masked_regions = [max(mask_dist - unmask_dist, 0) for mask_dist in masked_distances]
    dissim_importance_masked_regions = [max(unmask_dist - mask_dist, 0) for mask_dist in masked_distances]
    
    # Similarity
    min_imp = min(sim_importance_masked_regions)
    max_imp = max(sim_importance_masked_regions)
    scaled_sim_importance_masked_regions = np.array([(f-min_imp) / (max_imp - min_imp)  for f in sim_importance_masked_regions])
    
    # Dissimilarity
    min_imp = min(dissim_importance_masked_regions)
    max_imp = max(dissim_importance_masked_regions)
    scaled_dissim_importance_masked_regions = np.array([(f-min_imp) / (max_imp - min_imp)  for f in dissim_importance_masked_regions])
    
    # The normalization factors considers the number of times
    # a region is overlapped by a mask
    norm = np.exp(-norm_factor)
    #norm = 1 / norm_factor
    
    
    # Saliency Maps
    sim_based_sm = np.zeros(image_size)
    dissim_based_sm = np.zeros(image_size)
    
    for i, e in enumerate(idx_mask):
        mask = np.zeros(image_size)
        start_row = e[0][0]
        end_row = e[0][1]
        start_col = e[1][0]
        end_col = e[1][1]
        #mask[start_row:end_row, start_col:end_col] = norm[start_row:end_row, start_col:end_col]
        mask[start_row:end_row, start_col:end_col] = 1.0
        sim_based_sm += mask * scaled_sim_importance_masked_regions[i]
        dissim_based_sm +=  mask * scaled_dissim_importance_masked_regions[i]
    
    # Normalize to [0, 1]
    sim_based_sm /= sim_based_sm.max()
    dissim_based_sm /= dissim_based_sm.max()
    
    return sim_based_sm, dissim_based_sm

# ----------------------- CONFIG FILE -------------------------------------

# File containing the predictions from a classifier over the training set
train_pred_file = "./70_by_70_predictions_images_2010_2017/pred_BB_Buy_classif.csv"

# File containing the predictions from a classifier over the test set
test_pred_file = "./70_by_70_predictions_images_2018/pred_BB_Buy_classif.csv"

# File containing the latent representations of the images in the train set
train_lat_file = "./70_by_70_pca/train_pca_BB_Buy_autoencoder.pkl"

# File containing the latent representations of the images in the test set
test_lat_file = "./70_by_70_pca/test_pca_BB_Buy_autoencoder.pkl"

# File containing the path of a trained autoencoder
path_autoencoder = './70_by_70_autoencoders/BB_Buy_autoencoder.pth'

# File containing a PCA object trained with the training images
path_pca = './70_by_70_pca/pca_BB_Buy_autoencoder.pkl'

# File containing the scaler used to train the PCA
path_scaler = './70_by_70_pca/scaler_BB_Buy_autoencoder.pkl'

# Number of query predictions from the test set to analyze
num_query = 3

# Type of prediction (True => right prediction, False => Wrong prediction)
bool_type = True

# Number of neighbors to compare with (neighbors come from training data)
num_neighbors = 3

# Output Figure size
figsize = (40, 40)

# Image size
image_size = (70, 70)

# Mask size
mask_size = 8 # Need to think how to select this parameter

# stride size
stride = 3 # Need to think how to select this parameter

# Labels for positive/negative class
lab_pos = 'Buy'
lab_neg = 'No Buy'
# ---------------------END OF CONFIG FILE ------------------------

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
model = resnet.ResNetAutoEncoder(image_size)
checkpoint = torch.load(path_autoencoder)
model.load_state_dict(checkpoint['model_state_dict'])
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

# Fit nearest neighbors object
knn = NearestNeighbors(n_neighbors = num_neighbors)
knn.fit(latent_train.iloc[:, 1:].to_numpy())
fig, axs = plt.subplots(2 * num_query, num_neighbors + 1, figsize = figsize)

# TO DO: Set seed

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

    # Get neighbors and orders them (increasingly) according to their distance
    # to the latent representation of the query image
    dist, idx_neighbors = knn.kneighbors(latent_query.reshape((1, -1)), return_distance = True)
    dist = dist.reshape((dist.shape[1], ))
    idx_neighbors = idx_neighbors.reshape((idx_neighbors.shape[1], ))
    idx_neighbors = idx_neighbors[dist.argsort()[::-1]]

    # Plot row
    for j, idx_n in enumerate(idx_neighbors):
        
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
        sim_map, dis_map = get_sd_map(img_neighbor,
                                     latent_query,
                                     unmask_dist,
                                     mask_size,
                                     stride,
                                     model,
                                     pca,
                                     scaler)
        # Avoid plotting background part
        #sim_mask = np.ma.masked_greater(sim_map, 0)
        #dis_mask = np.ma.masked_greater(dis_map, 0)
        sim_map = torch.from_numpy(sim_map)
        sim_map = sim_map.unsqueeze(0).permute([1, 2, 0])
        sim_map[sim_map == 0] = torch.nan
        
        dis_map = torch.from_numpy(dis_map)
        dis_map = dis_map.unsqueeze(0).permute([1, 2, 0])
        dis_map[dis_map == 0] = torch.nan

        # Plot neighbor image with similarity map
        img_neighbor = img_neighbor.permute([1, 2, 0])
        axs[i, j + 1].imshow(img_neighbor)
        #axs[i, j + 1].imshow(sim_mask, alpha = 0.6, cmap = 'Greens')
        axs[i, j + 1].imshow(sim_map, alpha = 0.6, cmap = 'Greens')
        axs[i, j + 1].set_title(f'Pred: {pred_neighbor}\n True: {true_neighbor}')
        axs[i,j + 1].set_xticks([])
        axs[i,j + 1].set_yticks([])
        
        # Plot neighbor image with dissimilarity map
        axs[i + 1, j + 1].imshow(img_neighbor)
        #axs[i + 1, j + 1].imshow(dis_mask, alpha = 0.6, cmap = 'Greens')
        axs[i + 1, j + 1].imshow(dis_map, alpha = 0.6, cmap = 'Greens')
        axs[i + 1, j + 1].set_title('')
        axs[i + 1,j + 1].set_xticks([])
        axs[i + 1,j + 1].set_yticks([])
        
    # Plot query image with similarity map
    img_query = img_query.permute([1, 2, 0])
    axs[i, 0].imshow(img_query)
    #axs[i, 0].imshow(sim_mask, alpha = 0.6, cmap = 'Greens')
    axs[i, 0].imshow(sim_map, alpha = 0.6, cmap = 'Greens')
    axs[i, 0].set_title(f'Pred: {pred_query}\n True: {true_query}')
    axs[i, 0].set_xticks([])
    axs[i, 0].set_yticks([])
    
    # Plot query image with dissimilarity map
    axs[i + 1, 0].imshow(img_query)
    #axs[i + 1, 0].imshow(dis_mask, alpha = 0.6, cmap = 'Greens')
    axs[i + 1, 0].imshow(dis_map, alpha = 0.6, cmap = 'Greens')
    axs[i + 1, 0].set_title('')
    axs[i + 1, 0].set_xticks([])
    axs[i + 1, 0].set_yticks([])
    
    i = i + 2

plt.show()
