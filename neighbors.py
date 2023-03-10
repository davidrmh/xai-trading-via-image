import os
import pickle
import numpy as np
import pandas as pd
from utils import image2tensor
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Labels for positive/negative class
lab_pos = 'Buy'
lab_neg = 'No Buy'

# File containing the predictions from a classifier over a
# test set
pred_file = "./70_by_70_predictions_images_2018/pred_BB_Buy_classif.csv"
df_pred = pd.read_csv(pred_file)

# File containing the latent representations of the images
# in the train set
train_lat_file = "./70_by_70_pca/train_pca_BB_Buy_autoencoder.pkl"
with open(train_lat_file, 'rb') as f:
    latent_train = pickle.load(f)

# File containing the latent representations of the images
# in the test set
test_lat_file = "./70_by_70_pca/test_pca_BB_Buy_autoencoder.pkl"
with open(test_lat_file, 'rb') as f:
    latent_test = pickle.load(f)

# Grab one wrong predictions randomly
num_wrong = 3
wrong_idx = np.random.choice(df_pred[df_pred['is_correct?'] == False].index,
                             size = num_wrong,
                             replace = False)

# Select one wrong classification
# and plot its neighbors
num_neighbors = 5
knn = NearestNeighbors(n_neighbors = num_neighbors)
knn.fit(latent_train.iloc[:, 1:].to_numpy())
fig, axs = plt.subplots(num_wrong, num_neighbors, figsize = (10, 10))
for i, idx in enumerate(wrong_idx):
    # Get query image
    path_img_query = df_pred.iloc[idx, 0]
    str_aux_query = df_pred.iloc[idx, 0].split('/')[-1]
    pred_query = lab_pos if df_pred.iloc[idx]['prediction'] == 1 else lab_neg
    true_query = lab_pos if df_pred.iloc[idx]['truth'] == 1 else lab_neg
    
    # TO DO: There must be an easier way of doing this
    for j, f in zip(latent_test['file'].index, latent_test['file']):
        if str_aux_query in f:
            latent_query = latent_test.iloc[j, 1:].to_numpy()
            break
    img_query = image2tensor(path_img_query)
    
    # Get neighbors
    # To be compatible with NearestNeighbors
    # we need to reshape
    idx_neighbors = knn.kneighbors(latent_query.reshape((1, -1)), return_distance = False)
    idx_neighbors = idx_neighbors.reshape((idx_neighbors.shape[1], ))
    
    # Permute (only for plotting purposes)
    img_query = img_query.permute([1, 2, 0])
    
    # Plot row
    for j, idx_n in enumerate(idx_neighbors):
        
        # Find labels for neighbor
        str_aux_neighbor = latent_train.iloc[idx_n]['file']
        str_aux_neighbor = str_aux_neighbor.split('\\')[-1]
        
        # I need the DataFrame with predictions from the training data
        for k, f_str in enumerate(df_pred['file']):
            if str_aux_neighbor in f_str:
                pred_neighbor = lab_pos if df_pred.iloc[k]['prediction'] == 1 else lab_neg
                true_neighbor = lab_pos if df_pred.iloc[k]['truth'] == 1 else lab_neg
                
        
        # Plot query image
        if j == 0:
            axs[i, j].imshow(img_query)
            
        # Plot neighbor image
        else:
            img_neighbor = image2tensor(latent_train['file'][idx_n])
            img_neighbor = img_neighbor.permute([1, 2, 0])
            axs[i, j].imshow(img_neighbor)
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])
plt.subplots_adjust(bottom = 0.1, top = 0.5)
plt.show()

