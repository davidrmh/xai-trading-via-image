import os
import pickle
import numpy as np
import pandas as pd
from utils import image2tensor
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

# File containing the latent representations of the images in the train set
train_lat_file = "./70_by_70_pca/train_pca_RSI_Buy_autoencoder.pkl"

# File containing the latent representations of the images in the test set
test_lat_file = "./70_by_70_pca/test_pca_RSI_Buy_autoencoder.pkl"

# Path of the directory storing the training images for the positive class
path_train_pos = './70_by_70_images_2010_2017/RSI_Buy/'

# Path of the directory storing the training images for the negative class
path_train_neg = './70_by_70_images_2010_2017/no_RSI_Buy/'

# Path of the directory storing the test images for the positive class
path_test_pos = './70_by_70_images_2018/RSI_Buy/'

# Path of the directory storing the test images for the negative class
path_test_neg = './70_by_70_images_2018/no_RSI_Buy/'

# Number of neighbors
num_neighbors = 10

# Get latent representations of the training data
with open(train_lat_file, 'rb') as f:
    latent_train = pickle.load(f)

# Get latent representations of test data
with open(test_lat_file, 'rb') as f:
    latent_test = pickle.load(f)

# Get labels of training data
# TO THINK: PROBABLY I CAN MODIFY train_pca.py FILE
# FOR THE MOMENT I STICK WITH THIS SOLUTION
files_train_pos = [e for e in os.listdir(path_train_pos)]
files_train_neg = [e for e in os.listdir(path_train_neg)]

# Binary classification assumption is used here (1 => positive class)
# (0 => negative class)
train_labels = np.zeros(latent_train.shape[0])

for i in range(train_labels.shape[0]):
    s = latent_train['file'].iloc[i]
    s = s.replace('\\', '/')
    if s.split('/')[-1] in files_train_pos:
        train_labels[i] = 1.0
        
if not sum(train_labels) == len(files_train_pos):
    raise ValueError('Number of positive training labels does not match with number of positive training files')

# Get labels for test data
files_test_pos = [e for e in os.listdir(path_test_pos)]
files_test_neg = [e for e in os.listdir(path_test_neg)]

# Binary classification assumption is used here (1 => positive class)
# (0 => negative class)
test_labels = np.zeros(latent_test.shape[0])

for i in range(test_labels.shape[0]):
    s = latent_test['file'].iloc[i]
    s = s.replace('\\', '/')
    if s.split('/')[-1] in files_test_pos:
        test_labels[i] = 1.0
        
if not sum(test_labels) == len(files_test_pos):
    raise ValueError('Number of positive test labels does not match with number of positive test files')

# Fit K-NN Classifier
knn = KNeighborsClassifier(n_neighbors = num_neighbors)
knn.fit(latent_train.iloc[:, 1:], train_labels)

# Make predictions over test data
pred_test = knn.predict(latent_test.iloc[:, 1:])

# Compute accuracy
acc_test = sum(pred_test == test_labels) / len(pred_test)
print(f'With {num_neighbors} the test accuracy is {acc_test:.4f}')


### STILL WORKING ###
### MAKE PLOTS FOR THE K-NEAREST NEIGHBORS OF SOME QUERY IMAGES IN THE TEST SET
# Number of query predictions from the test set to analyze
num_query = 5

# Type of prediction (True => right prediction, False => Wrong prediction)
bool_type = False

# Number of neighbors to compare with (neighbors come from training data)
num_neighbors = 5

# Figure size
figsize = (20, 20)

# Labels for positive/negative class
lab_pos = 'Buy'
lab_neg = 'No Buy'

# TO DO: path to save the figure
# For the moment I need to also visualize S/D maps

df_test_pred = pd.read_csv(test_pred_file)
df_train_pred = pd.read_csv(train_pred_file)

query_idx = np.random.choice(df_test_pred[df_test_pred['is_correct?'] == bool_type].index,
                             size = num_query,
                             replace = False)

# Fit nearest neighbors object
knn = NearestNeighbors(n_neighbors = num_neighbors)
knn.fit(latent_train.iloc[:, 1:].to_numpy())
fig, axs = plt.subplots(num_query, num_neighbors + 1, figsize = figsize)

for i, idx in enumerate(query_idx):
    # Get query image from test set
    path_img_query = df_test_pred.iloc[idx, 0]
    str_aux_query = df_test_pred.iloc[idx, 0].split('/')[-1]
    pred_query = lab_pos if df_test_pred.iloc[idx]['prediction'] == 1 else lab_neg
    true_query = lab_pos if df_test_pred.iloc[idx]['truth'] == 1 else lab_neg
    
    # TO DO: There must be an easier way of doing this
    for j, f in zip(latent_test['file'].index, latent_test['file']):
        if str_aux_query in f:
            latent_query = latent_test.iloc[j, 1:].to_numpy()
            break
    img_query = image2tensor(path_img_query)
    img_query = img_query.permute([1, 2, 0])
    
    # Plot query image
    axs[i, 0].imshow(img_query)
    axs[i, 0].set_title(f'Pred: {pred_query}\n True: {true_query}')
    axs[i, 0].set_xticks([])
    axs[i, 0].set_yticks([])
    
    # Get neighbors and orders (increasingly) them according to their distance
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

        # Plot neighbor image
        img_neighbor = image2tensor(latent_train['file'][idx_n])
        img_neighbor = img_neighbor.permute([1, 2, 0])
        axs[i, j + 1].imshow(img_neighbor)
        axs[i, j + 1].set_title(f'Pred: {pred_neighbor}\n True: {true_neighbor}')
        axs[i,j + 1].set_xticks([])
        axs[i,j + 1].set_yticks([])
#plt.subplots_adjust
plt.show()