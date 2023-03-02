'''
This file contains all the functions used to create the candle charts images for training the models.
It is assumed that the data uses Yahoo Finance CSV structure (as of 22nd January 2023) and is stored
in pandas dataframe with column `Date` used as index.

Since matplotlib uses inches to specify figure size. If we want the output image
to be an array with shape (H, W, C) and we are saving using a DPI value of D,
then the width of the figure is equal to  W / D. Similarly, the height of the
figure must be equal to H / D.
'''
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image, ImageReadMode
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the JSON file with the configuration for making the plots',
                   type = str)
args = parser.parse_args()


def save_candle_chart(data: pd.DataFrame,
                      filename: str,
                      filepath: str,
                      dict_chart: dict) -> None:
    '''
    Create a candle chart.
    Notice that Close prices must be used instead of Adj Close otherwise the plot is not displayed
    correctly since the lower shadow might get drawn upwards instead of downwards.
    Adj Close prices are used as reference price for the next day and are not really observed in practice.
    https://www.geeksforgeeks.org/how-to-create-a-candlestick-chart-in-matplotlib/
    
    :param data: Pandas dataframe.
    :param filename: String with the name of the output file
    :param filename: String with the path for storing the output file
    :param dict_chart: Dictionary with the following keys
    `color_up`: String. Color of the bars when the price increases
    `color_down`: String. Color of the bars when the price decreases
    `width_bar`: Float. Width of the bars
    `width_shadow`: Float. Width of the shadow
    `shape0`: Positive integer. Length of axis 0 of the array representing the figure (height)
    `shape1`: Positive integer. Length of axis10 of the array representing the figure (width)
    `bg_color`: String. Background color
    `dpi`: Float. DPI resolution
    :return: None
    '''
    up = data[data['Close'] > data['Open']]
    down = data[data['Close'] <= data['Open']]
    color_up = dict_chart['color_up'] if 'color_up' in dict_chart else 'green'
    color_down = dict_chart['color_down'] if 'color_down' in dict_chart else 'red'
    width_bar = dict_chart['width_bar'] if 'width_bar' in dict_chart else 0.6
    width_shadow = dict_chart['width_shadow'] if 'width_shadow' in dict_chart else 0.06
    shape0 = dict_chart['shape0'] if 'shape0' in dict_chart else 224
    shape1 = dict_chart['shape1'] if 'shape1' in dict_chart else shape0
    bg_color = dict_chart['bg_color'] if 'bg_color' in dict_chart else 'black'
    dpi = dict_chart['dpi'] if 'dpi' in dict_chart else 600
    
    fig_height = shape0 / dpi
    fig_width = shape1 / dpi
    plt.ioff()
    plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    ax.set_facecolor(bg_color)
    ax.axis('off')
    
    #Increase price bars
    ax.bar(up.index, up['Close']-up['Open'], width_bar, bottom = up['Open'], color = color_up)
    ax.bar(up.index, up['High']-up['Close'], width_shadow, bottom = up['Close'], color = color_up)
    ax.bar(up.index, up['Low']-up['Open'], width_shadow, bottom = up['Open'], color = color_up)
    
    #Decrease price bars
    ax.bar(down.index, down['Close']-down['Open'], width_bar, bottom = down['Open'], color = color_down)
    ax.bar(down.index, down['High']-down['Open'], width_shadow, bottom = down['Open'], color = color_down)
    ax.bar(down.index, down['Low']-down['Close'], width_shadow, bottom = down['Close'], color = color_down)
    
    #Saves the figure
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    plt.savefig(os.path.join(filepath, f'{filename}.png'), 
                dpi = dpi, 
                backend = 'Agg',
               facecolor = bg_color)
    plt.close()
    plt.ion()
    
def image2tensor(image_path, norm = True):
    image = read_image(image_path, ImageReadMode.RGB)
    #Change dtype to float32
    image = image.to(torch.float32)
    
    # Normalize in [0, 1]
    if norm:
        image = image / image.max()
    
    # Permute is used for plotting using matplotlib imshow
    # a tensor with shape (channels, heigth, width)
    # is reshaped to (height, width, channels)
    # curiosly reshape method does not work
    image = image.permute([1, 2, 0])
    return image

def main(config: dict) -> None:
    triggers = config['triggers']
    path_lab = config['path_labels']
    out_dir = config['out_dir']
    samp_size = config['sample_size']
    dict_chart = config['dict_chart']
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    files = os.listdir(path_lab)
    
    for trig in triggers:
        if trig.startswith('BB'):
            w = config['bb_w']
        elif trig.startswith('MACD'):
            w = config['macd_w']
        elif trig.startswith('RSI'):
            w = config['rsi_w']
        print(f' ===== Creating files for {trig} ===== \n')
        for f in files:
            data = pd.read_csv(os.path.join(path_lab, f))
            
            # Indices where a decision triggers
            idx = data[data[trig] == 1.0].index
            
            if len(idx) < samp_size:
                print(f' ===== Not enough {trig} signals for {f} and sample size {samp_size} ===== \n')
                continue
                       
            # Indices where a decision is not triggered
            idx_no = data.index.difference(idx)
            
            # To always have an image for the negative class
            # we force to pick indices such that there is enough
            # data to create an image using the specified
            # window size.
            # This might create problems in the remote case
            # when there are less indices than samp_size
            # but this is virtually impossible (I hope so)
            idx_no = idx_no[idx_no > w]
            
            if len(idx_no) < samp_size:
                print(f' ===== Not enough Negative {trig} signals for {f} and sample size {samp_size} ===== \n')
                continue             
            
            # Sample samp_size indices. No replacement
            idx_samp = np.random.choice(idx, size = samp_size, replace = False)
            idx_samp_no = np.random.choice(idx_no, size = samp_size, replace = False)
            
            # Create images for trigger moments (positive class)
            file_path = os.path.join(out_dir, trig)
            for i in idx_samp:
                data_img = data[['Open', 'High', 'Low', 'Close']].iloc[i - w + 1:i + 1]
                file_name = f"{trig}_{f.split('_')[0]}_{i}" # trig + _ + symbol + _ + i
                save_candle_chart(data_img, file_name, file_path, dict_chart)
            
            # Create images for non-trigger moments (negative class)
            file_path = os.path.join(out_dir, f'no_{trig}')
            for i in idx_samp_no:
                data_img = data[['Open', 'High', 'Low', 'Close']].iloc[i - w + 1:i + 1]
                file_name = f"no_{trig}_{f.split('_')[0]}_{i}"
                save_candle_chart(data_img, file_name, file_path, dict_chart)

if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config)
    print(f' ===== Images created :) ===== \n')
    
    


