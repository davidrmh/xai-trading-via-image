'''
This file contains all the functions used to create the candle charts images for training the models.
It is assumed that the data uses Yahoo Finance CSV structure (as of 22nd January 2023) and is stored
in pandas dataframe with column `Date` used as index.
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
    `fig_width`: Float. Width of figure in inches
    `fig_height`: Float. Height of figure in inches
    `bg_color`: String. Background color
    :return: None
    '''
    up = data[data['Close'] > data['Open']]
    down = data[data['Close'] <= data['Open']]
    color_up = dict_chart['color_up'] if 'color_up' in dict_chart else 'green'
    color_down = dict_chart['color_down'] if 'color_down' in dict_chart else 'red'
    width_bar = dict_chart['width_bar'] if 'width_bar' in dict_chart else 0.6
    width_shadow = dict_chart['width_shadow'] if 'width_shadow' in dict_chart else 0.06
    fig_width = dict_chart['fig_width'] if 'fig_width' in dict_chart else 0.72916  # Equivalent to 70 pixels 
    fig_height = dict_chart['fig_height'] if 'fig_width' in dict_chart else 0.72916
    bg_color = dict_chart['bg_color'] if 'bg_color' in dict_chart else 'black'
    plt.ioff()
    plt.figure(figsize = (fig_width, fig_height))
    ax = plt.axes()
    ax.set_facecolor(bg_color)
    
    #Increase price bars
    ax.bar(up.index, up['Close']-up['Open'], width_bar, bottom=up['Open'], color=color_up)
    ax.bar(up.index, up['High']-up['Close'], width_shadow, bottom=up['Close'], color=color_up)
    ax.bar(up.index, up['Low']-up['Open'], width_shadow, bottom=up['Open'], color=color_up)
    
    #Decrease price bars
    ax.bar(down.index, down['Close']-down['Open'], width_bar, bottom=down['Open'], color=color_down)
    ax.bar(down.index, down['High']-down['Open'], width_shadow, bottom=down['Open'], color=color_down)
    ax.bar(down.index, down['Low']-down['Close'], width_shadow, bottom=down['Close'], color=color_down)
    
    #We only care of the visual pattern and not the numeric values
    ax.set_xticks([])
    ax.set_yticks([])
    
    #Saves the figure
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    plt.savefig(os.path.join(filepath, f'{filename}.png'), dpi=300, backend = 'Agg')
    plt.close()
    plt.ion()
    
def image2tensor(image_path):
    image = read_image(image_path, ImageReadMode.RGB)
    #Change dtype to float32
    image = image.to(torch.float32)
    return image

def main(config: dict, triggers: list) -> None:
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
        elif trig.startswith('macd'):
            w = config['macd_w']
        elif trig.startswith('rsi'):
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
    triggers = ['BB_Buy', 'BB_Sell',
                'MACD_Buy', 'MACD_Sell',
               'RIS_Buy', 'RSI_Sell']
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config, triggers)
    print(f' ===== Images created :) ===== \n')
    
    


