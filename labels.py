import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
import json
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                   help = 'Path of the json file containing the configuration of each indicator',
                   type = str)
args = parser.parse_args()

def bollinger_buy_(data: pd.DataFrame, window: int) -> None:
    nobs = data.shape[0]
    signal = [0.0] * nobs
    for t in range(window, nobs):
        if (data['Adj Close'].iloc[t - 1] < data['Low_Band'].iloc[t - 1]) and (data['Adj Close'].iloc[t] > data['Low_Band'].iloc[t]):
            signal[t] = 1.0
    data['BB_Buy'] = signal

def bollinger_sell_(data: pd.DataFrame, window: int) -> None:
    nobs = data.shape[0]
    signal = [0.0] * nobs
    for t in range(window, nobs):
        if (data['Adj Close'].iloc[t - 1] > data['Up_Band'].iloc[t - 1]) and (data['Adj Close'].iloc[t] < data['Up_Band'].iloc[t]):
            signal[t] = 1.0
    data['BB_Sell'] = signal
    
def bollinger_signals_(data: pd.DataFrame, window: int = 20, n_std: int = 2) -> None:
    # Compute Bollinger bands
    bollinger = BollingerBands(data['Adj Close'], window, n_std)
    low_band = bollinger.bollinger_lband()
    up_band = bollinger.bollinger_hband()
    data['Low_Band'] = low_band
    data['Up_Band'] = up_band
    bollinger_buy_(data, window)
    bollinger_sell_(data, window)

def macd_buy_(data: pd.DataFrame, w_slow, w_sig) -> None:
    nobs = data.shape[0]
    signal = [0.0] * nobs
    for t in range(w_sig + w_slow - 1, nobs):
        if (data['MACD'].iloc[t] > data['MACD_Sig'].iloc[t]) and (data['MACD'].iloc[t - 1] < data['MACD_Sig'].iloc[t - 1]):
            signal[t] = 1.0
    data['MACD_Buy'] = signal

def macd_sell_(data: pd.DataFrame, w_slow, w_sig) -> None:
    nobs = data.shape[0]
    signal = [0.0] * nobs
    for t in range(w_sig + w_slow - 1, nobs):
        if (data['MACD'].iloc[t] < data['MACD_Sig'].iloc[t]) and (data['MACD'].iloc[t - 1] > data['MACD_Sig'].iloc[t - 1]):
            signal[t] = 1.0
    data['MACD_Sell'] = signal

def macd_signals_(data: pd.DataFrame, w_slow: int = 26, w_fast: int = 12, w_sig: int = 9) -> None:
    macd = MACD(data['Adj Close'], window_slow = w_slow, window_fast = w_fast, window_sign = w_sig)
    data['MACD'] = macd.macd()
    data['MACD_Sig'] = macd.macd_signal()
    macd_buy_(data, w_slow, w_sig)
    macd_sell_(data, w_slow, w_sig)

def rsi_buy_(data: pd.DataFrame, window: int, dowlev: int) -> None:
    nobs = data.shape[0]
    signal = [0.0] * nobs
    for t in range(window, nobs):
        if (data['RSI'].iloc[t] > dowlev) and (data['RSI'].iloc[t - 1] < dowlev):
            signal[t] = 1.0
    data['RSI_Buy'] = signal

def rsi_sell_(data: pd.DataFrame, window: int, uplev: int) -> None:
    nobs = data.shape[0]
    signal = [0.0] * nobs
    for t in range(window, nobs):
        if (data['RSI'].iloc[t] < uplev) and (data['RSI'].iloc[t - 1] > uplev):
            signal[t] = 1.0
    data['RSI_Sell'] = signal

def rsi_signals_(data: pd.DataFrame, window: int = 14, uplev: int = 70, dowlev: int = 30) -> None:
    rsi = RSIIndicator(data['Adj Close'], window = window)
    data['RSI'] = rsi.rsi()
    rsi_buy_(data, window, dowlev)
    rsi_sell_(data, window, uplev)

def main(config: dict) -> None:
    path_files = config['path_files']
    files = os.listdir(path_files)
    n_files = len(files)
    out_dir = config['out_dir']
    
    # Parameters for Bollinger bands
    bol_w = config['bollinger']['window']
    bol_std = config['bollinger']['n_std']
    
    # Parameters for MACD
    w_slow = config['macd']['w_slow']
    w_fast = config['macd']['w_fast']
    w_sig = config['macd']['w_sig']
    
    # Parameters for RSI
    rsi_w = config['rsi']['window']
    uplev = config['rsi']['uplev']
    dowlev = config['rsi']['dowlev']
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    if os.path.exists("log_labels.txt"):
        os.remove("log_labels.txt")
    
    log_file = open('log_labels.txt', 'w')
    bool_log = False
    print(" ===== Labelling files ===== \n")
    for f in files:
        # Read file
        data = pd.read_csv(os.path.join(path_files, f))
        
        # Drop rows where Volume is zero
        data.drop(data[data['Volume'] == 0].index, inplace = True)
        data.reset_index(drop = True, inplace = True)
        
        # Drop observations where the price did not move
        data.drop(data[data['Low'] == data['High']].index, inplace = True)
        data.reset_index(drop = True, inplace = True)
        
        # Only label if there's enough data
        if data.shape[0] < max(bol_w, w_slow, rsi_w):
            bool_log = True
            log_file.write(f'Not possible to label file {f}. Not enough useful data \n')
            continue
        
        # Label according to technical indicators
        bollinger_signals_(data, bol_w, bol_std)
        macd_signals_(data, w_slow, w_fast, w_sig)
        rsi_signals_(data, rsi_w, uplev, dowlev)
        
        # Save labelled data
        data.to_csv(os.path.join(out_dir, f), index = False)
        
    log_file.close()
    if bool_log:
        print('WARNING: Not all files were labelled. Check log_labels.txt file')

if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    main(config)
    print(' === Files Labelled === \n')