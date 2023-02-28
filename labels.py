import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from typing import Union, Optional

def bollinger_buy(data: pd.DataFrame, window: int) -> pd.DataFrame:
    nobs = data.shape[0]
    signal = [0.0] * nobs
    for t in range(window - 1, nobs):
        if (data['Adj Close'].iloc[t - 1] < data['Low_Band'].iloc[t - 1]) and (data['Adj Close'].iloc[t] > data['Low_Band'].iloc[t]):
            signal[t] = 1.0
    data['Buy_Signal'] = signal
    return data

def bollinger_sell(data: pd.DataFrame, window: int) -> pd.DataFrame:
    nobs = data.shape[0]
    signal = [0.0] * nobs
    for t in range(window - 1, nobs):
        if (data['Adj Close'].iloc[t - 1] > data['Up_Band'].iloc[t - 1]) and (data['Adj Close'].iloc[t] < data['Up_Band'].iloc[t]):
            signal[t] = 1.0
    data['Sell_Signal'] = signal
    return data
    
def bollinger_signals(data: pd.DataFrame, window: int = 20, window_dev: Optional[int] = None) -> pd.DataFrame:
    window_dev = window if window_dev == None else window_dev
    
    # Compute Bollinger bands
    bollinger = BollingerBands(data['Adj Close'], window, window_dev)
    low_band = bollinger.bollinger_lband()
    up_band = bollinger.bollinger_hband()
    data['Low_Band'] = low_band
    data['Up_Band'] = up_band
    data = bollinger_buy(data, window)
    data = bollinger_sell(data, window)
    return data

def macd_buy(data: pd.DataFrame, w_slow, w_sig) -> pd.DataFrame:
    nobs = data.shape[0]
    signal = [0.0] * nobs
    for t in range(w_sig + w_slow - 1, nobs):
        if (data['MACD'].iloc[t] > data['MACD_Sig'].iloc[t]) and (data['MACD'].iloc[t - 1] < data['MACD_Sig'].iloc[t - 1]):
            signal[t] = 1.0
    data['MACD_Buy'] = signal
    return data

def macd_sell(data: pd.DataFrame, w_slow, w_sig) -> pd.DataFrame:
    nobs = data.shape[0]
    signal = [0.0] * nobs
    for t in range(w_sig + w_slow - 1, nobs):
        if (data['MACD'].iloc[t] < data['MACD_Sig'].iloc[t]) and (data['MACD'].iloc[t - 1] > data['MACD_Sig'].iloc[t - 1]):
            signal[t] = 1.0
    data['MACD_Sell'] = signal
    return data

def macd_signals(data: pd.DataFrame, w_slow: int = 26, w_fast: int = 12, w_sig: int = 9) -> pd.DataFrame:
    macd = MACD(data['Adj Close'], window_slow = w_slow, window_fast = w_fast, window_sign = w_sig)
    data['MACD'] = macd.macd()
    data['MACD_Sig'] = macd.macd_signal()
    data = macd_buy(data, w_slow, w_sig)
    data = macd_sell(data, w_slow, w_sig)
    return data
    