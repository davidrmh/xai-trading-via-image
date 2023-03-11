# How to use

---
## Step 0 - Gather Stock Symbols from Yahoo Finance

* This process has to be done manually. The result must be a _csv_ file with the tcikers (symbols) for the stock series. (See _tickers.csv_ file).

* Yahoo Finance -> [link](finance.yahoo.com)
---
## Step 1 - Download data from Yahoo Finance

* To download stock data from Yahoo Finance use python file `download_yahoo_data.py` together with `config_download.json`.

In Windows using Anaconda powershell:

```
python.exe download_yahoo_data.py -f config_download.json
```

The structure of file `config_download.json` is the following

```json
{
    "tickers":"./tickers.csv",
    "start": "01-01-2018",
    "end": "12-31-2018",
    "min": 1,
    "out_dir": "./test_data_2018"
}
```
* `tickers`: contains a string (using double quotes) with the path of the _csv_ file storing the tickers of stock symbols we want to download the data from (see step 0).

* `start`: String in format `MM-DD-YYYY` with the start date to of the period of data we want to consider.

* `end`: String in format `MM-DD-YYYY` with the end date to of the period of data we want to consider.

* `min`: Positive integer. Minimum number of observations between `start` and `end` date that the stock must have in order to download the data.

* `out_dir`: String with the path where the to store the CSV files.

<b style="color:MediumSeaGreen">Output</b>: CSV files are created in `out_dir`. Each CSV file corresponds to historical data for each ticker in file `tickers`.

<b style="color:Orange">Note</b>: In case that `start` or `end` is a non-trading day, the closest next trading day is used.

---
## Step 2 - Label the Data

* To label the data obtained in Step 1 use file python `labels.py` together with _json_ file `config_labels.json`.

In Windows using Anaconda powershell:

```
python.exe labels.py -f config_labels.json
```

The structure of the file `config_labels.json` is the following
```json
{
    "path_files": "./test_data_2018",
    "bollinger": {"window": 20, "n_std": 2},
    "macd": {"w_slow": 26, "w_fast": 12, "w_sig": 9},
    "rsi": {"window": 14, "uplev": 70, "dowlev": 30},
    "out_dir": "./labels_2018"
}
```

* `path_files`: String with the path of the directory containing the _CSV_ files with historical data for each stock (See Step 1).

* `bollinger`: JSON object with the parameters for Bollinger bands technical indicator.
  * `window`: Positive integer. The size of the lookback window.
  
  * `n_std`: Positive float. Number of standard deviations used to compute the lower and upper band.

* `macd`: JSON object with the parameters for MACD indicator.
  * `w_slow`: Positive integer. Size of the lookback window for the slow moving average.
  * `w_fast`: Positive integer. Size of the lookback window for the fast moving average.
  * `w_sig`: Positive integer. Size of the lookback window for the signal line.

* `rsi`: JSON object  with the parameters for the RSI indicator.
  * `window`: Positive integer. Size of the lookback window.
  * `uplev`: Positive integer. Value of the upper level.
  * `dowlev`: Positive integer. Value of the lower level.
  
* `out_dir`: String with the path of the directory storing the labelled data. For each csv file in `path_files` there is a corresponding csv file in `out_dir` containing the labelled data.

<b style="color:MediumSeaGreen">Output</b>: _csv_ files with labelled data. These files are stored in `out_dir`.

<b style="color:Orange">Note</b>: In the case when there is not enough data to compute a technical indicator the log file `log_labels.txt` is created in the current working directory. This file looks like in the following example

```
Not possible to label file AMCR_2018-01-02_2018-12-28.csv. Not enough useful data
Not possible to label file MRNA_2018-12-07_2018-12-28.csv. Not enough useful data
```
---

## Step 3 - Create Images Using Labelled Data

* To create the images using the labelled data from Step 2 use python file `candlechart.py` together with `config_images.json`.

In Windows using Anaconda powershell:

```
python.exe candlechart.py -f config_images.json
```

The structure of `config_image.json` is the following
```json
{
    "path_labels": "./labels_2018",
    "out_dir": "./70_by_70_images_2018",
    "sample_size": 0,
    "bb_w":20,
    "macd_w": 26,
    "rsi_w": 14,
    "triggers": ["BB_Buy", "BB_Sell", "MACD_Buy", "MACD_Sell", "RSI_Buy", "RSI_Sell"],
    "dict_chart":{
        "color_up":"green",
        "color_down": "red",
        "width_bar": 0.6,
        "width_shadow": 0.1,
        "shape0": 70,
        "shape1": 70,
        "bg_color": "black",
        "dpi": 600
    }
}
```

* `path_labels`: String. Path of the directory storing the csv files with the labelled data (see Step 2).

* `out_dir`: String. Path of the directory storing the images.

* `sample_size`: Positive integer. Number of images in the positive and negative class for each signal (trigger) and each file in `path_labels`. If set to 0, then for each trigger all the images corresponding to the positive class (relative to the trigger) as well as same number of images in the negative class. Using this parameter we can ensure we always have balanced datasets.

* `bb_w`: Positive integer. Size of the lookback window for triggers related to Bollinger Bands. Ideally it should be the same value as the one used for `window` in the file `config_labels.json` in Step 2 for Bollinger bands.

* `macd_w`: Positive integer. Size of the lookback window for triggers related to MACD indicator. Ideally it should be the largest of `w_slow`, `w_fast`, `w_sig` used in the file `config_labels.json` in Step 2.

* `rsi_w`: Positive integer. Size of the lookback window for triggers related to RSI indicator. Ideally it should be the same value as the one used for `window` in the file `config_labels.json` in Step 2 for RSI.

* `triggers`: List with strings. Each string corresponds to one possible trigger signal.

* `dict_chart`: JSON object. Contains attributes related to the candlechart.

  * `color_up`: Matplotlib compatible string for colors. Color used for up candles.
  * `color_down`: Matplotlib compatible string for colors. Color used for down candles.
  
  * `width_bar`: Positve float. Width in inches of the candle bar.
  * `width_shadow`: Positive float. Width in inches of the shadow of the candle bar.
  * `shape0`: Positive integer. Height of the array representing the output image.
  * `shape1`: Positive integer. Width of the array representing the output image.
  * `bg_color`: Matplotlib compatible string for colors. Background color.
  * `dpi`: Positive integer. Dots per inch.

<b style="color:MediumSeaGreen">Output</b>: For each trigger in `triggers` two subdirectories in `out_dir` are created. One of them contains images the images for the positive class of the trigger. Similarly, the other subdirectory contains the images for the negative class of the triggers.