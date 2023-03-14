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

<b style="color:Orange">Note</b>: This program might take a while in finishing, this depends on the number of triggers, the sample size and the number of labelled files.

---

## Step 4 - Train classifiers

To train the classifiers using the images created in step 3 use the python file `train_classifier.py` together with the configuration file `config_train.json`.

In Windows using Anaconda powershell:

```
python.exe train_classifier.py -f config_train.json
```

The structure of file `config_train.json` is the following

```json
{
    "train_dir_pos": ["./70_by_70_images_2010_2017/BB_Buy",
    "./70_by_70_images_2010_2017/MACD_Buy"],

    "train_dir_neg": ["./70_by_70_images_2010_2017/no_BB_Buy",
    "./70_by_70_images_2010_2017/no_MACD_Buy"],

    "test_dir_pos": ["./70_by_70_images_2018/BB_Buy",
    "./70_by_70_images_2018/MACD_Buy"],

    "test_dir_neg": ["./70_by_70_images_2018/no_BB_Buy",
    "./70_by_70_images_2018/no_MACD_Buy"],

    "out_path": "./70_by_70_trained_classifiers",

    "out_file": ["BB_Buy_classif", "MACD_Buy_classif"],

    "adam_par": {"lr": 0.001, "betas": [0.9, 0.999], "eps":1e-08},
    
    "batch_size": 16,

    "epochs": 50,

    "accept_lev": 0.5,

    "early": 5
}
```
* `train_dir_pos`: List of strings. Directory where **training** images of the positive class are stored. One directory per model to train.

* `train_dir_neg`: List of strings. Directory where **training** images of the negative class are stored. One directory per model to train.

* `test_dir_pos`: List of strings. Directory where **test** images of the positive class are stored. One directory per model to train.

* `test_dir_neg`: List of strings. Directory where **test** images of the negative class are stored. One directory per model to train.

* `out_path`: String. Path of the directory where the trained models are stored.

* `out_file`: List of strings. Name(s) of the file(s) storing the trained model(s).

* `adam_par`: JSON object. Parameters for Adam optimizer (see pytorch documentation).

* `batch_size`: Positive integer. Batch size.

* `epochs`: Positive integer. Number of epochs.

* `early`: Positive integer.

<b style="color:MediumSeaGreen">Output</b>: one .pth file per element in `out_file` is created in `out_path`. This file contains the trained classifier.

---

## Step 5 - Make predictions

After training the classifiers (see step 4), make predictions over test or validation images. For validations images it is assumed that we still have labels, for test images no labels are provided.

To make predictions use python file `pred_classifier.py` together with the configuration file `config_pred.json`.

In Windows using Anaconda powershell:

```
python.exe pred_classifier.py -f config_pred.json
```

The structure of the file `config_pred.json` is as follows

```json
{
    "path_model":["./70_by_70_trained_classifiers/BB_Buy_classif.pth",
    "./70_by_70_trained_classifiers/MACD_Buy_classif.pth",
    "./70_by_70_trained_classifiers/RSI_Buy_classif.pth",
    "./70_by_70_trained_classifiers/BB_Sell_classif.pth",
    "./70_by_70_trained_classifiers/MACD_Sell_classif.pth",
    "./70_by_70_trained_classifiers/RSI_Sell_classif.pth"],

    "path_img": [["./70_by_70_images_2010_2017/BB_Buy/", "./70_by_70_images_2010_2017/no_BB_Buy/"],
    ["./70_by_70_images_2010_2017/MACD_Buy/", "./70_by_70_images_2010_2017/no_MACD_Buy/"],
    ["./70_by_70_images_2010_2017/RSI_Buy/", "./70_by_70_images_2010_2017/no_RSI_Buy/"],
    ["./70_by_70_images_2010_2017/BB_Sell/", "./70_by_70_images_2010_2017/no_BB_Sell/"],
    ["./70_by_70_images_2010_2017/MACD_Sell/", "./70_by_70_images_2010_2017/no_MACD_Sell/"],
    ["./70_by_70_images_2010_2017/RSI_Sell/", "./70_by_70_images_2010_2017/no_RSI_Sell/"]],

    "mode": "val",

    "batch_size": 32,

    "accept_lev": 0.5,

    "out_dir":"./70_by_70_predictions_images_2010_2017"

}
```

* `path_model`: List of strings. Path of the pth file containing the model.

* `path_img`: Nested list. The $i$-th inner list contains the directory (or directories) storing the images to predict using the $i$-th model in `path_model`.

* `mode`: String. one of `val` or `pred`.

  * if `mode: "val"`, then validation mode is used. In this mode, the length of each inner list in `path_img` must be equal to two. The first element of each inner list must contain the path of the directory storing images from the positive class. Similarly, the second element of each inner list must contain the path of the directory storing images from the negative class.
  
  * if `mode: "pred"`, then prediction mode is used. In this mode, the length of each inner list is at least one. Each inner list contains path to directories storing images to classify.
  
  * `batch_size`: Positive integer. Batch size.
  
  * `accept_lev`: Float in [0, 1]. Minimum level of the predicted probability for an image to be considered a member of the possitive class.
  
  * `out_dir`: String. Path of the directory where the predictions are stored.

<b style="color:MediumSeaGreen">Output</b>: For each model in `path_model` a csv file is created. The csv file contains the path of each classified image (column file), its predicted label (column precition) and, if `mode: "val"`, the true label (column truth) and a column specifying if the prediction was correct (column is_correct?).