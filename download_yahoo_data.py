from datetime import datetime
import pandas as pd
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', 
                    help = "Path of the JSON file with all the parameters needed",
                   type = str)

args = parser.parse_args()

def main(config: dict) -> None:
    str_start_date = config['start']
    str_end_date = config['end']
    minobs = config['min']
    outdir = config['out_dir']
    sym_path = config['tickers']
    symbols = pd.read_csv(sym_path)
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    t_start_date = datetime.strptime(str_start_date, '%m-%d-%Y')
    t_end_date = datetime.strptime(str_end_date, '%m-%d-%Y')
    # Yahoo Finance API uses Unix timestamp seconds since January 1, 1970
    epoch_date = datetime.strptime('01-01-1970', '%m-%d-%Y')
    period1 = int( (t_start_date - epoch_date).total_seconds() )
    period2 = int( (t_end_date - epoch_date).total_seconds() )
    fail = []
    # TO DO: PARALLELIZE
    print(f' ===== Downloading data ===== \n')
    for i in range(symbols.shape[0]):
        s = symbols.iloc[i][0]
        s = s.replace('.', '-')
        query = f'https://query1.finance.yahoo.com/v7/finance/download/{s}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true'
        try:
            data = pd.read_csv(query)
            if data.shape[0] >= minobs:
                file_path = os.path.join(outdir, f'{s}_{data.iloc[0,0]}_{data.iloc[-1,0]}.csv')
                data.to_csv(file_path, index = False)
            else:
                print(f' ===== For stock {s} there are less than {minobs} observations. Skipping ===== \n')
        except:
            print(f'Error for {s}')
            fail.append(s)
            continue
    if len(fail) > 0:
        df_fails = pd.DataFrame({"Symbol": fail})
        df_fails.to_csv("failed_download.csv", index = False)
    
if __name__ == '__main__':
    with open(args.file, 'r') as f:
        config = json.load(f)
    
    main(config)
    print(f' ===== Data Downloaded ===== \n')