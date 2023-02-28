from datetime import datetime
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tickers', 
                    help = "Path of the csv files containing stock tickers to download data from",
                   type = str)

parser.add_argument('-s', '--start', 
                    help = 'String in format MM-DD-YYYY. Start date', 
                    type = str)

parser.add_argument('-e', '--end', 
                    help = 'String in format MM-DD-YYYY. End date', 
                    type = str)

parser.add_argument('-o', '--out', 
                    help = 'String with the path of the directory to store the files', 
                    type = str)
args = parser.parse_args()

def main(symbols: pd.DataFrame, str_start_date: str, str_end_date: str, outdir: str):
    
    t_start_date = datetime.strptime(str_start_date, '%m-%d-%Y')
    t_end_date = datetime.strptime(str_end_date, '%m-%d-%Y')
    # Yahoo Finance API uses Unix timestamp seconds since January 1, 1970
    epoch_date = datetime.strptime('01-01-1970', '%m-%d-%Y')
    period1 = int( (t_start_date - epoch_date).total_seconds() )
    period2 = int( (t_end_date - epoch_date).total_seconds() )
    fail = []
    # TO DO: PARALLELIZE
    for i in range(symbols.shape[0]):
        s = symbols.iloc[i][0]
        s = s.replace('.', '-')
        query = f'https://query1.finance.yahoo.com/v7/finance/download/{s}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true'
        try:
            data = pd.read_csv(query)
            file_path = os.path.join(outdir,f'{s}_{data.iloc[0,0]}_{data.iloc[-1,0]}.csv')
            data.to_csv(file_path, index = False)
        except:
            print(f'Error for {s}')
            fail.append(s)
            continue
    if len(fail) > 0:
        df_fails = pd.DataFrame({"Symbol": fail})
        df_fails.to_csv("failed_download.csv", index = False)
    
if __name__ == '__main__':
    sym_path = args.tickers
    symbols = pd.read_csv(sym_path)
    str_start_date = args.start if args.start else '01-01-2010'
    str_end_date = args.end if args.end else '12-31-2018'
    outdir = args.out if args.out else './'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    main(symbols, str_start_date, str_end_date, outdir)
    print(f' ===== Data Downloaded ===== \n')