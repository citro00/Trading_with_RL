#  This file contains utility functions for the project.
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_stock_data(tickers, start_date=None, end_date=None, interval='1d'):
    """
    Fetches historical stock data for given tickers using yfinance.
    
    Parameters:
    - tickers (list of str): List of stock ticker symbols.
    - start_date (str): Start date for the data in format 'YYYY-MM-DD' (optional).
    - end_date (str): End date for the data in format 'YYYY-MM-DD' (optional).
    - interval (str): Data interval. Valid intervals: '1d', '1wk', '1mo', etc.

    Returns:
    - dict: A dictionary with tickers as keys and DataFrames as values.
    """
    stock_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)
        stock_data[ticker] = data
    
    return stock_data

# Compute additional metrics
def compute_metrics(data):
    """
    Computes additional metrics for stock data.
    
    Parameters:
    - data (dict): A dictionary with tickers as keys and DataFrames as values.
    
    Returns:
    - dict: A dictionary with tickers as keys and DataFrames as values.
    """
    metrics = {}
    for ticker, df in data.items():
        # Calculate daily return
        df['Daily Return'] = df['Close'].pct_change().fillna(0)
        
        # Calculate cumulative return
        df['Cumulative Return'] = (1 + df['Daily Return']).cumprod().fillna(1)
        
        # Calculate simple moving averages for specified periods
        df['SMA_3'] = df['Close'].rolling(window=3).mean()
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()  # Approximating 1 month
        df['SMA_90'] = df['Close'].rolling(window=90).mean()  # Approximating 3 months
        
        # Calculate VWAP (Volume Weighted Average Price)
        if 'Volume' in df.columns:
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            df['VWAP_3'] = (df['Close'] * df['Volume']).rolling(window=3).sum() / df['Volume'].rolling(window=3).sum()
            df['VWAP_7'] = (df['Close'] * df['Volume']).rolling(window=7).sum() / df['Volume'].rolling(window=7).sum()
            df['VWAP_30'] = (df['Close'] * df['Volume']).rolling(window=30).sum() / df['Volume'].rolling(window=30).sum()
            df['VWAP_90'] = (df['Close'] * df['Volume']).rolling(window=90).sum() / df['Volume'].rolling(window=90).sum()
        
        metrics[ticker] = df
    
    return metrics

def download(asset, start_date, end_date):
    try:
        data = yf.download(asset, start=start_date, end=end_date)
        if data.empty:
            print("Non sono stati trovati dati")
        else:
            print(f"Dati scaricati per {asset}")
            data.to_csv(f"./csv/{asset}_data.csv")
            print(f"Dati salavati in {asset}_data.csv")
    except Exception as e:
        print(f"Errorre durante il download dei dati: {e}")
    return data

def cleaning(data:pd.DataFrame):
    #Trasforma il DataFrame da MultiLevelInde in plain DataFrame
    data = data.stack(level=1).rename_axis().reset_index(level=1)
    #Rimuovi nome colonne
    data.columns.name = None
    #Droppa colonna con nome ticker (inutile)
    data = data.drop(columns='Ticker')

    #print("Colonne:{}".format(data.columns))
    #print("Index:\n{}".format(data.index))
    print("My data head:\n{}".format(data.head()))
    return data

def get_close_data(data:pd.DataFrame):
    close = data.Close.values.tolist()
    for i in range(0,len(close)):
        close[i] = round(close[i], 4)
    return close

def state_formatter(state):
    if len(state) == 2:
        new_state = [[float(arr[1]) for arr in state[0]]]
    else:
        new_state = [[float(arr[1]) for arr in state]]
    return new_state