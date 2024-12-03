import numpy as np
import pandas as pd
import yfinance as yf

def fetch_stock_data(tickers, start_date=None, end_date=None, interval='1d'):
    stock_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)
        stock_data[ticker] = data
    return stock_data

def compute_metrics(data):
    metrics = {}
    for ticker, df in data.items():
        df['Daily Return'] = df['Close'].pct_change().fillna(0)
        df['Cumulative Return'] = (1 + df['Daily Return']).cumprod().fillna(1)
        df['SMA_3'] = df['Close'].rolling(window=3).mean()
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['SMA_90'] = df['Close'].rolling(window=90).mean()
        if 'Volume' in df.columns:
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            df['VWAP_3'] = (df['Close'] * df['Volume']).rolling(window=3).sum() / df['Volume'].rolling(window=3).sum()
            df['VWAP_7'] = (df['Close'] * df['Volume']).rolling(window=7).sum() / df['Volume'].rolling(window=7).sum()
            df['VWAP_30'] = (df['Close'] * df['Volume']).rolling(window=30).sum() / df['Volume'].rolling(window=30).sum()
        metrics[ticker] = df
    return metrics

def download(asset, start_date, end_date):
    print("Scarico i dati storici dell'azione...")
    try:
        data = yf.download(asset, start=start_date, end=end_date)
        if data.empty:
            print("Non sono stati trovati dati per le date e l'asset specificati.")
        else:
            print(f"Dati scaricati per {asset}")
            data.to_csv(f"./csv/{asset}_data.csv")
            print(f"Dati salvati in {asset}_data.csv")
    except Exception as e:
        print(f"Errore durante il download dei dati: {str(e)}")
    return data

def cleaning(data: pd.DataFrame):
    print("Pulizia dei dati...")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]
    data.columns = [col.split(' ')[0] for col in data.columns]
    data.reset_index(inplace=True)
    print(f"Dati puliti: {data.shape}")
    return data

def get_close_data(data: pd.DataFrame):
    close = data['Close'].values.tolist()
    close = [round(price, 4) for price in close]
    return close

def state_formatter(state):
    if isinstance(state, np.ndarray):
        return state.flatten()
    elif isinstance(state, list):
        return np.array(state).flatten()
    else:
        print(f"Stato ricevuto in state_formatter: {state}, tipo: {type(state)}")
        raise ValueError("Formato dello stato non riconosciuto")
