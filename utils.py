import numpy as np
import pandas as pd
import yfinance as yf

"""
Questo modulo fornisce funzioni per scaricare, pulire e analizzare i dati finanziari di azioni. 
Utilizza l'API Yahoo Finance (via `yfinance`) per ottenere i dati storici di prezzo e volume. 
Include anche funzioni per calcolare metriche come rendimenti cumulativi, medie mobili semplici (SMA) 
e il prezzo medio ponderato per il volume (VWAP). Le funzioni sono pensate per l'integrazione con 
ambienti di apprendimento basati su trading.
"""

def fetch_stock_data(tickers, start_date=None, end_date=None, interval='1d'):
    
    """
    Scarica i dati storici di prezzo e volume per una lista di azioni.
    :param tickers: Lista di ticker delle azioni.
    :param start_date: Data di inizio per i dati storici (opzionale).
    :param end_date: Data di fine per i dati storici (opzionale).
    :param interval: Intervallo temporale dei dati ('1d', '1h', ecc.).
    :return: Dizionario con i dati storici per ciascun ticker.
    """
    
    stock_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)
        stock_data[ticker] = data
    return stock_data

def compute_metrics(data):
    
    """
    Calcola metriche utili per l'analisi dei dati finanziari, inclusi:
    - Rendimento giornaliero
    - Rendimento cumulativo
    - Medie mobili (SMA) su diverse finestre temporali
    - Prezzo medio ponderato per il volume (VWAP)
    :param data: Dizionario di DataFrame contenente i dati storici.
    :return: Dizionario con i DataFrame arricchiti con metriche aggiuntive.
    """
    
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
    
    """
    Scarica i dati storici di un'azione specifica utilizzando `yfinance` e li salva in un file CSV.
    :param asset: Ticker dell'azione da scaricare.
    :param start_date: Data di inizio per i dati storici.
    :param end_date: Data di fine per i dati storici.
    :return: DataFrame con i dati storici dell'azione.
    """
    
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
    
    """
    Pulisce e normalizza i dati finanziari per un formato standardizzato. 
    Appiattisce eventuali indici multi-livello e resetta l'indice.
    :param data: DataFrame contenente i dati grezzi.
    :return: DataFrame pulito e standardizzato.
    """
    
    print("Pulizia dei dati...")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]
    data.columns = [col.split(' ')[0] for col in data.columns]
    data.reset_index(inplace=True)
    print(f"Dati puliti: {data.shape}")
    return data

def get_close_data(data: pd.DataFrame):
    
    """
    Estrae i prezzi di chiusura dai dati e li restituisce come lista arrotondata a 4 decimali.
    :param data: DataFrame contenente i dati storici.
    :return: Lista dei prezzi di chiusura.
    """

    close = data['Close'].values.tolist()
    close = [round(price, 4) for price in close]
    return close

def state_formatter(state):
    
    """
    Trasforma uno stato in un formato adatto per l'elaborazione. 
    Supporta input in formato `numpy.ndarray` o `list`.
    :param state: Stato da formattare.
    :return: Stato formattato come array monodimensionale.
    :raises ValueError: Se il formato dello stato non è riconosciuto.
    """

    if isinstance(state, np.ndarray):
        return state.flatten()
    elif isinstance(state, list):
        return np.array(state).flatten()
    else:
        print(f"Stato ricevuto in state_formatter: {state}, tipo: {type(state)}")
        raise ValueError("Formato dello stato non riconosciuto")

def get_data_dict(start_date, end_date, ticks:list):
    
    """
    Scarica e pulisce i dati storici per una lista di ticker, restituendo un dizionario di DataFrame.
    :param start_date: Data di inizio per i dati storici.
    :param end_date: Data di fine per i dati storici.
    :param ticks: Lista di ticker delle azioni.
    :return: Dizionario con i dati storici puliti per ciascun ticker.
    """

    data_dict = dict()
    for tick in ticks:
        data_temp = download(tick, start_date, end_date)
        data_temp = cleaning(data_temp)
        data_dict[tick] = data_temp
    return data_dict

def get_discrete_data(data):
    pass