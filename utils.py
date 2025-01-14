import numpy as np
import pandas as pd
import yfinance as yf

def fetch_stock_data(tickers, start_date=None, end_date=None, interval='1d'):
    """
    Recupera i dati storici delle azioni specificate.
    Scarica i dati storici per ciascun ticker fornito utilizzando yfinance.
    Args:
        tickers (list): Lista di simboli delle azioni da scaricare.
        start_date (str, opzionale): Data di inizio nel formato 'YYYY-MM-DD'. Defaults to None.
        end_date (str, opzionale): Data di fine nel formato 'YYYY-MM-DD'. Defaults to None.
        interval (str, opzionale): Intervallo dei dati ('1d', '1wk', '1mo', ecc.). Defaults to '1d'.
    Returns:
        dict: Dizionario con i ticker come chiavi e i rispettivi DataFrame dei dati storici come valori.
    """
    
    stock_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)
        stock_data[ticker] = data
    return stock_data

def compute_metrics(data):
    """
    Calcola metriche aggiuntive sui dati delle azioni.
    Aggiunge colonne per i rendimenti giornalieri, rendimenti cumulativi, medie mobili e VWAP.
    Args:
        data (dict): Dizionario con i ticker come chiavi e i rispettivi DataFrame dei dati storici come valori.
    Returns:
        dict: Dizionario con i ticker come chiavi e i rispettivi DataFrame con le metriche calcolate come valori.
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
    Scarica i dati storici di un'azione specifica e li salva in un file CSV.
    Utilizza yfinance per scaricare i dati e li salva in un file nella directory 'csv'.
    Args:
        asset (str): Simbolo dell'azione da scaricare.
        start_date (str): Data di inizio nel formato 'YYYY-MM-DD'.
        end_date (str): Data di fine nel formato 'YYYY-MM-DD'.
    Returns:
        pd.DataFrame: DataFrame contenente i dati storici scaricati.
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
    Pulisce il DataFrame rimuovendo eventuali multi-indici e reindicizzando.
    Rinomina le colonne se necessario e reimposta l'indice.
    Args:
        data (pd.DataFrame): DataFrame da pulire.
    Returns:
        pd.DataFrame: DataFrame pulito.
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
    Estrae i prezzi di chiusura da un DataFrame.
    Converte i prezzi di chiusura in una lista di valori arrotondati a quattro decimali.
    Args:
        data (pd.DataFrame): DataFrame contenente la colonna 'Close'.
    Returns:
        list: Lista dei prezzi di chiusura arrotondati.
    """
    close = data['Close'].values.tolist()
    close = [round(price, 4) for price in close]
    return close

def state_formatter(state):
    """
    Estrae i prezzi di chiusura da un DataFrame.
    Converte i prezzi di chiusura in una lista di valori arrotondati a quattro decimali.
    Args:
        data (pd.DataFrame): DataFrame contenente la colonna 'Close'.
    Returns:
        list: Lista dei prezzi di chiusura arrotondati.
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
    Crea un dizionario di dati storici per una lista di ticker.
    Scarica, pulisce e organizza i dati per ciascun ticker.
    Args:
        start_date (str): Data di inizio nel formato 'YYYY-MM-DD'.
        end_date (str): Data di fine nel formato 'YYYY-MM-DD'.
        ticks (list): Lista di simboli delle azioni da scaricare.
    Returns:
        dict: Dizionario con i ticker come chiavi e i rispettivi DataFrame puliti come valori.
    """
    data_dict = dict()
    for tick in ticks:
        data_temp = download(tick, start_date, end_date)
        data_temp = cleaning(data_temp)
        data_dict[tick] = data_temp
    return data_dict

