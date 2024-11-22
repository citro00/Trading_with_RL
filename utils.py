import numpy as np
import yfinance as yf
import gymnasium as gym
import gym_anytrading
from gym_anytrading import datasets
import pandas

def make_state_hashable(state):
    """
    Converte lo stato in una tupla di float per renderlo hashable.
    
    Args:
        state (array-like): Stato corrente dell'ambiente.
    
    Returns:
        tuple: Tupla di float che rappresenta lo stato.
    """
    try:
        # Se lo stato è multi-dimensionale, appiattiscilo
        flat_state = np.array(state).flatten()
        return tuple(float(x) for x in flat_state)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Errore nella conversione dello stato a float: {e}")

#Scarica i dati storici relativi all'asset passato come parametro, nel range temporale tra start_date e end_date
def download(asset, start_date, end_date):
    try:
        data = yf.download(asset, start=start_date, end=end_date)
        if data.empty:
            print("Non sono stati trovati dati")
        else:
            print(f"Dati scaricati per {asset}")
            data.to_csv(f"{asset}_data.csv")
            print(f"Dati salavati in {asset}_data.csv")
    except Exception as e:
        print(f"Errorre durante il download dei dati: {e}")
    return data

def cleaning(data: pandas.DataFrame):
    #Trasforma il DataFrame da MultiLevelInde in plain DataFrame
    data = data.stack(level=1).rename_axis().reset_index(level=1)
    #Rimuovi nome colonne
    data.columns.name = None
    #Droppa colonna con nome ticker (inutile)
    data = data.drop(columns='Ticker')

    print("Colonne:{}".format(data.columns))
    print("Index:\n{}".format(data.index))
    print("My data head:\n{}".format(data.head()))
    return data

