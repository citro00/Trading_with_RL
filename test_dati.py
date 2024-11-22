import yfinance as yf
import gymnasium as gym
import gym_anytrading
from gym_anytrading import datasets
import pandas
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

