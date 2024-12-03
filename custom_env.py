import numpy as np
import pandas as pd
from gym_anytrading.envs import TradingEnv
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from action import *
from position import *

class CustomStocksEnv(TradingEnv):
    """
    Ambiente di trading personalizzato estendendo TradingEnv da gym_anytrading.
    """

    def __init__(self, df, window_size, frame_bound, initial_balance=1000):
        """
        Inizializza l'ambiente personalizzato.
        
        :param df: DataFrame contenente i dati storici di trading.
        :param window_size: Dimensione della finestra di osservazione (quanti giorni considerare come input).
        :param frame_bound: Limiti di indice per i dati da utilizzare (inizio e fine).
        :param initial_balance: Bilancio iniziale del portafoglio dell'agente.
        """
        # Inizializza variabili dell'ambiente
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.initial_balance = initial_balance

        # Richiama il costruttore della classe base (TradingEnv)
        super().__init__(df=df, window_size=window_size)

        # Prepara i dati e crea le feature da usare come input per l'agente
        self.prices, self.signal_features = self._process_data()

        # Definisce lo spazio delle azioni (0 = Hold, 1 = Buy, 2 = Sell)
        self.action_space = spaces.Discrete(len(Action))

        # Definisce lo spazio delle osservazioni, ovvero l'input che l'agente riceverà (osservazioni passate)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.signal_features.shape[1]),
            dtype=np.float32
        )

        # Stato iniziale dell'ambiente
        self._position = Position.Short  # 0: posizione short
        self._entry_price = 0  # Prezzo di entrata per una posizione
        self.total_profit = 0  # Profitto totale
        self._previous_portfolio_value = self.initial_balance  # Valore del portafoglio precedente

    def _process_data(self):
        """
        Prepara i dati per l'utilizzo nell'ambiente.

        :return: Prezzi e feature di segnale normalizzate.
        """
        # Controlla che ci siano le colonne necessarie 
        required_columns = ['Close', 'Volume']
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"Le colonne richieste {required_columns} non sono presenti nel DataFrame.")
        # Ottieni il range dei dati da usare basandoti sui limiti specificati
        start = self.frame_bound[0]
        end = self.frame_bound[1]
        df = self.df.iloc[start:end].reset_index(drop=True)

        # Estrai i prezzi di chiusura
        prices = df['Close'].to_numpy()

        # Estrai le feature che serviranno come input per l'agente (Close e Volume)
        signal_features = df[['Close', 'Volume']].to_numpy()

        # Normalizza le feature per avere valori con media 0 e deviazione standard 1
        scaler = StandardScaler()
        self.scaler = scaler.fit(signal_features)  # Salva lo scaler per un utilizzo futuro
        signal_features = self.scaler.transform(signal_features)

        return prices, signal_features

    def _calculate_reward(self, action): # todo ristrutturare
        """
        Calcola il reward basato sull'azione eseguita.

        :param action: Azione scelta dall'agente (0 = Hold, 1 = Buy, 2 = Sell)
        :return: Reward ottenuto dall'azione.
        """
        transaction_cost = 0.001  # 0.1% di costo di transazione

        # Calcola il valore attuale del portafoglio
        cash = self.initial_balance + self.total_profit
        current_holdings = self.prices[self._current_tick] if self._position == 1 else 0
        portfolio_value = cash + current_holdings * self._position

        # Calcola il reward basato sulla variazione del valore del portafoglio
        if hasattr(self, '_previous_portfolio_value'):
            step_reward = portfolio_value - self._previous_portfolio_value
        else:
            step_reward = 0

        # Penalità per comprare o vendere in modo inappropriato
        if action == 1:  # Buy
            if self._position == 1:
                step_reward -= 0.1  # Penalità per comprare quando già in posizione
            else:
                step_reward -= transaction_cost  # Costo di transazione per comprare
        elif action == 2:  # Sell
            if self._position == 0:
                step_reward -= 0.1  # Penalità per vendere senza avere posizioni aperte
            else:
                step_reward -= transaction_cost  # Costo di transazione per vendere

        # Incentivo per mantenere posizioni profittevoli
        if self._position == 1:
            unrealized_profit = self.prices[self._current_tick] - self._entry_price
            step_reward += unrealized_profit * 0.1  # Incentivo per mantenere posizioni profittevoli

        # Incentivo per vendere in profitto
        if action == 2 and self._position == 1:
            realized_profit = self.prices[self._current_tick] - self._entry_price
            if realized_profit > 0:
                step_reward += realized_profit * 0.1  # Incentivo per vendere in profitto

        # Penalità per mantenere posizioni troppo a lungo
        if self._position == 1:
            holding_duration = self._current_tick - self._entry_price
            step_reward -= 0.001 * holding_duration  # Penalità per mantenere una posizione troppo a lungo

        # Reward relativo alla performance del mercato (misura il successo dell'agente rispetto al mercato)
        market_return = (self.prices[self._current_tick] / self.prices[0]) - 1
        agent_return = self.total_profit / self.initial_balance
        relative_reward = (agent_return - market_return) * 0.1
        step_reward += relative_reward

        # Normalizza il reward per ridurre la sua variabilità
        step_reward = step_reward / 10
        self._previous_portfolio_value = portfolio_value

        # Aggiorna la posizione e i dati di profitto totali basandosi sull'azione presa
        if action == 1:  # Buy
            if self._position == 0:
                self._position = 1
                self._entry_price = self.prices[self._current_tick]
        elif action == 2:  # Sell
            if self._position == 1:
                self._position = 0
                price_diff = self.prices[self._current_tick] - self._entry_price
                self.total_profit += price_diff

        return step_reward

    def _update_profit(self, action):
        """
        Aggiorna il profitto totale e calcola la ricompensa in base all'azione eseguita.

        :param action: Azione scelta dall'agente.
        :return: Reward ottenuto dall'azione.
        """
        step_reward = self._calculate_reward(action)
        return step_reward

    def _get_observation(self):
        """
        Ottiene l'osservazione corrente basata sulla finestra di osservazione.

        :return: Array numpy che rappresenta l'osservazione corrente.
        """
        start = self._current_tick - self.window_size
        end = self._current_tick

        if start < 0:
            start = 0

        if end > len(self.signal_features):
            end = len(self.signal_features)

        obs = self.signal_features[start:end]

        # Aggiungi padding se l'osservazione è inferiore alla dimensione della finestra
        if len(obs) < self.window_size:
            padding = np.zeros((self.window_size - len(obs), self.signal_features.shape[1]))
            obs = np.vstack((padding, obs))

        return obs

    def _get_info(self):
        """
        Ottiene informazioni aggiuntive sull'ambiente.

        :return: Dizionario con informazioni sul profitto totale e la posizione corrente.
        """
        return {
            'total_profit': self.total_profit,
            'position': self._position
        }

    def get_current_tick(self):
        """
        Ottiene l'indice corrente (il tick) del trading.

        :return: L'indice attuale.
        """
        return self._current_tick

    def step(self, action):
        # TO DO
        return super().step()
 
    def reset(self):
        """
        Resetta l'ambiente al suo stato iniziale.

        :return: L'osservazione iniziale e le informazioni dell'ambiente.
        """
        obs, info = super().reset()
        self._position = 0
        self._entry_price = 0
        self.total_profit = 0
        self._previous_portfolio_value = self.initial_balance
        return obs, info
