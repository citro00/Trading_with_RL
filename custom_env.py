import numpy as np
import pandas as pd
from gym_anytrading.envs import TradingEnv
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from action import Action
# from position import *
from position import Positions
import random

class CustomStocksEnv(TradingEnv):
    """
    Ambiente di trading personalizzato estendendo TradingEnv da gym_anytrading.
    """

    def __init__(self, df:dict, window_size, frame_bound, initial_balance=1000):
        """
        Inizializza l'ambiente personalizzato.
        
        :param df: DataFrame contenente i dati storici di trading.
        :param window_size: Dimensione della finestra di osservazione (quanti giorni considerare come input).
        :param frame_bound: Limiti di indice per i dati da utilizzare (inizio e fine).
        :param initial_balance: Bilancio iniziale del portafoglio dell'agente.
        """

        # Inizializza variabili dell'ambiente
        self.frame_bound = frame_bound
        self.initial_balance = initial_balance
        self._terminate = None
        self._step_profit = None
        self._step_reward = None
        self._actual_budget = None
        self._purchased_assets = None
        self._done_deal = None
        self._last_trade_tick = None
        self._last_buy = None
        self.sell_rois = [] 

        self.df_dict = df        
        # Inizializza la posizione come Flat
        self._position = Positions.Flat #AGGIUNTO DA ME CARMINE

        self._current_asset = random.choice(list(self.df_dict.keys()))

        # Richiama il costruttore della classe base (TradingEnv)
        super().__init__(df=self.df_dict[self._current_asset], window_size=window_size)

        # Definisce lo spazio delle azioni (0 = Hold, 1 = Buy, 2 = Sell)
        self.action_space = spaces.Discrete(len(Action))

        # Definisce lo spazio delle osservazioni, ovvero l'input che l'agente riceverà (osservazioni passate)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.signal_features.shape[1]),
            dtype=np.float32
        )

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
        #estraggo i volumi
        volumes = df['Volume'].to_numpy()

        # Estrai le feature che serviranno come input per l'agente (Close e Volume)
        diff = np.insert(np.diff(df['Close']), 0, 0)
        
        #signal_features = np.column_stack((prices,diff))
        
        signal_features = np.column_stack((prices, diff, volumes))
        
        #signal_features = df['Close'].to_numpy()
        
        # Normalizza le feature per avere valori con media 0 e deviazione standard 1
        scaler = StandardScaler()
        self.scaler = scaler.fit(signal_features)  # Salva lo scaler per un utilizzo futuro
        signal_features = self.scaler.transform(signal_features)

        return prices, signal_features

    def _calculate_reward(self, action, time_step):
            
        if action == Action.Sell.value and self._total_profit > 0 and self._done_deal:
            return self.prices[self._current_tick]/self.prices[self._last_buy]
        elif action == Action.Sell.value and self._total_profit <= 0 and self._done_deal:
            return np.log(self.prices[self._current_tick]/self.prices[self._last_buy])+1

        if action == Action.Hold.value and self.prices[self._current_tick] > self.prices[self._last_buy]:
            return 2
        elif action == Action.Hold.value and self.prices[self._current_tick] <= self.prices[self._last_buy]:
            return -4
        elif action == Action.Hold.value and self.prices[self._current_tick] < self.prices[self._last_trade_tick]:
            return 1
        elif action == Action.Hold.value and self.prices[self._current_tick] >= self.prices[self._last_trade_tick]:
            return -2
        
        if action == Action.Buy.value and self._done_deal:
            return np.log(self.prices[self._current_tick]/self.prices[self._last_trade_tick])
        
        if not self._done_deal:
            return -3
        
        return 0
            
    def _update_profit(self, action) :
        """
        Aggiorna il profitto totale e calcola la ricompensa in base all'azione eseguita.

        :param action: Azione scelta dall'agente.
        :return: Reward ottenuto dall'azione.
        """
        
        self._total_profit = self._actual_budget - self.initial_balance
        if action == Action.Sell.value:
          self._step_profit = (self.prices[self._current_tick]-self.prices[self._last_buy])
        #self._total_profit = self._actual_budget - self.initial_balance


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

    def step(self, action):
        # Reset parametri iniziali
        self._terminate = False
        self._current_tick += 1
        self._step_profit = 0.
        self._step_reward = 0.
        self._done_deal = False

        # Controllimao che l'episodio non sia terminato
        if self._current_tick == self._end_tick:
            self._terminate = True
        
        # Controllimao che il budget disponibile non sia a 0 (altrimenti tronchiamo l'esecuzione)
        if self._actual_budget <= 0:
            self._truncated = True

        if action == Action.Buy.value and self._actual_budget >= self.prices[self._current_tick]:
            # Se l'azione selezionata è buy e il budget disponibile è superiore al prezzo corrente dell'asset: 
            self.buy()

        elif action == Action.Sell.value and len(self._purchased_assets)>0:
            # Se l'azione selezionata è sell e la lista degli asset acquistati non è vuota:
            self.sell()
        elif action == Action.Hold:
            self._position = Positions.Flat
            self._done_deal = True
            
        # Calcoliamo il profitto data l'azione scelta
        self._update_profit(action)

        # Calcoliamo la reard
        self._step_reward = self._calculate_reward(action, self._current_tick)
        
        # Aggiorniamo la reward totale totale dell'episodio
        self._total_reward += self._step_reward

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame() 

        #self.print_env_var(action)
        if self._terminate or self._truncated:
            info['sell_rois'] = self.sell_rois
            

        return observation, self._step_reward, self._terminate, self._truncated, info
    
    def buy(self):
        # Sottraiamo dal budget attuale il prezzo dell'asset (compriamo)
        num_action = round(self._actual_budget/self.prices[self._current_tick])-1

        if num_action <= 1:
            num_action = 1
        
        self._actual_budget -= (self.prices[self._current_tick]*num_action)
        
        # Salviamo il tick corrente come il tick in cui è stato effettuata l'ultima compera
        self._last_buy = self._current_tick
        
        # Aggiungiamo il prezzo corrente alla lista delle azioni acquistate
        for i in range(0, num_action):
            self._purchased_assets.append(self.prices[self._current_tick])
        
        # Settiamo la posizione a Long
        self._position = Positions.Long
        
        # Aggiorna last_trade
        self._last_trade_tick = self._current_tick
        self._done_deal = True
    
    def sell(self): 
        self._actual_budget += (self.prices[self._current_tick] * len(self._purchased_assets))
        
        self._purchased_assets.clear()
        
        self._position = Positions.Short
        self._done_deal = True
        self._last_trade_tick = self._current_tick
    
    
    def _get_info(self):
        return dict(
            step_reward  = self._step_reward,
            total_reward = self._total_reward,
            step_profit  = self._step_profit,
            total_profit = self._total_profit,
            position     = self._position,
            asset = self._current_asset
        )

    def reset(self, seed=None):
        """
        Resetta l'ambiente al suo stato iniziale.

        :return: L'osservazione iniziale e le informazioni dell'ambiente.
        """
        self._current_asset = random.choice(list(self.df_dict.keys()))
        self.df = self.df_dict[self._current_asset]
        self._total_profit = 0.
        self._step_profit = 0.
        self._total_reward = 0.
        self._step_reward = 0.
        self._actual_budget = self.initial_balance
        self._purchased_assets = []
        self._last_trade_tick = 0
        self._last_buy = 0
        self._reward_history = []  # TODO: rimuovere (Vincenzo)
        self._position = Positions.Short
        self._truncated = False  # Reset del flag truncated
        obs, info = super().reset(seed=seed)
        return obs, info

    #def print_env_var(self, action):
     #   print("##############################################")
      #  print(f"Intial Balance: {self.initial_balance}\n"+\
       #       #f"Terminate: {self._terminate}\n"+\
        #      #f"Truncated: {self._truncated}\n"+\
         #     f"Action:{action}\n"+\
          #    f"Done deal: {self._done_deal}\n"+\
           #   f"Step profit: {self._step_profit}\n"+\
            #  f"Total profit: {self._total_profit}\n"+\
             # f"Actual budget: {self._actual_budget}\n"+\
       #       f"Actual price: {self.prices[self._current_tick]}\n"+\
        #      f"Length of purchased asset: {len(self._purchased_assets)}\n"+\
         #     #f"Wallet value: {self._wallet_value}\n"+\
              #f"Type of wallet_vale var: {type(self._wallet_value)}\n"+\
        #      f"Current tick: {self._current_tick}\n"+\
         #     f"Last buy tick: {self._last_buy}\n"+\
              #f"Position: {self._position}\n"+\
          #    f"Total reward: {self._total_reward}\n"+\
           #   f"Step reward: {self._step_reward}")

    def get_current_tick(self):
        """
        Ottiene l'indice corrente (il tick) del trading.

        :return: L'indice attuale.
        """
        return self._current_tick
    
    def get_done_deal(self):
        return self._done_deal
    
    def get_total_profit(self):
        return self._total_profit + (self.prices[self._current_tick]*len(self._purchased_assets))
    
    def save_reward_history(self, name):
        # reward_history = pd.DataFrame(self._reward_history, columns=['Tick', 'Reward', 'Step_profit', 'Total_profit'])
        # reward_history.to_csv(f'./csv/{name}')
        history = pd.DataFrame.from_dict(self.history)
        history.to_csv(f'./csv/{name}')