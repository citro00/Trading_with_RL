import numpy as np
from gym_anytrading.envs import TradingEnv
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from action import Action
import random
import matplotlib.pyplot as plt
import time


class CustomStocksEnv(TradingEnv):
    """
    CustomStocksEnv è un'estensione di TradingEnv progettata per simulare un ambiente di trading di azioni. 
    Fornisce una struttura per valutare strategie di trading su più asset, utilizzando dati di mercato reali 
    o simulati. L'ambiente supporta l'acquisto, la vendita e il mantenimento di posizioni e calcola ricompense 
    basate sui profitti o perdite ottenuti.
    """

    metadata = {'render_modes': ['human'], 'render_fps': 30, 'figure_num': 999, 'plot_holds': False}

    def __init__(self, df:dict, window_size, frame_bound, normalize=True, initial_balance=1000):
        
        """
        Inizializza l'ambiente con i parametri forniti, come il bilancio iniziale, la finestra di osservazione 
        e i dati di input. Seleziona un asset casuale dai dati forniti per iniziare.
        :param df: Dizionario contenente i dati per ogni asset.
        :param window_size: Dimensione della finestra di osservazione.
        :param frame_bound: Limiti del range temporale per i dati.
        :param normalize: Flag per normalizzare i dati di input.
        :param initial_balance: Bilancio iniziale del portafoglio.
        """
        # Inizializza variabili dell'ambiente
        self.frame_bound = frame_bound
        self.initial_balance = initial_balance
        self._terminate = None
        self._step_profit = None
        self._step_reward = None
        self._actual_budget = None
        self._normalize = normalize
        self._assets_num = None
        self._done_deal = None
        self._last_trade_tick = None
        self.sell_rois = [] 
        self._last_action: tuple[int, Action] = None
        self.df_dict = df        
        self._current_asset = random.choice(list(self.df_dict.keys()))
        self._transaction_number = None
        self._delta_p = None
        self._last_assets_num = None
        
        super().__init__(df=self.df_dict[self._current_asset], window_size=window_size)
        self.action_space = spaces.Discrete(len(Action))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            #shape=(self.window_size, self.signal_features.shape[1]),
            shape=(1, self.window_size*self.signal_features.shape[1]+1),
            dtype=np.float32
        )

    def _process_data(self):
        
        """
        Prepara i dati di input per l'ambiente, includendo la normalizzazione e il calcolo delle feature 
        per la finestra di osservazione.
        :return: Prezzi e feature di segnale processate.
        """

        required_columns = ['Close', 'Volume']
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"Le colonne richieste {required_columns} non sono presenti nel DataFrame.")
        
        start = self.frame_bound[0]
        end = self.frame_bound[1]
        df = self.df.iloc[start:end].reset_index(drop=True)
        prices = df['Close'].to_numpy()
        volumes = df['Volume'].to_numpy()
        diff = np.insert(np.diff(df['Close']), 0, 0)
        signal_features = np.column_stack((prices, diff, volumes))
        
        if self._normalize:
            scaler = StandardScaler()
            self.scaler = scaler.fit(signal_features)  # Salva lo scaler per un utilizzo futuro
            signal_features = self.scaler.transform(signal_features)

        return prices, signal_features

    def _calculate_reward(self, action):
        
        """
        Calcola la ricompensa in base all'azione eseguita, considerando il profitto o la perdita e lo stato 
        del deal corrente.
        :param action: Azione eseguita (Buy, Sell, Hold).
        :return: Ricompensa associata all'azione.
        """
        
        '''drawdown = max(0, (self._get_wallet_value()/self.initial_balance)-self._delta_p)
        
        if not action==Action.Hold.value:
            transaction_cost = 0.05*self._delta_p
        else:
            transaction_cost = 0

        if self._current_tick % 30 == 0:
            self._transaction_number = 0
        if self._done_deal:
            if not action == Action.Hold.value:
                self._transaction_number += 1

        inactivity_steps = self._current_tick - self._last_trade_tick
        h_action = max(0,0.3*(self._transaction_number-15))
        h_inactivity = max(0,0.7*(inactivity_steps-5))
        h = h_action + h_inactivity

        if self._delta_p > 1:
            reward = self._delta_p - transaction_cost - h
        elif self._delta_p < 1:
            reward = self._delta_p - drawdown - transaction_cost - h
        else:
            reward = - h '''
        print(f"Action reward: {action}")
        if not self._done_deal:
            return -0.2
        delta_p = self._delta_p * 10
        transaction_cost = 0.1*delta_p
        '''if not action == Action.Hold.value:
            return (np.log(self._delta_p)*2)-transaction_cost
        else:
            return -transaction_cost'''
        if action == Action.Hold.value:
            return -transaction_cost
        else:
            return 0


        """print(f"Step: {self._current_tick}")
        print(f"Action: {action} # Done_deal: {self._done_deal}")
        print(f"Delta_p: {self._delta_p}")
        print(f"Transaction cost: {transaction_cost}")
        print(f"H: {h}")
        print(f"DrawDown: {drawdown}")
        print(f"Prezzo corrente: {self.prices[self._current_tick]}")
        print(f"Prezzo step precedente: {self.prices[self._current_tick-1]}")
        print(f"Numero di asset corrente: {self._assets_num}")
        print(f"Numero di asset step precedente: {self._last_assets_num}")
        print("###############################\n")
        time.sleep(0.5)"""
        return reward

            
    def _update_profit(self, action) :
        
        """
        Aggiorna il profitto del passo corrente in base all'azione eseguita.
        :param action: Azione eseguita (Buy, Sell, Hold).
        """
        
        #self._step_profit = (self.prices[self._current_tick]-self.prices[self._last_trade_tick])*assets
        if action == Action.Sell.value:
            self._step_profit = (self.prices[self._current_tick]-self.prices[self._last_trade_tick])*self._last_assets_num
        '''elif action == Action.Buy.value:
            self._step_profit = - (self.prices[self._current_tick])*assets'''
          

    def step(self, action):
        
        """
        Esegue un passo nell'ambiente in base all'azione selezionata. Aggiorna lo stato, calcola la ricompensa 
        e controlla se l'episodio è terminato.
        :param action: Azione selezionata (Buy, Sell, Hold).
        :return: Osservazione corrente, ricompensa del passo, stato di termine, stato di troncamento, e informazioni aggiuntive.
        """

        self._terminate = False
        self._current_tick += 1
        #self._step_profit = 0.
        self._step_reward = 0.
        self._done_deal = False

        if self._current_tick == self._end_tick:
            self._terminate = True
        
        # Tronchiamo l'esecuzione se budget è 0
        if self._actual_budget <= 0:
            self._truncated = True

        self._last_action = None
        self._last_assets_num = self._assets_num
        last_p = (self._actual_budget + (self.prices[self._current_tick-1])*self._last_assets_num)
        #print(f"Last_p: {last_p}")
        if action == Action.Buy.value and self._actual_budget >= self.prices[self._current_tick]:
            # Se l'azione selezionata è buy e il budget disponibile è superiore al prezzo corrente dell'asset: 
            self.buy()
            
        elif action == Action.Sell.value and self._assets_num > 0:
            # Se l'azione selezionata è sell e la lista degli asset acquistati non è vuota:
            self.sell()
            
        elif action == Action.Hold.value:
            self.hold()
        print(f"Action step: {action}")
        print(f"Current tick: {self._current_tick}")
        print("####################")
        actual_p = (self._actual_budget + (self.prices[self._current_tick])*self._assets_num)
        #print(f"Actual_p: {actual_p}")
        self._delta_p = actual_p / last_p
        self._update_profit(action)
        self._step_reward = self._calculate_reward(action)
        if (action == Action.Sell.value or action == Action.Buy.value) and self._done_deal:
            self._last_trade_tick = self._current_tick

        self._total_reward = self._set_total_reward(self._step_reward)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame() 
            
        return observation, self._step_reward, self._terminate, self._truncated, info
    
    
    def buy(self):
        """
        Esegue un'azione di acquisto, aggiornando il budget e il numero di asset posseduti.
        """
        
        qty = self._actual_budget // self.prices[self._current_tick]
        self._actual_budget -= (self.prices[self._current_tick] * qty)
        self._assets_num += qty

        # Salviamo il tick corrente come tick di acquisto e tick dell'ultima transazione
        self._last_action = (self._current_tick, Action.Buy)

        self._done_deal = True
    
    def sell(self): 
        """
        Esegue un'azione di vendita, aggiornando il budget e resettando il numero di asset posseduti.
        """
        
        self._actual_budget += self.prices[self._current_tick] * self._assets_num
        self._assets_num = 0
        self._done_deal = True
        self._last_action = (self._current_tick, Action.Sell)

    def hold(self):
        self._done_deal = True
        self._last_action = (self._current_tick, Action.Hold)
        
    def _seed(self, seed=None):
        """
        Imposta il seed per la riproducibilità.
        :param seed: Seed op
        """
        np.random.seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)

    def reset(self, seed=None):
        
        """
        Reinizializza l'ambiente per un nuovo episodio, resettando lo stato e selezionando un asset casuale.
        :param seed: Seed opzionale per la riproducibilità.
        :return: Osservazione iniziale e informazioni aggiuntive.       
        """
        self._step_profit = 0.
        self._total_reward = 0.
        self._step_reward = 0.
        self._actual_budget = self.initial_balance
        self._assets_num = 0
        self._last_action = None
        self._last_trade_tick = 0
        self._transaction_number = 0
        self._delta_p = 0
        self._last_assets_num = 0
        self._truncated = False 
        self._terminate = False
        obs, info = super().reset(seed=seed)

        self._total_profit = 0.
        self._seed(seed)
        self._current_asset = random.choice(list(self.df_dict.keys()))
        self.df = self.df_dict[self._current_asset]
        self.prices, self.signal_features = self._process_data()
        
        return obs, info

    def render(self, mode='human'):

        def _plot_action(tick, action):
            '''if self._current_tick != tick:
                return # Plot only last action'''
            if action == Action.Sell:
                plt.plot(tick, self.prices[tick], 'v', markersize=8, color='k', label='Sell Signal')
            elif action == Action.Buy:
                plt.plot(tick, self.prices[tick], '^', markersize=8, color='m', label='Buy Signal')
            elif action == Action.Hold and self.metadata['plot_holds']:
                plt.plot(tick, self.prices[tick], 'o', markersize=4, color='b', label='Hold Signal')


        fig = plt.figure(self.metadata.get('figure_num', 1))
        if self._first_rendering:
            self._first_rendering = False
            fig.clear()
            plt.plot(self.prices, color='k', lw=1.1, label='Price')


        if self._last_action:
            _plot_action(*self._last_action)

        plt.suptitle(
            "Total Reward: %.6f" % self._get_total_reward() + ' ~ ' +
            "Total Profit: %.6f" % self._get_total_profit() + ' ~ ' +
            "Wallet value: %.6f" % self._get_wallet_value() + ' ~ ' +
            "ROI: %.6f" % self._get_roi() + ' ~ ' +
            "Asset: %s" % self._current_asset
        )

        pause_time = (1 / self.metadata['render_fps'])
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."

        plt.pause(pause_time)

    def render_all(self, title=None):
        actions_history = [act for act in self.history.get('action', []) if act is not None]
        fig = plt.figure(self.metadata.get('figure_num', 1))
        fig.clear()

        sell_signals = [tick for (tick, action) in actions_history if action == Action.Sell]
        buy_signals = [tick for (tick, action) in actions_history if action == Action.Buy]
        hold_signals = [tick for (tick, action) in actions_history if action == Action.Hold]

        plt.plot(self.prices, color='k', lw=1.1, label='Price')
        plt.plot(buy_signals, self.prices[buy_signals], '^', markersize=8, color='m', label='Buy Signal')
        plt.plot(sell_signals, self.prices[sell_signals], 'v', markersize=8, color='k', label='Sell Signal')
        if self.metadata['plot_holds']:
            plt.plot(hold_signals, self.prices[hold_signals], 'o', markersize=4, color='b', label='Hold Signal')

        if title:
            plt.title(title)
        
        plt.suptitle(
            "Total Reward: %.6f" % self._get_total_reward() + ' ~ ' +
            "Total Profit: %.6f" % self._get_total_profit() + ' ~ ' +
            "Wallet value: %.6f" % self._get_wallet_value() + ' ~ ' +
            "ROI: %.6f" % self._get_roi() + ' ~ ' +
            "Asset: %s" % self._current_asset
        )

        plt.xlabel('Tick')
        plt.ylabel('Price')
        plt.legend()
        
        pause_time = (1 / self.metadata['render_fps'])
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."

        plt.pause(pause_time)

    def _get_observation(self):
        """
        Genera l'osservazione corrente basata sulla finestra temporale specificata, includendo padding se necessario.
        :return: Osservazione corrente come array di feature.
        """

        start = self._current_tick - self.window_size
        end = self._current_tick
        if start < 0:
            start = 0

        if end > len(self.signal_features):
            end = len(self.signal_features)
            
        obs = self.signal_features[start:end]
        if len(obs) < self.window_size:
            padding = np.zeros((self.window_size - len(obs), self.signal_features.shape[1]))
            obs = np.vstack((padding, obs))
        obs = obs.flatten()
        obs = np.concatenate((obs, [self._get_total_profit()/self.initial_balance]), axis=0)

        return obs
    
    def _get_info(self):
        """
        Genera un dizionario con informazioni dettagliate sullo stato corrente dell'ambiente, inclusi 
        profitti, bilancio e dettagli dell'asset.
        :return: Dizionario con informazioni sullo stato.
        """

        return dict(
            step_reward   = self._step_reward,
            total_reward  = self._get_total_reward(),
            step_profit   = self._step_profit,
            total_profit  = self._get_total_profit(),
            roi           = self._get_roi(),  
            wallet_value = self._get_wallet_value(),
            action        = self._last_action,
            actual_budget = self._actual_budget,
            asset         = self._current_asset,
            max_price = np.max(self.prices),
            min_price = np.min(self.prices),
            df_lenght = len(self.df)
        )
    
    def _get_wallet_value(self):
        """
        Calcola il valore totale del portafoglio, includendo il bilancio e il valore degli asset posseduti.
        :return: Valore totale del portafoglio.
        """
        return self._actual_budget + (self.prices[self._current_tick]*self._assets_num)
    
    def _get_total_reward(self):
        """
        Restituisce la ricompensa totale accumulata fino al momento corrente.
        :return: Ricompensa totale.
        """
        return self._total_reward
    
    def _get_total_profit(self):
        """
        Calcola il profitto totale accumulato fino al momento corrente.
        :return: Profitto totale.
        """
        return self._get_wallet_value() - self.initial_balance
    
    def _get_roi(self):
        """
        Calcola il ritorno sull'investimento (ROI) in percentuale.
        :return: ROI in percentuale.
        """
        return (self._get_total_profit() / self.initial_balance) * 100
    
    def _set_total_reward(self, step_reward):
        """
        Aggiorna la ricompensa totale accumulata aggiungendo la ricompensa del passo corrente.
        :param step_reward: Ricompensa del passo corrente.
        :return: Nuova ricompensa totale.
        """

        return self._total_reward + step_reward
