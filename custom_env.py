import numpy as np
from gym_anytrading.envs import TradingEnv
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from action import Action
import random
import matplotlib.pyplot as plt



class CustomStocksEnv(TradingEnv):

    metadata = {'render_modes': ['human'], 'render_fps': 30, 'figure_num': 999, 'plot_holds': False}

    def __init__(self, df:dict, window_size, frame_bound, normalize=True, initial_balance=1000):
        
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
        self._last_action: tuple[int, Action] = None
        self.df_dict = df        
        self._current_asset = random.choice(list(self.df_dict.keys()))
        self._transaction_number = None
        self._delta_p = None
        self._delta_p_normalized = None
        self._drawdown = None
        self._h = None
        self._last_assets_num = None
        self._max_wallet_value = None
        self._min_wallet_value = None
        
        super().__init__(df=self.df_dict[self._current_asset], window_size=window_size)
        self.action_space = spaces.Discrete(len(Action))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            #shape=(self.window_size, self.signal_features.shape[1]),
            shape=(1, self.window_size*self.signal_features.shape[1]+1),
            dtype=np.float32
        )

    def _process_data(self):
        

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
        

        beta1=0.02  # Peso penalità per trading eccessivo
        beta2=0.01  # Peso penalità per inattività
        alpha=0.3 # Peso penalità per drawdown
        lambda_T=0.005  # Peso riduzione costi di transazione
        lambda_D=0.5  # Peso riduzione drawdown
        lambda_H=0.03  # Peso riduzione penalità comportamentali
        self._delta_p_normalized = self._delta_p/(self._max_wallet_value-self._min_wallet_value)

        # Calcolo il transaction cost rispetto al prezzo corrente dell'azione
        transaction_cost = abs(0.05*self.prices[self._current_tick])
        transaction_cost_norm = transaction_cost * lambda_T

        if ((self._max_wallet_value - self._get_wallet_value()) / self._max_wallet_value) > 0.5:
            # Se la perdita allo step attuale è più grande del 10 percento del valore massimo di drawdown setta drawdown
            drawdown = max(0, alpha*((self._max_wallet_value - self._get_wallet_value()) / self._max_wallet_value))
        else:
            drawdown = 0

        # Setto il parametro per troppo trading
        h_trading_eccessivo = max(0, beta1*(self._transaction_number))

        # Calcolo step inattivi
        step_inattivo = self._current_tick - self._last_trade_tick
        
        # Setto parametro troppo inattivo
        h_inattività = max(0, beta2 *(step_inattivo))
        
        # Calcolo penalità comportamentale 
        h = h_trading_eccessivo + h_inattività

        # Calcolo i parametri di penalizzazione normalizzati
        self._drawdown = drawdown * lambda_D
        self._h = h * lambda_H
        
        if self._done_deal:
            reward = self._delta_p_normalized - transaction_cost_norm - self._drawdown - self._h
        else:
            reward = - 0.5
        return reward
        """
        def _calculate_reward(self, action):
            # QUESTA FUNZIONA. NON TOCCARE !!!!!
            beta1=2.5  # Peso penalità per trading eccessivo
            beta2=1.2  # Peso penalità per inattività
            alpha=1.4 # Peso penalità per drawdown
            lambda_T=0.005  # Peso riduzione costi di transazione
            lambda_D=1.5  # Peso riduzione drawdown
            lambda_H=2  # Peso riduzione penalità comportamentali
            limite_penalità=100  # Soglia massima per penalità complessiva
            #min_price = min(self.prices)
            #max_price = max(self.prices)

            self._delta_p_normalized = self._delta_p/(self._max_wallet_value-self._min_wallet_value)

            # Calcolo il transaction cost rispetto al prezzo corrente dell'azione
            transaction_cost = abs(0.05*self.prices[self._current_tick])
            transaction_cost_norm = transaction_cost * lambda_T

            if self._done_deal:
                reward = self._delta_p_normalized - transaction_cost_norm
            else:
                reward = - 0.5
        
            return reward
         """      
    
    def _update_profit(self, action) :
        
        if action == Action.Sell.value:
            self._step_profit = (self.prices[self._current_tick]-self.prices[self._last_trade_tick])*self._last_assets_num
        #aggiornamnto delta p dentro

    def update_max_min_wallet_value(self, actual_wallet_value):
        self._max_wallet_value = max(self._max_wallet_value, actual_wallet_value)
        self._min_wallet_value = min(self._min_wallet_value, actual_wallet_value)

        

    def step(self, action):
        
        self._terminate = False
        self._current_tick += 1

        self._step_reward = 0.
        self._done_deal = False

        if self._current_tick == self._end_tick:
            self._terminate = True
        
        # Tronchiamo l'esecuzione se budget è 0
        if self._actual_budget <= 0:
            self._truncated = True

        self._last_action = None
        self._last_assets_num = self._assets_num
        last_p = (self._actual_budget + self.prices[self._last_trade_tick]*self._last_assets_num)
        
        if self._current_tick % 30 == 0:
            self._transaction_number=0
        if action == Action.Buy.value and self._actual_budget >= self.prices[self._current_tick]:
            # Se l'azione selezionata è buy e il budget disponibile è superiore al prezzo corrente dell'asset: 
            self.buy()
            
        elif action == Action.Sell.value and self._assets_num > 0:
            # Se l'azione selezionata è sell e la lista degli asset acquistati non è vuota:
            self.sell()
            
        elif action == Action.Hold.value:
            self.hold()
        actual_p = (self._actual_budget + self.prices[self._current_tick]*self._assets_num)
        self._delta_p = actual_p - last_p
        self._update_profit(action)
        self.update_max_min_wallet_value(self._get_wallet_value())
        self._step_reward = self._calculate_reward(action)
        if (action == Action.Sell.value or action == Action.Buy.value) and self._done_deal:
            self._last_trade_tick = self._current_tick
            self._transaction_number += 1
        self._total_reward = self._set_total_reward(self._step_reward)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame() 
            
        return observation, self._step_reward, self._terminate, self._truncated, info
    
    
    def buy(self):
        
        qty = self._actual_budget // self.prices[self._current_tick]
        self._actual_budget -= (self.prices[self._current_tick] * qty)
        self._assets_num += qty
        # Salviamo il tick corrente come tick di acquisto e tick dell'ultima transazione
        self._last_action = (self._current_tick, Action.Buy)
        self._done_deal = True
    
    def sell(self): 

        self._actual_budget += self.prices[self._current_tick] * self._assets_num
        self._assets_num = 0
        self._done_deal = True
        self._last_action = (self._current_tick, Action.Sell)

    def hold(self):
        self._done_deal = True
        self._last_action = (self._current_tick, Action.Hold)
        
    def _seed(self, seed=None):
        
        np.random.seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)

    def reset(self, seed=None):
        
        self._step_profit = 0.
        self._total_reward = 0.
        self._step_reward = 0.
        self._actual_budget = self.initial_balance
        self._assets_num = 0
        self._last_action = None
        self._last_trade_tick = 0
        self._transaction_number = 0
        self._delta_p = 0
        self._delta_p_normalized = 0
        self._last_assets_num = 0
        self._max_wallet_value = 0
        self._min_wallet_value = 0
        self._drawdown = 0
        self._h = 0
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
            if action == Action.Sell:
                plt.plot(tick, self.prices[tick], 'v', markersize=8, color='r', label='Sell Signal')
            elif action == Action.Buy:
                plt.plot(tick, self.prices[tick], '^', markersize=8, color='g', label='Buy Signal')
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
        plt.plot(buy_signals, self.prices[buy_signals], '^', markersize=8, color='g', label='Buy Signal')
        plt.plot(sell_signals, self.prices[sell_signals], 'v', markersize=8, color='r', label='Sell Signal')
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
        
        obs_arr = []
        for col in range(0,obs.shape[1]):
            obs_arr.append(obs[:,col])
        obs_arr.append(self._get_total_profit()/self.initial_balance)
        return obs_arr
    
    def _get_info(self):
       
        return dict(
            step_reward   = self._step_reward,
            total_reward  = self._get_total_reward(),
            step_profit   = self._step_profit,
            delta_p = self._h,
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
        return self._actual_budget + (self.prices[self._current_tick]*self._assets_num)
    
    def _get_total_reward(self):
        return self._total_reward
    
    def _get_total_profit(self):
        return self._get_wallet_value() - self.initial_balance
    
    def _get_roi(self):
        return (self._get_total_profit() / self.initial_balance) * 100
    
    def _set_total_reward(self, step_reward):
        return self._total_reward + step_reward
