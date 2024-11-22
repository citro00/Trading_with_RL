import numpy as np
from gym_anytrading.envs import StocksEnv

class CustomStocksEnv(StocksEnv):
    def __init__(self, df, window_size, frame_bound, render_fps=1, initial_cash=10000):
        """
        Inizializza l'ambiente personalizzato per il trading di azioni.
        
        Args:
            df (pd.DataFrame): Dati finanziari.
            window_size (int): Dimensione della finestra di osservazione.
            frame_bound (tuple): Limiti dell'osservazione (inizio e fine).
            render_fps (int): Frequenza dei fotogrammi per il rendering.
            initial_cash (int): Quantità iniziale di denaro disponibile per il trading.
        """
        super().__init__(df, window_size, frame_bound)
        self.render_fps = render_fps
        self.cash = float(initial_cash)
        self.shares = 0
        self.initial_cash = float(initial_cash)
        self.num_trades = 0
        self.max_portfolio_value = float(initial_cash)
        self.current_step = None

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()
        diff = np.diff(prices)

        min_length = min(len(prices), len(diff))
        prices = prices[-min_length:]
        diff = diff[-min_length:]

        signal_features = np.column_stack((prices, diff))
        return prices, signal_features

    def step(self, action):
        if self.current_step is None:
            raise ValueError("`current_step` non è stato inizializzato. Assicurati di chiamare `reset()` prima di `step()`.")

        current_price = float(self.df['Close'].iloc[self.current_step].item())

        # Inizializza la reward con un valore di default
        reward = 0.0

        # Azione dell'agente
        
        if action == 1:  # Acquisto di azioni
            
            if self.cash >= current_price:
                num_shares_to_buy = self.cash // current_price
                self.shares += num_shares_to_buy
                self.cash -= num_shares_to_buy * current_price
                self.num_trades += 1
                # Piccolo reward per l'acquisto
                reward += 0.05

        elif action == 0:  # Vendita di azioni
            if self.shares > 0:
                potential_cash = self.shares * current_price
                # Verifica se il valore delle azioni è maggiore dell'investimento iniziale per garantire un profitto
                if potential_cash > self.initial_cash:
                    # Calcola il profitto e assegna il reward
                    profit = potential_cash - self.initial_cash
                    reward += profit * 0.1  # Incentiva la vendita quando si ottiene un profitto
                    self.cash += potential_cash
                    self.shares = 0
                    self.num_trades += 1
                else:
                    # Penalizza leggermente se vende a un prezzo più basso del prezzo di acquisto medio
                    reward -= 0.1
        
        else:  # Attendere
            reward -= 0.1  # Penalizza l'attesa per spingere a prendere una decisione 
       
        # Penalizza l'agente per mantenere le azioni senza venderle
        if self.shares > 0 and action != 2:  # Se possiede azioni e non vende
            reward -= 0.05  # Penalità per non vendere

        # Calcola il valore attuale del portafoglio
        previous_portfolio_value = self.portfolio_value
        self.portfolio_value = float(self.cash + (self.shares * current_price))
        reward += (self.portfolio_value - previous_portfolio_value) * 0.01  # Ricompensa proporzionale all'incremento

        # Aggiorna il massimo valore del portafoglio per il calcolo del drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)

        # Incrementa current_step
        self.current_step += 1

        # Verifica se hai raggiunto la fine dei dati
        if self.current_step >= self.frame_bound[1] or self.current_step >= len(self.df) - 1:
            terminated = True
        else:
            terminated = False

        # Chiama il metodo step originale per ottenere lo stato successivo, reward e altre informazioni
        next_state, _, _, _, info = super().step(action)

        return next_state, reward, terminated, False, info

    def reset(self):
        self.cash = float(self.initial_cash)
        self.shares = 0
        self.portfolio_value = float(self.initial_cash)
        self.num_trades = 0
        self.max_portfolio_value = float(self.initial_cash)

        initial_state = super().reset()
        self.current_step = self.frame_bound[0]

        return initial_state

    def render(self, mode='human'):
        from time import sleep
        if self.render_fps <= 0:
            raise ValueError("Il valore di 'render_fps' deve essere maggiore di 0.")

        pause_time = max(1.0 / self.render_fps, 0.1)
        print(f"Rendering frame. Pause time: {pause_time:.2f}s")
        super().render(mode)
        sleep(pause_time)
