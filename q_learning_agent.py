import os
from typing import Literal
from collections import defaultdict
import numpy as np
import random
from gym_anytrading.envs import TradingEnv
from tqdm import tqdm
import utils as ut
from plots import MetricPlots


class QLAgent:
    """
     Agente Q-Learning per il trading.
     Utilizza una tabella Q per apprendere la migliore politica di trading attraverso interazioni con l'ambiente.
     Args:
         action_size (int): Numero di azioni possibili.
         render_mode (Literal['step', 'episode', 'off'], opzionale): Modalità di rendering. Defaults to 'off'.
     """
    def __init__(self, action_size, gamma=0.95, epsilon_decay=0.991, lr=0.001, render_mode: Literal['step', 'episode', 'off']='off'):
        
        self.action_size = action_size
        self.render_mode = render_mode

        self.gamma = gamma
        self.epsilon = 1.0 
        self.epsilon_min = 0.01  
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr
        self.q_values = defaultdict(lambda: np.zeros(self.action_size))
        self._metrics_display = MetricPlots()

    def act(self, obs):
        """
        Seleziona un'azione basata sulla politica epsilon-greedy.
        Args:
            obs (tuple): Stato corrente discretizzato.
        Returns:
            int: Azione selezionata.
        """
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)

        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        """
        Aggiorna la tabella Q utilizzando la formula di aggiornamento Q-Learning.
        Args:
            obs (tuple): Stato corrente.
            action (int): Azione eseguita.
            reward (float): Ricompensa ricevuta.
            terminated (bool): Indicatore di fine episodio.
            next_obs (tuple): Stato successivo.
        Returns:
            float: Errore temporale calcolato.
        """
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (reward + self.gamma * future_q_value - self.q_values[obs][action])
        self.q_values[obs][action] = (self.q_values[obs][action] + self.learning_rate * temporal_difference)
        
        return temporal_difference
    
    def decay_epsilon(self):
        """
        Riduce l'epsilon per diminuire progressivamente l'esplorazione.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_agent(self, env:TradingEnv, episodes, seed=None):
        """
        Addestra l'agente attraverso interazioni con l'ambiente.
        Args:
            env (TradingEnv): Ambiente di trading.
            episodes (int): Numero di episodi di addestramento.
            seed (bool, opzionale): Se True, imposta un seed per la riproducibilità. Defaults to False.
        """
        per_step_metrics = {
            'step_reward': [],
            'delta_p': [],
            'drawdown': [],
        }
        
        per_episode_metrics = {
            'roi': [],
            'total_reward': [],
            'total_profit': [],
            'performance': [],
            'loss': [],
            'epsilon': [],
            'deal_actions_num': [],
            'deal_errors_num': [],
            'drawdown_mean': [],
        }

        if seed is not None:
            self.seed_everything(seed)

        print(f"Inizio addestramento per {episodes} episodi.")
        for episode in tqdm(range(1, episodes + 1), desc="Training Progress", unit="episode"):
            state, info = env.reset(seed=episode if seed else None)
            max_possible_profit = env.max_possible_profit()
            prices = state[0]
            profit = state[-1]
            state = ut.state_formatter(prices)
            state = np.concatenate((state, [profit]), axis=0)
            state = self._discretize(state, info["max_price"], info["min_price"])

            for metric in per_step_metrics.keys():
                per_step_metrics[metric] = []

            if self.render_mode == 'step':
                env.render()

            done = False

            while not done:
                total_loss, loss_count = 0, 0.
                action = self.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_prices = next_state[0]
                next_profit = next_state[-1]
                next_state = ut.state_formatter(next_prices)
                next_state = np.concatenate((next_state, [next_profit]), axis=0)
                next_state = self._discretize(next_state, info["max_price"], info["min_price"])
                td_error = self.update(state, action, reward, terminated, next_state)
                total_loss += td_error
                loss_count += 1

                done = terminated or truncated
                state = next_state
                if self.render_mode == 'step':
                    env.render()
                for metric, arr in per_step_metrics.items():
                    arr.append(info[metric])

                if self.render_mode == 'step':
                    self._metrics_display.plot_metrics(**per_step_metrics)

            self.decay_epsilon()

            if self.render_mode == 'episode':
                env.render_all(f"Episode {episode}")

            # Salva le metriche per l'episodio corrente
            average_loss = total_loss / loss_count if loss_count > 0 else 0
            for metric in per_episode_metrics.keys():
                if metric in info.keys():
                    per_episode_metrics[metric].append(info[metric])
            
            for metric in per_episode_metrics.keys():
                if metric in info.keys():
                    per_episode_metrics[metric].append(info[metric])
            
            per_episode_metrics['performance'].append((info['total_profit'] / max_possible_profit) * 100)
            per_episode_metrics['loss'].append(average_loss)
            per_episode_metrics['epsilon'] = self.epsilon
            per_episode_metrics['drawdown_mean'].append(np.mean(per_step_metrics['drawdown']))

            if self.render_mode == 'episode':
                self._metrics_display.plot_metrics(**per_step_metrics)
                self._metrics_display.plot_metrics(**per_episode_metrics)

            if self.render_mode == 'episode':
                pass
            total_profit = info['total_profit']
            wallet_value = info['wallet_value']
            roi = info['roi']

            tqdm.write(f"Episode {episode}/{episodes} # Dataset: {info['asset']} # ROI: {roi:.2f}% # Total Profit: {total_profit:.2f} # Wallet value: {wallet_value:.2f} # Error: {td_error:.4f} # Epsilon: {self.epsilon:.4f}")

        return info, per_step_metrics, per_episode_metrics

    def evaluate_agent(self, env:TradingEnv, seed=None):
        """
        Valuta le prestazioni dell'agente sull'ambiente.
        Args:
            env (TradingEnv): Ambiente di trading.
            seed (int, opzionale): Seed per la riproducibilità. Defaults to None.
        Returns:
            tuple: Contiene il profitto totale, la ricompensa totale e altre informazioni.
        """
        if seed is not None:
            self.seed_everything(seed)
        self.epsilon = 0
        state, info = env.reset()
        max_possible_profit = env.max_possible_profit()
        prices = state[0]
        profit = state[-1]
        state = ut.state_formatter(prices)
        state = np.concatenate((state, [profit]), axis=0)
        state = self._discretize(state, info["max_price"], info["min_price"])
        done = False

        while not done:
            action = self.act(state)
            print(f"Azione: {action}")
            next_state, reward, terminated, truncated, info = env.step(action)
            next_prices = next_state[0]
            next_profit = next_state[-1]
            next_state = ut.state_formatter(next_prices)
            next_state = np.concatenate((next_state, [next_profit]), axis=0)
            next_state = self._discretize(next_state, info["max_price"], info["min_price"])

            done = terminated or truncated
            state = next_state

            if self.render_mode == 'step':
                env.render()


        if self.render_mode == 'episode':
            env.render_all()

        return {**info, 'performance': (info['total_profit']/max_possible_profit) * 100}, env.history

    def _discretize(self, state, max_price, min_price):
        """
        Discretizza lo stato continuo in uno stato discreto.
        Args:
            state (np.ndarray): Stato continuo.
            max_price (float): Prezzo massimo dell'asset.
            min_price (float): Prezzo minimo dell'asset.
        Returns:
            tuple: Stato discretizzato.
        """
        k = 5
        bins = np.linspace(min_price, max_price, k + 1)
        prices_discretized = np.digitize(state[:-1], bins, right=False)
        total_profit = state[-1]
        modulo_profit = np.mod(total_profit, k)
        if modulo_profit < 1:
            modulo_profit = 0 
        return tuple(prices_discretized.astype(np.int64))+(modulo_profit,)
    
    def set_render_mode(self, render_mode: Literal['step', 'episode', 'off']):
        """
        Imposta la modalità di rendering.
        Args:
            render_mode (Literal['step', 'episode', 'off']): Modalità di rendering.
        """
        self.render_mode = render_mode
   
    def seed_everything(self, seed):
        """
        Imposta i seed per la riproducibilità.
        Args:
            seed (int): Seed per la riproducibilità.
        """

        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)