from typing import Literal
from collections import defaultdict
import numpy as np
import random
from gym_anytrading.envs import TradingEnv
import matplotlib.pyplot as plt
import matplotlib.axes
from tqdm import tqdm
import utils as ut
from plots import MetricPlots
import pandas as pd
import time
class QLAgent:
    
    """
    QLAgent è un agente di apprendimento per il trading basato su Q-Learning tabulare. 
    Utilizza una tabella Q per stimare il valore delle azioni in base agli stati discreti. 
    L'agente supporta l'apprendimento tramite aggiornamenti incrementali dei valori Q e l'addestramento 
    in ambienti di trading simulati.
    """
    
    def __init__(self, action_size, initial_balance=1000, render_mode: Literal['step', 'episode', 'off']='off'):
        
        """
        Inizializza l'agente con i parametri principali per il Q-Learning, inclusi la tabella Q,    
        i parametri di esplorazione e apprendimento, e la configurazione per il rendering.
        :param action_size: Numero di azioni disponibili.
        :param initial_balance: Bilancio iniziale per il trading.
        :param render_mode: Modalità di rendering ('step', 'episode', 'off').
        """
        self.action_size = action_size
        self._initial_balance = initial_balance
        self.render_mode = render_mode

        self.gamma = 0.95  
        self.epsilon = 1.0 
        self.epsilon_min = 0.01  
        self.epsilon_decay = 0.991
        self.learning_rate = 0.001

        self.q_values = defaultdict(lambda: np.zeros(self.action_size))
        self._metrics_display = MetricPlots()

    def act(self, obs):
        
        """
        Seleziona un'azione basandosi sull'esplorazione casuale (ε-greedy) o sfruttando la tabella Q.
        :param obs: Stato corrente osservato.
        :return: Azione selezionata (indice).
        """
        
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)

        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        
        """
        Aggiorna la tabella Q utilizzando l'equazione di aggiornamento del Q-Learning. 
        Calcola la differenza temporale per migliorare la stima dei valori Q.
        :param obs: Stato corrente.
        :param action: Azione eseguita.
        :param reward: Ricompensa ricevuta.
        :param terminated: Flag che indica se l'episodio è terminato.
        :param next_obs: Stato successivo osservato.
        """
        
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (reward + self.gamma * future_q_value - self.q_values[obs][action])

        self.q_values[obs][action] = (self.q_values[obs][action] + self.learning_rate * temporal_difference)
        # self.training_error.append(temporal_difference)
        return temporal_difference
    
    def dacay_epsilon(self):
        
        """
        Riduce gradualmente il valore di epsilon per diminuire l'esplorazione e favorire lo sfruttamento 
        della conoscenza acquisita.
        """

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_agent(self, env:TradingEnv, episodes, seed=False):
        
        """
        Addestra l'agente interagendo con l'ambiente di trading per un numero specificato di episodi. 
        Registra le metriche di performance e aggiorna la tabella Q in base alle esperienze.
        :param env: Ambiente di trading.
        :param episodes: Numero di episodi di addestramento.
        """

        per_step_metrics = {
            'step_reward': [],
            'delta_p': [],
        }
        
        per_episode_metrics = {
            'roi': [],
            'total_reward': [],
            'total_profit': [],
            'wallet_value': []
        }

        print(f"Inizio addestramento per {episodes} episodi.")
        for episode in tqdm(range(1, episodes + 1), desc="Training Progress", unit="episode"):
            state, info = env.reset(seed=episode if seed else None)
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
                action = self.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_prices = next_state[0]
                next_profit = next_state[-1]
                next_state = ut.state_formatter(next_prices)
                next_state = np.concatenate((next_state, [next_profit]), axis=0)
                next_state = self._discretize(next_state, info["max_price"], info["min_price"])
                td_error = self.update(state, action, reward, terminated, next_state)
                done = terminated or truncated
                state = next_state
                if self.render_mode == 'step':
                    env.render()
                for metric, arr in per_step_metrics.items():
                    arr.append(info[metric])

                if self.render_mode == 'step':
                    self._metrics_display.plot_metrics(**per_step_metrics)

            self.dacay_epsilon()

            if self.render_mode == 'episode':
                env.render_all(f"Episode {episode}")

            for metric in per_episode_metrics.keys():
                per_episode_metrics[metric].append(info[metric])
            if self.render_mode == 'episode':
                self._metrics_display.plot_metrics(**per_step_metrics)
                self._metrics_display.plot_metrics(**per_episode_metrics)

            if self.render_mode == 'episode':
                pass
            total_profit = info['total_profit']
            wallet_value = info['wallet_value']
            roi = info['roi']

            tqdm.write(f"Episode {episode}/{episodes} # Dataset: {info['asset']} # ROI: {roi:.2f}% # Total Profit: {total_profit:.2f} # Wallet value: {wallet_value:.2f} # Error: {td_error:.4f} # Epsilon: {self.epsilon:.4f}")

        if self.render_mode == 'off':
            self._metrics_display.plot_metrics(**per_step_metrics)
            self._metrics_display.plot_metrics(**per_episode_metrics, show=True)
            pass

        print("Addestramento completato.")

    def evaluate_agent(self, env: TradingEnv):
        
        """
        Valuta l'agente eseguendo un episodio di trading con epsilon impostato a 0 per disabilitare l'esplorazione. 
        Registra e restituisce i risultati finali.
        :param env: Ambiente di trading.
        :return: Profitto totale, ricompensa totale e informazioni aggiuntive.
        """

        self.epsilon = 0
        state, info = env.reset()
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

        history = pd.DataFrame(env.history)
        history.to_csv("csv/history.csv", index=False)  # Salva
        
        print("___ Valutazione ___")
        print(f"Total Profit: {info['total_profit']:.2f} - Mean: {np.mean(history['total_profit']):.2f} - Std: {np.std(history['total_profit']):.2f}")
        print(f"Wallet value: {info['wallet_value']:.2f} - Mean: {np.mean(history['wallet_value']):.2f} - Std: {np.std(history['wallet_value']):.2f}")
        print(f"Total Reward: {info['total_reward']:.2f} - Mean: {np.mean(history['total_reward']):.2f} - Std: {np.std(history['total_reward']):.2f}")
        print(f"ROI: {info['roi']:.2f}% - Mean: {np.mean(history['roi']):.2f}% - Std: {np.std(history['roi']):.2f}%")


        if self.render_mode == 'episode':
            env.render_all()

        return info['total_profit'], info['total_reward'], info

    def _discretize(self, state, max_price, min_price):
        
        """
        Trasforma uno stato continuo in uno discreto utilizzando bin definibili sull'intervallo 
        dei prezzi minimi e massimi osservati.
        :param state: Stato continuo da discretizzare.
        :param max_price: Prezzo massimo osservato nell'ambiente.
        :param min_price: Prezzo minimo osservato nell'ambiente.
        :return: Stato discretizzato.
        """

        k = 5
        bins = np.linspace(min_price, max_price, k + 1)
        # print(f"Bins: {bins}")
        # print(f"State: {state[:10]}")
        prices_discretized = np.digitize(state[:-1], bins, right=False)
        total_profit = state[-1]
        modulo_profit = np.mod(total_profit, k)
        if modulo_profit < 1:
            modulo_profit = 0 
        # print("\nDiscretizzazione in 12 bin (prime 10 posizioni):\n", discretized[:10])
        return tuple(prices_discretized.astype(np.int64))+(modulo_profit,)

    def set_render_mode(self, render_mode: Literal['step', 'episode', 'off']):
        
        """
        Configura la modalità di rendering per l'agente.
        :param render_mode: Modalità di rendering ('step', 'episode', 'off').
        """
        
        self.render_mode = render_mode

    def init_plots(self):
        fig = plt.figure(1000, figsize=(15, 5),  layout="constrained")
        self.plots = fig.subplot_mosaic(
            [
                ["total_profit", "total_reward", "loss"],
                ["step_profit", "step_reward", "wallet_value"]
            ]
        )
        
        self._set_plot_labels()
        plt.ion()

    def plot_metrics(self, **kwargs):
        if not self.plots:
            self.init_plots()
        
        for metric, value in kwargs.items():
            self.plots[metric].clear()
            if metric == 'step_reward':
                self.plots[metric].scatter(range(len(value)), value, s=2**2)
            else:
                self.plots[metric].plot(value)

        self._set_plot_labels()

        plt.draw()
        plt.pause(0.01)