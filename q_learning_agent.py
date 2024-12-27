from typing import Literal
from collections import defaultdict
import numpy as np
import random
from gym_anytrading.envs import TradingEnv
import matplotlib.pyplot as plt
import matplotlib.axes
import utils as ut
class QLAgent:
    def __init__(self, action_size, initial_balance=1000, render_mode: Literal['step', 'episode', 'off']='off'):
        self.action_size = action_size
        self._initial_balance = initial_balance
        self.render_mode = render_mode

        self.gamma = 0.95  # Fattore di sconto per il valore delle ricompense future (0 < gamma < 1)
        self.epsilon = 1.0  # Probabilità iniziale di esplorazione (tasso di esplorazione)
        self.epsilon_min = 0.01  # Probabilità minima di esplorazione
        self.epsilon_decay = 0.9999  # Tasso di decadimento di epsilon per ridurre gradualmente l'esplorazione
        self.learning_rate = 0.01

        self.q_values = defaultdict(lambda: np.zeros(self.action_size))
        self.training_error = []

    def act(self, obs):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (reward + self.gamma * future_q_value - self.q_values[obs][action])

        self.q_values[obs][action] = (self.q_values[obs][action] + self.learning_rate * temporal_difference)
        self.training_error.append(temporal_difference)
    
    def dacay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def train_agent(self, env:TradingEnv, episodes):
        """
        Addestra l'agente interagendo con l'ambiente.
        """
        per_step_metrics = {
            'step_reward': [],
            'step_profit': [],
            #'actual_budget': []
        }
        
        per_episode_metrics = {
            'loss': [],
            'total_reward': [],
            'total_profit': [],
            'wallet_value': []
        }

        print(f"Inizio addestramento per {episodes} episodi.")
        for episode in range(1, episodes + 1):
            state, info = env.reset()
            state = state[:,0]
            state = ut.state_formatter(state)
            state = self._discretize(state, info["max_price"], info["min_price"])

            for metric in per_step_metrics.keys():
                per_step_metrics[metric] = []

            if self.render_mode == 'step':
                env.render()

            done = False

            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = next_state[:,0]
                next_state = ut.state_formatter(next_state)
                next_state = self._discretize(next_state, info["max_price"], info["min_price"])
                self.update(state, action, reward, terminated, next_state)
                done = terminated or truncated
                state = next_state
                # FINE DI UN TIMESTEP
                if self.render_mode == 'step':
                    env.render()

                # Salva le metriche per timestep
                for metric, arr in per_step_metrics.items():
                    arr.append(info[metric])

                if self.render_mode == 'step':
                    self.plot_metrics(**per_step_metrics)

            if self.render_mode == 'episode':
                env.render_all(f"Episode {episode}")

            for metric in ['total_profit', 'total_reward']:
                per_episode_metrics[metric].append(info[metric])
            if self.render_mode == 'episode':
                #self.plot_metrics(**per_step_metrics)
                #self.plot_metrics(**per_episode_metrics)
                pass

            #total_profit = info.get('total_profit', 0)
            total_profit = info['total_profit']
            wallet_value = info['wallet_value']
            average_roi = (total_profit / self._initial_balance) * 100

            print(f"Episode {episode}/{episodes} # Dataset: {info['asset']} # ROI: {average_roi:.2f}% # Total Profit: {total_profit:.2f} # Wallet value: {wallet_value:.2f} # Epsilon: {self.epsilon:.4f}")

        if self.render_mode == 'off':
            #self.plot_metrics(**per_step_metrics)
            #self.plot_metrics(**per_episode_metrics)
            #plt.show(block=True)
            pass

        print("Addestramento completato.")

    def evaluate_agent(self, env: TradingEnv):
        self.epsilon = 0
        state, info = env.reset()
        state = state[:,0]

        state = ut.state_formatter(state)
        state = self._discretize(state, info["max_price"], info["min_price"])
        done = False
        total_profit = 0
        total_reward = 0

        while not done:
            action = self.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state[:,0]
            next_state = ut.state_formatter(next_state)
            next_state = self._discretize(next_state, info["max_price"], info["min_price"])

            done = terminated or truncated
            state = next_state
            total_reward += reward

            if self.render_mode == 'step':
                env.render()

        print(f"Total profit: {info['total_profit']}\nTotal reward: {total_reward}")
        # Stampa il profitto totale ottenuto durante la valutazione
        print(f"Valutazione - Total Profit: {info['total_profit']:.2f}")
 
        if self.render_mode == 'episode':
            env.render_all()

        return info['total_profit'], total_reward, info

    def _discretize(self, state, max_price, min_price):
        k = 25
        bins = np.linspace(min_price, max_price, k + 1)
        print(f"Bins: {bins}")
        print(f"State: {state[:10]}")
        discretized = np.digitize(state, bins, right=False)

        print("\nDiscretizzazione in 30 bin (prime 10 posizioni):\n", discretized[:10])

    def set_render_mode(self, render_mode: Literal['step', 'episode', 'off']):
        self.render_mode = render_mode

    def _set_plot_labels(self):
        self.plots['total_profit'].set_title("Total Profit")
        self.plots['total_profit'].set_xlabel("Episode")
        self.plots['total_profit'].set_ylabel("Profit")
        
        self.plots['step_profit'].set_title("Step Profit")
        self.plots['step_profit'].set_xlabel("Timesteps")
        self.plots['step_profit'].set_ylabel("Profit")

        self.plots['total_reward'].set_title("Total Reward")
        self.plots['total_reward'].set_xlabel("Episode")
        self.plots['total_reward'].set_ylabel("Reward")

        self.plots['step_reward'].set_title("Step Reward")
        self.plots['step_reward'].set_xlabel("Timesteps")
        self.plots['step_reward'].set_ylabel("Reward")

        self.plots['loss'].set_title("Loss")
        self.plots['loss'].set_xlabel("Episode")
        self.plots['loss'].set_ylabel("Loss")

        self.plots['wallet_value'].set_title("Wallet Value")
        self.plots['wallet_value'].set_xlabel("Episode")
        self.plots['wallet_value'].set_ylabel("Value")

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

        # plt.ioff()
        plt.draw()
        plt.pause(0.01)