from typing import Literal
from collections import defaultdict
import numpy as np
import random
from gym_anytrading.envs import TradingEnv
from tqdm import tqdm
import utils as ut
from plots import MetricPlots
import pandas as pd


class QLAgent:
    
    def __init__(self, action_size, render_mode: Literal['step', 'episode', 'off']='off'):
        
        self.action_size = action_size
        self.render_mode = render_mode

        self.gamma = 0.95  
        self.epsilon = 1.0 
        self.epsilon_min = 0.01  
        self.epsilon_decay = 0.991
        self.learning_rate = 0.001
        self.q_values = defaultdict(lambda: np.zeros(self.action_size))
        self._metrics_display = MetricPlots()

    def act(self, obs):
        
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)

        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
    
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (reward + self.gamma * future_q_value - self.q_values[obs][action])
        self.q_values[obs][action] = (self.q_values[obs][action] + self.learning_rate * temporal_difference)
        
        return temporal_difference
    
    def decay_epsilon(self):
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_agent(self, env:TradingEnv, episodes, seed=False):
        
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

            self.decay_epsilon()

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
        history.to_csv("csv/history.csv", index=False)  
        
        print("___ Valutazione ___")
        print(f"Total Profit: {info['total_profit']:.2f} - Mean: {np.mean(history['total_profit']):.2f} - Std: {np.std(history['total_profit']):.2f}")
        print(f"Wallet value: {info['wallet_value']:.2f} - Mean: {np.mean(history['wallet_value']):.2f} - Std: {np.std(history['wallet_value']):.2f}")
        print(f"Total Reward: {info['total_reward']:.2f} - Mean: {np.mean(history['total_reward']):.2f} - Std: {np.std(history['total_reward']):.2f}")
        print(f"ROI: {info['roi']:.2f}% - Mean: {np.mean(history['roi']):.2f}% - Std: {np.std(history['roi']):.2f}%")


        if self.render_mode == 'episode':
            env.render_all()

        return info['total_profit'], info['total_reward'], info

    def _discretize(self, state, max_price, min_price):
        
        k = 5
        bins = np.linspace(min_price, max_price, k + 1)
        prices_discretized = np.digitize(state[:-1], bins, right=False)
        total_profit = state[-1]
        modulo_profit = np.mod(total_profit, k)
        if modulo_profit < 1:
            modulo_profit = 0 
        return tuple(prices_discretized.astype(np.int64))+(modulo_profit,)
    
    def set_render_mode(self, render_mode: Literal['step', 'episode', 'off']):
        self.render_mode = render_mode
   