import random
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from tqdm import tqdm
from plots import MetricPlots
import matplotlib.pyplot as plt
import utils as ut
from gym_anytrading.envs import TradingEnv
import pandas as pd


class DQN(nn.Module):
    
    def __init__(self, n_observation, n_actions, hidden_layer_dim=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observation, hidden_layer_dim)
        self.layer2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.layer3 = nn.Linear(hidden_layer_dim, n_actions)

    def forward(self, x):
    
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    
class DQNAgent:


    def __init__(self, state_size, action_size, batch_size, device, epsilon_decay, render_mode: Literal['step', 'episode', 'off']='off'):
        
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000) 
        self.device = device
        
        # Parametri di apprendimento per la rete neurale
        self.gamma = 0.95
        self.epsilon = 1.0  
        self.epsilon_min = 0.01  
        self.epsilon_decay = epsilon_decay
        self.model = DQN(self.state_size, self.action_size, 128).to(self.device)

        self.target_model = DQN(self.state_size, self.action_size, 128).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  
        self.target_model.eval() 

        # Definizione dell'ottimizzatore e della funzione di perdita
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)  
        self.loss_fn = nn.SmoothL1Loss()  

        # Plots
        self._metrics_display = MetricPlots()
        self.render_mode = render_mode

    def set_render_mode(self, render_mode: Literal['step', 'episode', 'off']):
        
        self.render_mode = render_mode

  
    def dacay_epsilon(self):
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state):

        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        s = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.model(s).argmax().item()
    

    def remember(self, state, action, reward, next_state, done):
        
        self.memory.append((state, action, reward, next_state, done))
        

    def replay(self):
        

        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Converti tutti i dati del batch in tensor di PyTorch
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        current_q = self.model(states).gather(1, actions).squeeze()

        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1).values

        target_q = rewards + (self.gamma * max_next_q * (~dones))
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_agent(self, env:TradingEnv, episodes, seed=False):

        per_step_metrics = {
            'step_reward': [],
            'delta_p': [],
        }
        
        per_episode_metrics = {
            'roi': [],
            'total_reward': [],
            'total_profit': [],
            'wallet_value': [],
        }

        print(f"Inizio addestramento per {episodes} episodi.")
        for episode in tqdm(range(1, episodes + 1), desc="Training Progress", unit="episode"):
            state, info = env.reset(seed=episode if seed else None)
            for metric in per_step_metrics.keys():
                per_step_metrics[metric] = []

            if self.render_mode == 'step':
                env.render()
            prices = state[:-1]
            profit = state[-1]
            state = ut.state_formatter(prices)
            state = np.concatenate((state, [profit]), axis=0)
            done = False
         
            while not done:
                total_loss, loss_count = 0, 0.
                action = self.act(state) 
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_prices = next_state[:-1]
                next_profit = next_state[-1]
                next_state = ut.state_formatter(next_prices)
                next_state = np.concatenate((next_state, [next_profit]), axis=0)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                loss = self.replay()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
                if self.render_mode == 'step':
                    env.render()
                for metric, arr in per_step_metrics.items():
                    arr.append(info[metric])
                if self.render_mode == 'step':
                    self._metrics_display.plot_metrics(**per_step_metrics)

            # Aggiorna il modello target ogni  5 episodi per stabilizzare l'apprendimento
            if episode % 5 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.dacay_epsilon()
            
            if self.render_mode == 'episode':
                env.render_all(f"Episode {episode}")

            average_loss = total_loss / loss_count if loss_count > 0 else 0
            for metric in per_episode_metrics.keys():
                per_episode_metrics[metric].append(info[metric])
            if self.render_mode == 'episode':
                self._metrics_display.plot_metrics(**per_step_metrics)
                self._metrics_display.plot_metrics(**per_episode_metrics)

            total_profit = info['total_profit']
            wallet_value = info['wallet_value']
            roi = info['roi']
            
            tqdm.write(f"Episode {episode}/{episodes} # Dataset: {info['asset']} # ROI: {roi:.2f}% # Total Profit: {total_profit:.2f} # Wallet value: {wallet_value:.2f} # Average Loss: {average_loss:.4f} # Epsilon: {self.epsilon:.4f}")

        if self.render_mode == 'off':
            self._metrics_display.plot_metrics(**per_step_metrics)
            self._metrics_display.plot_metrics(**per_episode_metrics, show=True)
        print("Addestramento completato.")

    def evaluate_agent(self, env:TradingEnv):

        self.epsilon = 0  # Disattiva esplorazione durante la valutazione
        state, info = env.reset()
        prices = state[:-1]
        profit = state[-1]
        state = ut.state_formatter(prices)
        state = np.concatenate((state, [profit]), axis=0)
        done = False
        
        while not done:
            action = self.act(state)
            print(f"Action: {action}")
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_prices = next_state[:-1]
            next_profit = next_state[-1]
            next_state = ut.state_formatter(next_prices)
            next_state = np.concatenate((next_state, [next_profit]), axis=0)
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
