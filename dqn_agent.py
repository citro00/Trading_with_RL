import os
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
import utils as ut
from gym_anytrading.envs import TradingEnv


class DQN(nn.Module):
    """
    Rete neurale Deep Q-Network per il trading.
    Architettura a tre layer fully connected con attivazioni ReLU.
    Args:
        n_observation (int): Numero di input (caratteristiche dello stato).
        n_actions (int): Numero di azioni possibili.
        hidden_layer_dim (int, opzionale): Dimensione del layer nascosto. Defaults to 128.
    """
    def __init__(self, n_observation, n_actions, hidden_layer_dim=128, num_hidden_layers=1):
        super(DQN, self).__init__()
        layers = []
        layers.append(nn.Linear(n_observation, hidden_layer_dim))
        for i in range(1, num_hidden_layers + 1):
            input_dim = hidden_layer_dim ##// 2**(i-1)
            output_dim = hidden_layer_dim ##// 2**i
            layers.append(nn.Linear(input_dim, output_dim))
        
        layers.append(nn.Linear(output_dim, n_actions))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Passaggio forward della rete.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output delle azioni Q-value.
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
    
class DQNAgent:
    """
    Agente Deep Q-Network per l'apprendimento nel trading.
    Gestisce l'interazione con l'ambiente, la memorizzazione delle esperienze,
    l'aggiornamento della rete neurale e l'addestramento.
    Args:
        state_size (int): Dimensione dello stato.
        action_size (int): Numero di azioni possibili.
        batch_size (int): Dimensione del batch per il replay.
        device (torch.device): Dispositivo per l'elaborazione (CPU o GPU).
        epsilon_decay (float): Fattore di decadimento dell'epsilon per l'esplorazione.
        render_mode (Literal['step', 'episode', 'off'], opzionale): Modalità di rendering. Defaults to 'off'.
    """   

    def __init__(self, 
                 state_size, 
                 action_size, 
                 batch_size, 
                 epsilon_decay, 
                 gamma, 
                 lr, 
                 device,
                 net_hidden_dim=320,
                 net_hidden_layers=2,
                 loss_fn=nn.SmoothL1Loss,
                 use_profit=True,
                 render_mode: Literal['step', 'episode', 'off']='off'):
        
        self._use_profit = use_profit
        self.state_size = state_size if self._use_profit else state_size-1
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000) 
        self.device = device
        
        # Parametri di apprendimento per la rete neurale
        self.gamma = gamma
        self.epsilon = 1.0  
        self.epsilon_min = 0.01  
        self.epsilon_decay = epsilon_decay
        self.model = DQN(self.state_size, self.action_size, net_hidden_dim, net_hidden_layers).to(self.device)
        self.target_model = DQN(self.state_size, self.action_size, net_hidden_dim, net_hidden_layers).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  
        self.target_model.eval() 

        # Definizione dell'ottimizzatore e della funzione di perdita
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)  
        self.loss_fn = loss_fn()

        # Plots
        self.render_mode = render_mode
        self._metrics_display = MetricPlots()

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
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


    def decay_epsilon(self):
        """
        Riduce l'epsilon per diminuire l'esplorazione nel tempo.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state):
        """
        Seleziona un'azione basata sulla politica epsilon-greedy.
        Args:
            state (np.ndarray): Stato corrente.
        Returns:
            int: Azione selezionata.
         """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        s = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.model(s).argmax().item()
    

    def remember(self, state, action, reward, next_state, done):
        """
        Memorizza un'esperienza nella memoria.
        Args:
            state (np.ndarray): Stato corrente.
            action (int): Azione eseguita.
            reward (float): Ricompensa ricevuta.
            next_state (np.ndarray): Stato successivo.
            done (bool): Indicatore di fine episodio.
        """
        self.memory.append((state, action, reward, next_state, done))
        

    def replay(self):
        """
        Esegue il replay delle esperienze per addestrare la rete neurale.
        Returns:
            float: Valore della perdita calcolata, oppure None se il batch non è sufficiente.
        """
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

        for episode in tqdm(range(1, episodes + 1), desc="Training Progress", unit="episode"):
            state, info = env.reset(seed=seed+episode if seed else None)
            max_possible_profit = env.max_possible_profit()

            for metric in per_step_metrics.keys():
                per_step_metrics[metric] = []

            if self.render_mode == 'step':
                env.render()
            prices = state[:-1]
            profit = state[-1]
            state = ut.state_formatter(prices)
            if self._use_profit:
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
                if self._use_profit:
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

            self.decay_epsilon()
            
            if self.render_mode == 'episode':
                env.render_all(f"Episode {episode}")

            # Salva le metriche per l'episodio corrente
            average_loss = total_loss / loss_count if loss_count > 0 else 0
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

            total_profit = info['total_profit']
            wallet_value = info['wallet_value']
            performance = (total_profit / max_possible_profit) * 100
            roi = info['roi']
            
            tqdm.write(f"Episode {episode}/{episodes} # Dataset: {info['asset']} # ROI: {roi:.2f}% # Total Profit: {total_profit:.2f}/{max_possible_profit:.2f} ({performance:.4f}) # Wallet value: {wallet_value:.2f} # Average Loss: {average_loss:.4f} # Epsilon: {self.epsilon:.4f}")

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

        self.epsilon = 0  # Disattiva esplorazione durante la valutazione
        state, info = env.reset(seed=seed if seed else None)
        max_possible_profit = env.max_possible_profit()

        prices = state[:-1]
        profit = state[-1]
        state = ut.state_formatter(prices)
        if self._use_profit:
            state = np.concatenate((state, [profit]), axis=0)
        done = False
        
        while not done:
            action = self.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_prices = next_state[:-1]
            next_profit = next_state[-1]
            next_state = ut.state_formatter(next_prices)
            if self._use_profit:
                next_state = np.concatenate((next_state, [next_profit]), axis=0)
            state = next_state

            if self.render_mode == 'step':
                env.render()

        if self.render_mode == 'episode':
            env.render_all()
        
        return {**info, 'performance': (info['total_profit']/max_possible_profit) * 100}, env.history

    def save_model(self, folder, suffix=None):
        file_name = "model.pth" if not suffix else f"model_{suffix}.pth"
        torch.save(self.model.state_dict(), f"{folder}{os.path.sep}{file_name}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage, weights_only=True))