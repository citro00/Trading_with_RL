import random
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from plots import MetricPlots
import utils as ut
from gym_anytrading.envs import TradingEnv




class DQN(nn.Module):
    
    """
    DQN è una rete neurale profonda utilizzata dall'agente per stimare i valori Q delle azioni 
    in uno stato dato.
    :param n_observation: Numero di feature nello stato di input.
    :param n_actions: Numero di azioni possibili.
    :param hidden_layer_dim: Dimensione del layer nascosto.
    """
    
    def __init__(self, n_observation, n_actions, hidden_layer_dim=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observation, hidden_layer_dim)
        self.layer2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.layer3 = nn.Linear(hidden_layer_dim, n_actions)

    def forward(self, x):
        """
        Esegue un passaggio in avanti attraverso la rete neurale, applicando ReLU alle unità nascoste 
        e producendo un output con i valori Q.
        :param x: Stato di input.
        :return: Valori Q per tutte le azioni possibili.
        """
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    
class DQNAgent:
    
    """
    DQNAgent è un agente basato su Deep Q-Learning (DQN) progettato per il trading in ambienti finanziari. 
    Utilizza una rete neurale profonda per selezionare azioni ottimali in base allo stato dell'ambiente. 
    Supporta l'apprendimento tramite replay buffer, l'aggiornamento del modello target per stabilità e 
    l'addestramento/valutazione su ambienti compatibili.
    """


    def __init__(self, state_size, action_size, batch_size, device, initial_balance=1000, render_mode: Literal['step', 'episode', 'off']='off'):
        
        """
        Inizializza l'agente DQN con parametri come dimensione dello stato, azioni, memoria di replay, 
        e reti neurali. Configura il processo di apprendimento e i parametri di esplorazione.
        :param state_size: Numero di feature nello stato di input.
        :param action_size: Numero di azioni possibili.
        :param batch_size: Dimensione del batch per il replay buffer.
        :param device: Dispositivo su cui eseguire i calcoli (CPU o GPU).
        :param initial_balance: Bilancio iniziale per il trading.
        :param render_mode: Modalità di rendering ('step', 'episode', 'off').
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000) 
        self.device = device
        self.initial_balance = initial_balance 
        
        # Parametri di apprendimento per la rete neurale
        self.gamma = 0.95 
        self.epsilon = 1.0  
        self.epsilon_min = 0.01  
        self.epsilon_decay = 0.9999
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
        """
        Configura la modalità di rendering per l'agente.
        :param render_mode: Modalità di rendering ('step', 'episode', 'off').
        """

        self.render_mode = render_mode

    def init_weights(self, m):
        """
        Inizializza i pesi delle reti neurali con la strategia di He e i bias a zero.
        :param m: Modulo della rete (layer).
        """
        
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  
  
    def act(self, state):
        """
        Seleziona un'azione basandosi su esplorazione casuale (ε-greedy) o sfruttamento del modello DQN.
        :param state: Stato corrente.
        :return: Azione selezionata (indice).
        """

        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        s = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.model(s).argmax().item()
    

    def remember(self, state, action, reward, next_state, done):
        """
        Memorizza una transizione nello storico delle esperienze.
        :param state: Stato corrente.
        :param action: Azione eseguita.
        :param reward: Ricompensa ricevuta.
        :param next_state: Stato successivo.
        :param done: Flag che indica se l'episodio è terminato.
        """
        self.memory.append((state, action, reward, next_state, done))
        

    def replay(self):
        """
        Addestra il modello prelevando un batch di esperienze dalla memoria e aggiornando i pesi 
        utilizzando il target Q e la perdita Huber.
        :return: Valore della perdita media per il batch.
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

        # Riduci il tasso di esplorazione (epsilon) per favorire l'uso delle azioni apprese
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def train_agent(self, env:TradingEnv, episodes ):
        """
        Addestra l'agente su un ambiente di trading per un numero specificato di episodi. Aggiorna 
        il modello target periodicamente e traccia le metriche di performance.
        :param env: Ambiente di trading.
        :param episodes: Numero di episodi di addestramento.
        """

        per_step_metrics = {
            'step_reward': [],
            'step_profit': [],
        }
        
        per_episode_metrics = {
            'roi': [],
            'total_reward': [],
            'total_profit': [],
            'wallet_value': [],
        }

        print(f"Inizio addestramento per {episodes} episodi.")
        for episode in range(1, episodes + 1):
            state, info = env.reset()
            for metric in per_step_metrics.keys():
                per_step_metrics[metric] = []

            if self.render_mode == 'step':
                env.render()
            state = ut.state_formatter(state)
            done = False
         
            while not done:
                total_loss, loss_count = 0, 0.
                action = self.act(state) 
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = ut.state_formatter(next_state)
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

            print(f"Episode {episode}/{episodes} # Dataset: {env._current_asset} # ROI: {roi:.2f}% # Total Profit: {total_profit:.2f} # Wallet value: {wallet_value:.2f} # Average Loss: {average_loss:.4f} # Loss: {loss} # Epsilon: {self.epsilon:.4f}")

        if self.render_mode == 'off':
            self._metrics_display.plot_metrics(**per_step_metrics)
            self._metrics_display.plot_metrics(**per_episode_metrics, show=True)
        print("Addestramento completato.")

    def evaluate_agent(self, env:TradingEnv):
        """
        Valuta l'agente eseguendo un episodio di trading senza esplorazione (ε=0). Mostra i risultati 
        finali e opzionalmente visualizza il comportamento dell'agente.
        :param env: Ambiente di trading.
        :return: Profitto totale, ricompensa totale e informazioni aggiuntive.
        """

        self.epsilon = 0  # Disattiva esplorazione durante la valutazione
        state, info = env.reset()
        state = ut.state_formatter(state)
        done = False
        total_profit = 0
        total_reward = 0

        while not done:
            action = self.act(state) 
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = ut.state_formatter(next_state)
            state = next_state
            total_reward += reward

            if self.render_mode == 'step':
                env.render()

        print(f"Total profit: {info['total_profit']}\nTotal reward: {total_reward}")
        print(f"Valutazione - Total Profit: {info['total_profit']:.2f}")

        if self.render_mode == 'episode':
            env.render_all()
        
        return info['total_profit'], total_reward, info
