import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import utils as ut
from gym_anytrading.envs import TradingEnv
from action import Action
import matplotlib.pyplot as plt
import matplotlib.axes

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
class Agent:
    def __init__(self, state_size, action_size, batch_size, device, initial_balance=1000):
        # Inizializza la dimensione dello stato e delle azioni
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000)  # Memoria per esperienze passate, con capacità massima di 50.000 elementi
        self.device = device
        self.initial_balance = initial_balance  # Bilancio iniziale
        # Parametri di apprendimento per la rete neurale
        self.gamma = 0.95  # Fattore di sconto per il valore delle ricompense future (0 < gamma < 1)
        self.epsilon = 1.0  # Probabilità iniziale di esplorazione (tasso di esplorazione)
        self.epsilon_min = 0.01  # Probabilità minima di esplorazione
        self.epsilon_decay = 0.9995  # Tasso di decadimento di epsilon per ridurre gradualmente l'esplorazione
        self.model = DQN(self.state_size, self.action_size, 128).to(self.device)

        # Modello target: utilizzato per la stabilità del processo di apprendimento
        self.target_model = DQN(self.state_size, self.action_size, 128).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())  # Inizializza il modello target con i pesi del modello principale
        self.target_model.eval()  # Il modello target viene usato solo per valutazione, non per addestramento

        # Definizione dell'ottimizzatore e della funzione di perdita
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)  # Ottimizzatore AdamW per aggiornare i pesi della rete
        self.loss_fn = nn.SmoothL1Loss()  # Funzione di perdita Huber Loss, utile per gestire outliers nelle ricompense 

        # Inizializzazione dei pesi della rete neurale
        # self.model.apply(self.init_weights)
        # self.target_model.apply(self.init_weights)

        # Plots
        self.plots: dict[str, matplotlib.axes.Axes] = None
        self.fig = None

    def init_plots(self):
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
        self.plots = {
            'total_profit': ax1,
            'step_profit': ax2,
            'total_reward': ax3,
            'step_reward': ax4,
            'loss': ax5
        }
        

        self.plots['total_profit'].set_title("Total Profit")
        self.plots['total_profit'].set_xlabel("Timesteps")
        self.plots['total_profit'].set_ylabel("Profit")
        
        self.plots['step_profit'].set_title("Step Profit")
        self.plots['step_profit'].set_xlabel("Timesteps")
        self.plots['step_profit'].set_ylabel("Profit")

        self.plots['total_reward'].set_title("Total Reward")
        self.plots['total_reward'].set_xlabel("Timesteps")
        self.plots['total_reward'].set_ylabel("Reward")

        self.plots['step_reward'].set_title("Step Reward")
        self.plots['step_reward'].set_xlabel("Timesteps")
        self.plots['step_reward'].set_ylabel("Reward")

        self.plots['loss'].set_title("Loss")
        self.plots['loss'].set_xlabel("Timesteps")
        self.plots['loss'].set_ylabel("Loss")

        self.fig = fig

    def plot_metrics(self, total_profits, step_profits, total_rewards, step_rewards, losses):
        if not self.plots:
            self.init_plots()

        self.plots['total_profit'].plot(total_profits)
        self.plots['step_profit'].plot(step_profits)
        self.plots['total_reward'].plot(total_rewards)
        self.plots['step_reward'].plot(step_rewards)
        self.plots['loss'].plot(losses)

        plt.show()

    def init_weights(self, m):
        """
        Inizializza i pesi della rete neurale utilizzando la strategia di He.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Inizializza i pesi in base alla strategia di He
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Inizializza i bias a 0

    
    def act(self, state):
        """Decide un'azione basata sullo stato attuale con una policy e-greedy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        s = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.model(s).argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """
        Memorizza un'esperienza nella memoria.
        """
        # Aggiungi lo stato attuale, l'azione presa, la ricompensa ricevuta, il prossimo stato e l'indicatore se è stato finale
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Addestra la rete neurale utilizzando un batch di esperienze passate.
        Restituisce il valore della loss.
        """
        # Controlla se ci sono abbastanza esperienze nella memoria per un batch completo
        if len(self.memory) < self.batch_size:
            return

        # batch = min(len(self.memory), self.batch_size)
        # Preleva un minibatch casuale dalla memoria
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Converti tutti i dati del batch in tensor di PyTorch
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Calcola i valori Q attuali per le azioni selezionate nel batch
        current_q = self.model(states).gather(1, actions).squeeze()

        # Calcola i valori Q futuri utilizzando il modello target
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1).values

        # Calcola i target Q: reward attuale + gamma * valore futuro (se non terminale)
        target_q = rewards + (self.gamma * max_next_q * (~dones))

        # Calcola la perdita (Huber Loss tra i valori Q attuali e quelli target)
        loss = self.loss_fn(current_q, target_q)

        # Backpropagation per aggiornare i pesi della rete
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # Riduci il tasso di esplorazione (epsilon) per favorire l'uso delle azioni apprese
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def train_agent(self, env:TradingEnv, episodes ):
        """
        Addestra l'agente interagendo con l'ambiente.
        """
        roi_history = []
        print(f"Inizio addestramento per {episodes} episodi.")
        for episode in range(1, episodes + 1):
            # Resetta l'ambiente all'inizio di ogni episodio
            state, info = env.reset()
            # env.render()
            state = ut.state_formatter(state)
            done = False
            loss_history = [] #TODO: loss per timestep vs loss per episodio
         
            # Ciclo fino a che l'episodio non termina
            while not done:
                action = self.act(state)  # L'agente decide un'azione
                # Esegui l'azione nell'ambiente
                next_state, reward, terminated, truncated, info = env.step(action)
                #print(f"Epsilon: {self.epsilon}")
                #print(f"Episodio: {episode}")
                done = terminated or truncated
                next_state = ut.state_formatter(next_state)

                # Salva l'esperienza nella memoria
                self.remember(state, action, reward, next_state, done)
                state = next_state

                # Addestra la rete con l'esperienza memorizzata
                loss = self.replay()
                if loss is not None:
                    # total_loss += loss
                    # loss_count += 1
                    loss_history.append(loss)
                #print(f"Loss: {loss}")
                # env.render()
            # Aggiorna il modello target ogni  5 episodi per stabilizzare l'apprendimento
            if   episode % 5 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            # Calcola e stampa la perdita media dell'episodio
            # average_loss = total_loss / loss_count if loss_count > 0 else 0
            average_loss = np.sum(loss_history) / len(loss_history) if len(loss_history) else 0
            
            
            total_profit = info.get('total_profit', 0)
            average_roi = (total_profit / self.initial_balance) * 100
            
            print(f"Episode {episode}/{episodes} #  ROI: {average_roi:.2f}% # Total Profit: {total_profit:.2f} # Average Loss: {average_loss:.4f} # Epsilon: {self.epsilon:.4f}")

            history = env.history
            if episode==episodes:
                self.plot_metrics(history['total_profit'], history['step_profit'], history['total_reward'], history['step_reward'], loss_history)

        print("Addestramento completato.")

    def evaluate_agent(self, env:TradingEnv): #todo implemetn
        """
        Valuta l'agente eseguendo un episodio di trading.
        """
        self.epsilon = 0  # Disattiva esplorazione durante la valutazione
        state, info = env.reset()
        state = ut.state_formatter(state)
        done = False
        states_buy = []
        states_sell = []
        total_profit = 0
        total_reward = 0

        # Ciclo fino a che l'episodio di valutazione non termina
        while not done:
            action = self.act(state)  # L'agente decide un'azione
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = ut.state_formatter(next_state)

            # Salva i tick di acquisto o vendita
            if action == Action.Buy.value and env.get_done_deal():
                states_buy.append(env.get_current_tick())
            elif action == Action.Sell.value and env.get_done_deal():
                states_sell.append(env.get_current_tick())

            state = next_state
            #print(f"Step reward: {reward}")
            total_reward += reward
        total_profit = env.get_total_profit()
        print(f"Total profit: {total_profit}\nTotal reward: {total_reward}")

        # Stampa il profitto totale ottenuto durante la valutazione
        print(f"Valutazione - Total Profit: {total_profit:.2f}")
        return states_buy, states_sell, total_profit, total_reward
