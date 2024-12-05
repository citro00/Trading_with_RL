import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import utils as ut
from gym_anytrading.envs import TradingEnv
from action import Action
from position import Position

class Agent:
    def __init__(self, state_size, action_size, batch_size, device):
        # Inizializza la dimensione dello stato e delle azioni
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000)  # Memoria per esperienze passate, con capacità massima di 50.000 elementi
        self.device = device

        # Parametri di apprendimento per la rete neurale
        self.gamma = 0.95  # Fattore di sconto per il valore delle ricompense future (0 < gamma < 1)
        self.epsilon = 1.0  # Probabilità iniziale di esplorazione (tasso di esplorazione)
        self.epsilon_min = 0.01  # Probabilità minima di esplorazione
        self.epsilon_decay = 0.999  # Tasso di decadimento di epsilon per ridurre gradualmente l'esplorazione

        # Definizione del modello di rete neurale (Q-Network) che rappresenta la policy dell'agente
        self.model = nn.Sequential(
            nn.Linear(self.state_size, 256),  # Layer denso che mappa dallo stato a 256 neuroni
            nn.ReLU(),  # Funzione di attivazione non lineare ReLU
            nn.Linear(256, self.action_size)  # Layer finale che restituisce i valori Q per ogni azione
        ).to(self.device)

        # Modello target: utilizzato per la stabilità del processo di apprendimento
        self.target_model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size)
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # Inizializza il modello target con i pesi del modello principale
        self.target_model.eval()  # Il modello target viene usato solo per valutazione, non per addestramento

        # Definizione dell'ottimizzatore e della funzione di perdita
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)  # Ottimizzatore AdamW per aggiornare i pesi della rete
        self.loss_fn = nn.SmoothL1Loss()  # Funzione di perdita Huber Loss, utile per gestire outliers nelle ricompense 

        # Inizializzazione dei pesi della rete neurale
        self.model.apply(self.init_weights)
        self.target_model.apply(self.init_weights)

    def init_weights(self, m):
        """
        Inizializza i pesi della rete neurale utilizzando la strategia di He.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Inizializza i pesi in base alla strategia di He
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Inizializza i bias a 0

    '''def act(self, state):
        """
        Decide un'azione basata sullo stato attuale.
        """
        # Converti lo stato in un tensor di PyTorch
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)  # Ottieni i valori Q per tutte le azioni
        # Usa softmax per generare una distribuzione di probabilità tra le azioni
        probabilities = torch.softmax(q_values, dim=0).cpu().numpy()
        return np.random.choice(self.action_size, p=probabilities)  # Seleziona un'azione in base alla distribuzione'''
    
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
        #if len(self.memory) < self.batch_size:
            #return None

        batch = min(len(self.memory), self.batch_size)
        # Preleva un minibatch casuale dalla memoria
        minibatch = random.sample(self.memory, batch)
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
            max_next_q = self.target_model(next_states).max(1)[0]

        # Calcola i target Q: reward attuale + gamma * valore futuro (se non terminale)
        target_q = rewards + (self.gamma * max_next_q * (~dones))

        # Calcola la perdita (Huber Loss tra i valori Q attuali e quelli target)
        loss = self.loss_fn(current_q, target_q)

        # Backpropagation per aggiornare i pesi della rete
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Aggiorna il modello target ogni 10 batch per stabilizzare l'apprendimento
        if len(self.memory) % 10 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Riduci il tasso di esplorazione (epsilon) per favorire l'uso delle azioni apprese
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def train_agent(self, env:TradingEnv, episodes):
        """
        Addestra l'agente interagendo con l'ambiente.
        """
        print(f"Inizio addestramento per {episodes} episodi.")
        for episode in range(1, episodes + 1):
            # Resetta l'ambiente all'inizio di ogni episodio
            state, info = env.reset()
            state = ut.state_formatter(state)
            done = False
            total_loss = 0
            loss_count = 0

            # Ciclo fino a che l'episodio non termina
            while not done:
                action = self.act(state)  # L'agente decide un'azione
                # Esegui l'azione nell'ambiente
                next_state, reward, terminated, truncated, info = env.step(action)
                print(f"Epsilon: {self.epsilon}")
                done = terminated or truncated
                next_state = ut.state_formatter(next_state)

                # Salva l'esperienza nella memoria
                self.remember(state, action, reward, next_state, done)
                state = next_state

                # Addestra la rete con l'esperienza memorizzata
                loss = self.replay()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

            # Calcola e stampa la perdita media dell'episodio
            average_loss = total_loss / loss_count if loss_count > 0 else 0
            print(f"Episode {episode}/{episodes} - Total Profit: {env._total_profit:.2f} - Average Loss: {average_loss:.4f} - Loss: {loss} - Epsilon: {self.epsilon:.4f}")

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
            total_profit += reward

        # Stampa il profitto totale ottenuto durante la valutazione
        print(f"Valutazione - Total Profit: {total_profit:.2f}")
        return states_buy, states_sell, total_profit
