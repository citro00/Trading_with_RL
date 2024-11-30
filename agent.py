import utils as ut
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
#  train: 
class Agent:
    def __init__(self, state_size, window_size, trend, skip, batch_size, device):
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        self.action_size = 3
        self.batch_size = batch_size
        self.memory = []
        self.inventory = []
        self.device = device

        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        self.model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size)
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        return self.model(state).argmax().item()

    #Funzione da sostituire con quella dell'environment in quanto lo stato deve essere fornito dall'ambiente
    def get_state(self, t):
        '''t è un indice temporale
        window_size è la dimensione dei punti dati da considerare per costruire lo stato
        si incrementa di uno in modo da includere gli estremi'''
        window_size = self.window_size + 1
        #d rappresenta l'indice di trend da cui iniziare a prendere i dati per costruire lo stato
        d = t - window_size + 1
        '''per selezionare il blocco si verifica prima se d è maggiore di 0. Se non lo è siamo troppo vicini all'inizio dei dati
        se d < 0 il blocco viene riempito con dati fittizzi uguali al primo valore della finestra mobile'''
        block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
        res = []
        #per ogni coppia di valori nella finestra viene calcolata la differenza tra il prezzo successivo e quello precedente
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        states = np.array([a[0][0] for a in mini_batch])
        new_states = np.array([a[3][0] for a in mini_batch])
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        new_states = torch.tensor(new_states, dtype=torch.float32).to(self.device)
        
        Q = self.model(states)
        Q_new = self.model(new_states)
        
        X = states
        y = Q.clone().detach()

        for i in range(len(mini_batch)):
            state, action, reward, next_state, done = mini_batch[i]
            target = reward
            if not done:
                target += self.gamma * torch.max(Q_new[i])
            y[i][action] = target

        loss = self.loss_fn(Q, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def buy(self, initial_money):
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        state = self.get_state(0)

        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)

            if action == 1 and initial_money >= self.trend[t] and t < (len(self.trend) - self.half_window):
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f' % (t, self.trend[t], initial_money))

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += self.trend[t]
                states_sell.append(t)
                try:
                    invest = ((self.trend[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print('day %d, sell 1 unit at price %f, investment %f %%, total balance %f' % (t, self.trend[t], invest, initial_money))

            state = next_state

        shares_held = len(inventory)
        if shares_held > 0:
            current_price = self.trend[-1]
            equity_value = shares_held * current_price
            total_gains = initial_money + equity_value - starting_money
            invest = ((initial_money + equity_value - starting_money) / starting_money) * 100
        else:
            total_gains = initial_money - starting_money
            invest = ((initial_money - starting_money) / starting_money) * 100

        print(inventory)
        return states_buy, states_sell, total_gains, invest, shares_held

    def train(self, iterations, checkpoint, initial_money): #applicare sull env
        '''La funzione train prende in input il numero di iterazioni da compiere su tutto il dataset
        il parametro checkpoint per indicare ogni quante iterazioni restituire delle informazioni
        e il budget iniziale da cui partire per la compravendita di azioni.'''
        print("Start training:")
        for i in range(iterations):
            '''Per ogni iterazione resetta il profitto ottenuto, le azioni contenute nel wallet e lo stato iniziale.'''
            total_profit = 0
            inventory = []
            state = self.get_state(0)
            starting_money = initial_money

            #print(f"Start second for loop. Range from 0 to {len(self.trend)-1}, with step size {self.skip}")
            print(f"Training iteration: {i}")
            for t in range(0, len(self.trend) - 1, self.skip):
                '''Esegue un for per la lunghezza del dataset a step di uno in cui fa scegliere all'agente un'azione da compiere
                e genera lo stato successivo.'''
                #seleziona l'azione da attuare
                action = self.act(state)
                next_state = self.get_state(t + 1)

                '''Se l'azione generata è compra allora controlla che il budget sia sufficiente
                per acquistare un'azione dato il prezzo attuale nel giorno che sta processando.
                (la terza condizione non si comprende) '''
                if action == 1 and starting_money >= self.trend[t] and t < (len(self.trend) - self.half_window):
                    inventory.append(self.trend[t])
                    starting_money -= self.trend[t]
            
                elif action == 2 and len(inventory) > 0: 
                    '''Se l'azione generata è vendi controlla solamente che sia possibile effettuare la vendita
                    in base alle azioni presenti nel wallet'''
                    bought_price = inventory.pop(0)
                    total_profit += self.trend[t] - bought_price
                    starting_money += self.trend[t]
                '''Invest è la reward e viene calcolata facendo la differenza tra il budget attuale e 
                quello iniziale, diviso per il budget iniziale'''
                invest = ((starting_money - initial_money) / initial_money) # fake reward 
                #carica lo stato nella memoria della NN
                self.memory.append((state, action, invest, next_state, starting_money < initial_money))
                #Aggiorna lo sato corrente
                state = next_state
                #print(f"State: {state}")
                #print(f"Index: {t}")

                #definisce la dimensione del batch da utilizzare per la NN
                batch_size = min(self.batch_size, len(self.memory))
                #cost rappresenta la loss (la MSE tra i valori predetti e quelli reali)
                cost = self.replay(batch_size)

            if (i+1) % checkpoint == 0:
                print('epoch: %d, total rewards: %f.3, cost: %f, total money: %f' % (i + 1, total_profit, cost, starting_money))

    def new_train(self, iterations, checkpoint, budget, env: gym.Env):
        for i in range(iterations):
            total_profit = 0
            inventory = []

            # Reset dell'ambiente e gestione dell'osservazione iniziale
            observation, info = env.reset(seed=None)  # Puoi rimuovere seed se non necessario
            state_formatted = ut.state_formatter(observation)
            starting_money = budget
            done = False
            current_tick = 0  # Inizializza il tick corrente

            print(f"Training iteration: {i+1}/{iterations}")

            while not done:
                # L'azione viene scelta in base allo stato corrente
                action = self.act(state_formatted)
        
                # Mappa le azioni dell'agente alle azioni dell'ambiente
                if action == 1:
                    env_action = 1  # Compra
                elif action == 2:
                    env_action = 0  # Vendi
                else:
                    env_action = 1  # Azione predefinita (Compra)

                # Esegui l'azione nell'ambiente
                observation, env_reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
                current_tick += 1  # Incrementa il tick corrente

                # Ottieni il prezzo attuale dalla tendenza
                if current_tick < len(self.trend):
                    actual_price = self.trend[current_tick]
                else:
                    actual_price = self.trend[-1]

                # Logica di compravendita basata sull'azione scelta
                if action == 1 and starting_money >= actual_price and current_tick < (len(self.trend) - self.half_window):
                    inventory.append(actual_price)
                    starting_money -= actual_price
                elif action == 2 and len(inventory) > 0:
                    bought_price = inventory.pop(0)
                    total_profit += actual_price - bought_price
                    starting_money += actual_price

                # Calcola la reward
                step_reward = float(env_reward)

                # Format dell'osservazione successiva
                observation_formatted = ut.state_formatter(observation)

                # Aggiungi l'esperienza alla memoria
                self.memory.append((state_formatted, action, step_reward, observation_formatted, starting_money < budget))
                state = observation_formatted

                # Esegui il replay se ci sono abbastanza esperienze
                batch_size = min(self.batch_size, len(self.memory))
                if batch_size > 0:
                    try:
                        loss = self.replay(batch_size)
                    except Exception as e:
                        print(f"Errore durante il replay: {e}")
                        print(f"Observation_formatted: {observation_formatted}")
                        print(f"Observation: {observation}")
                        break

            if (i + 1) % checkpoint == 0:
                print(f'Epoch: {i + 1}, Total Profit: {total_profit:.3f}, Loss: {loss:.6f}, Total Money: {starting_money:.2f}')
