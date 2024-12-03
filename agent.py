import utils as ut
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

 
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


    def train(self, iterations, checkpoint, budget, env: gym.Env):

        print("#######################################################")
        print(f"Start agent training over {iterations} iterations")
        
        for i in range(iterations):
            print(f"Training iteration number: {i}")
            
            '''total_profit e inventory devono essere gestiti dall'ambiente'''
            total_profit = 0 
            inventory = []

            # Reset dell'ambiente e gestione dell'osservazione iniziale
            observation, info = env.reset(seed=None)
            state = ut.state_formatter(observation)
            starting_money = budget
            done = False
            time_step = 0  # Inizializza il tick corrente

            #print(f"Training iteration: {i+1}/{iterations}")

            print(f"_current_tick: {env.unwrapped._current_tick}")
            while not done:
                # L'azione viene scelta in base allo stato corrente
                action = self.act(state)
        
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
                time_step += 1  # Incrementa il tick corrente

                # Ottieni il prezzo attuale dalla tendenza
                if time_step < len(self.trend):
                    actual_price = self.trend[time_step]
                else:
                    actual_price = self.trend[-1]

                # Logica di compravendita basata sull'azione scelta
                if action == 1 and starting_money >= actual_price and time_step < (len(self.trend) - self.half_window):
                    inventory.append(actual_price)
                    starting_money -= actual_price
                elif action == 2 and len(inventory) > 0:
                    bought_price = inventory.pop(0)
                    total_profit += actual_price - bought_price
                    starting_money += actual_price

                # Calcola la reward
                step_reward = float(env_reward)

                # Format dell'osservazione successiva
                new_state = ut.state_formatter(observation)

                # Aggiungi l'esperienza alla memoria
                self.memory.append((state, action, step_reward, new_state, starting_money < budget))
                state = new_state

                # Esegui il replay se ci sono abbastanza esperienze
                batch_size = min(self.batch_size, len(self.memory))
                if batch_size > 0:
                    loss = self.replay(batch_size)
            
            print(f"_current_tick (after training): {env.unwrapped._current_tick}")
            print(f"Len of ds: {len(self.trend)}")
            if (i + 1) % checkpoint == 0:
                print(f'Epoch: {i + 1}, Total Profit: {total_profit:.3f}, Loss: {loss:.6f}, Total Money: {starting_money:.2f}')


    def evalute_agent(self):
        #TO DO
        pass


    def remember(self):
        #TO DO
        pass