from typing import Literal
from collections import defaultdict
import numpy as np
import random
from gym_anytrading.envs import TradingEnv

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

    def act(self, env, obs):
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
        pass
        
    def evaluate_agent(self):
        pass

    def _discretize(self):
        pass