import numpy as np
import random
from utils import make_state_hashable

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def _get_q_values(self, state):
        state_key = make_state_hashable(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return self.q_table[state_key]

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_values = self._get_q_values(state)
        max_q = np.max(q_values)
        actions_with_max_q = np.where(q_values == max_q)[0]
        return np.random.choice(actions_with_max_q)

    def learn(self, state, action, reward, next_state):
        q_values = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)
        q_values[action] += self.alpha * (reward + self.gamma * np.max(next_q_values) - q_values[action])

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
