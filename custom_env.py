from enum import Enum
from gym_anytrading.envs import TradingEnv
import pandas as pd
from action import *
from position import *


class CustomStocksEnv(TradingEnv):
    def __init__(self, df:pd.DataFrame, window_size:int, render_mode=None):
        super().__init__(df, window_size, render_mode)

    def step(self, action):
        #TO DO
        pass


    def _process_data(self):
        #TO DO
        pass


    def _calculate_reward(self, action):
        #TO DO
        pass


    def _update_profit(self, action):
        #TO DO
        pass


    def max_possible_profit(self):  # trade fees are ignored
        #TO DO
        pass