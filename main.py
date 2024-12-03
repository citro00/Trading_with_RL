from agent import Agent
import utils as ut
import torch
import plotly.graph_objects as go
import gym_anytrading
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

#Download data
start_date = "2020-01-01"
end_date = "2024-12-30"
symbol = "AAPL"

data = ut.download(symbol, start_date, end_date)

#Data preprocessing
close_prcie = ut.get_close_data(ut.cleaning(data))

#Agent variables
budget = 10000
window_size = 30
skip = 1
batch_size = 32

#Select device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Create agent instance
agent = Agent(state_size=window_size,
              window_size=window_size,
              trend=close_prcie,
              skip=skip,
              batch_size=batch_size,
              device=device)

#Env initialization
env = gym.make('stocks-v0', frame_bound=(30, len(data)-1), window_size=window_size)

agent.new_train(iterations=50, checkpoint=1, budget=budget, env=env)


#Evaluate the agent
states_buy, states_sell, total_gains, invest, shares_held = agent.buy(budget)
print(f"States_buy: {states_buy}\nStates_sell: {states_sell}\nTotal_gains: {total_gains}\nInvest: {invest}\nShere_held: {shares_held}")


starting_money = 100000

final_share_price = close_prcie[-1]  # Final share price
total_portfolio_value = starting_money + shares_held * final_share_price
total_gains = total_portfolio_value - starting_money

fig = go.Figure()

# Candlestick trace
fig.add_trace(go.Candlestick(x=data.index,
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close']))

# Buy signals trace
fig.add_trace(go.Scatter(x=[data.index[i] for i in states_buy],
                         y=[close_prcie[i] for i in states_buy],
                         mode='markers',
                         name='Buy Signals',
                         marker=dict(symbol='triangle-up', size=10, color='green')))

# Sell signals trace
fig.add_trace(go.Scatter(x=[data.index[i] for i in states_sell],
                         y=[close_prcie[i] for i in states_sell],
                         mode='markers',
                         name='Sell Signals',
                         marker=dict(symbol='triangle-down', size=10, color='red')))

# Set layout
fig.update_layout(
    title=f'Total Gains: {total_gains:.2f}, Total Portfolio Value: {total_portfolio_value:.2f}',
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_dark',
    legend=dict(x=0, y=1, orientation='h')
)

fig.show()