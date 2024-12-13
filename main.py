import pandas as pd
from custom_env import CustomStocksEnv
from agent import Agent
import utils as ut
import torch
import matplotlib.pyplot as plt

# Parametri per il download dei dati
start_date = "2020-01-01"
end_date = "2024-12-30"
symbol = "AAPL"

data = ut.download(symbol, start_date, end_date)

data = ut.cleaning(data)

window_size = 30
end_frame = (len(data)//4)*3
frame_bound = (window_size, end_frame)
initial_balance = 1000

print("Inizializzazione dell'ambiente...")
env = CustomStocksEnv(
    df=data,
    window_size=window_size,
    frame_bound=frame_bound,
    initial_balance=initial_balance
)
print(f"Ambiente inizializzato. Prezzi shape: {env.prices.shape}, Signal features shape: {env.signal_features.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n
batch_size = 64

print(f"State size: {state_size}, Action size: {action_size}, Batch size: {batch_size}, Device: {device}")

print("Inizializzazione dell'agente...")
agent = Agent(
    state_size=state_size,
    action_size=action_size,
    batch_size=batch_size,
    device=device,
    initial_balance=initial_balance
)

episodes = 50
agent.train_agent(env, episodes)
env.save_reward_history("Reward_History_Training.csv")

#model_path = "agent_model.pth"
#torch.save(agent.model.state_dict(), model_path)
#print(f"Modello salvato in {model_path}")

env = CustomStocksEnv(
    df=data,
    window_size=window_size,
    frame_bound=(end_frame+1, len(data)),
    initial_balance=initial_balance
)

print("Inizio valutazione dell'agente.")
states_buy, states_sell, total_profit, total_reward = agent.evaluate_agent(env)
print(f"Total Profit: {total_profit}")
env.save_reward_history("Reward_history_evaluate.csv")

plt.figure(figsize=(15, 5))
plt.plot(env.prices, color='r', lw=2., label='Price')

if states_buy:
    plt.plot(states_buy, env.prices[states_buy], '^', markersize=10, color='m', label='Buy Signal')
if states_sell:
    plt.plot(states_sell, env.prices[states_sell], 'v', markersize=10, color='k', label='Sell Signal')

plt.title(f'Total Profit: {total_profit:.2f}; Total Reward: {total_reward}')
plt.xlabel('Tick')
plt.ylabel('Price')
plt.legend()
plt.show()
