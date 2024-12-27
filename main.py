import pandas as pd
from custom_env import CustomStocksEnv
from dqn_agent import DQNAgent
import utils as ut
import torch
import matplotlib.pyplot as plt
from q_learning_agent import QLAgent

# Parametri per il download dei dati
symbols = ["AAPL", "NVDA", "TSLA", "RIOT", "UBER", "AMZN", "UAA", "INTC", "F", "GME", "QUBT"]

data = ut.get_data_dict("2020-01-01", "2024-12-30", symbols)
keys = list(data.keys())

window_size = 30
frame_bound = (window_size, len(data.get(keys[0])))
initial_balance = 2000
model = "QL"


print("Inizializzazione dell'ambiente...")
env = CustomStocksEnv(
    df=data,
    window_size=window_size,
    frame_bound=frame_bound,
    normalize=True if model == "DQN" else False,
    initial_balance=initial_balance
)
print(f"Ambiente inizializzato. Prezzi shape: {env.prices.shape}, Signal features shape: {env.signal_features.shape}")

episodes = 200
action_size = env.action_space.n


if model == "DQN":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    batch_size = 64

    print(f"State size: {state_size}, Action size: {action_size}, Batch size: {batch_size}, Device: {device}")

    print("Inizializzazione dell'agente...")
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        batch_size=batch_size,
        device=device,
        initial_balance=initial_balance
    )


elif model == "QL":
    agent = QLAgent(
        action_size,
        initial_balance
    )

agent.set_render_mode("episode")

agent.train_agent(env, episodes)

symbols = ["MRVL"]
data = ut.get_data_dict("2020-01-01", "2024-12-30", symbols)

env = CustomStocksEnv(
    df=data,
    window_size=window_size,
    frame_bound=frame_bound,
    normalize=True if model == "DQN" else False,
    initial_balance=initial_balance
)

print("Inizio valutazione dell'agente.")
agent.set_render_mode("step")
total_profit, total_reward, info = agent.evaluate_agent(env)

plt.show(block=True)  # Aspetta la chiusura della finestra del grafico
