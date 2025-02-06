import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from custom_env import CustomStocksEnv
from dqn_agent import DQNAgent
from q_learning_agent import QLAgent
import utils as ut
from action import Action

def setup_environment(symbols, start_date, end_date, window_size, initial_balance, model):

    # Scarica i dati come dizionario
    data = ut.get_data_dict(start_date, end_date, symbols)
    
    # Crea i limiti del frame
    frame_bound = (window_size, len(data.get(list(data.keys())[0])))
    
    # Crea l'ambiente
    env = CustomStocksEnv(
        df=data,
        window_size=window_size,
        frame_bound=frame_bound,
        normalize=True if model == "DQN" else False,
        initial_balance=initial_balance
    )
    return env

def main():
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--initial-balance", type=int)
    argparser.add_argument("--symbol", type=str, default="NFLX")
    argparser.add_argument("--seed", type=int, default=None)

    args = argparser.parse_args()

    model = 'DQN'
    initial_balance = args.initial_balance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_symbols = [args.symbol]
    print("Inizializzazione dell'ambiente di valutazione...")
    env = setup_environment(
        symbols=eval_symbols,
        start_date="2020-01-01",
        end_date="2024-12-30",
        window_size=30,
        initial_balance=initial_balance,
        model=model
    )

    action_size = env.action_space.n
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    print(f"State_size = {state_size}")
    batch_size = 128
        
    print(f"State size: {state_size}, Action size: {action_size}, "
              f"Batch size: {batch_size}, Device: {device}")
    
    # Inizializza agente
    print("Inizializzazione dell'agente...")
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        batch_size=batch_size,
        device=device,
        epsilon_decay=0.99,
        gamma=0.99,
        lr=5e-4
    )
    agent.load_model(args.model)

    print("Inizio valutazione dell'agente.")
    print("Max possible profit: ", env.max_possible_profit())
    agent.set_render_mode("step")
    info, history = agent.evaluate_agent(env, seed=args.seed) 

    trade_num = 0
    trading_distances = []
    last_buy_tick = None
    for item in history['action']:
        if item is None:
            continue
    
        tick, action = item
        if action == Action.Buy:
            last_buy_tick = tick
        elif action == Action.Sell:
            trade_num += 1
            trading_distances.append(tick - last_buy_tick)

    
    print("___ Valutazione ___")
    print(f"Total Profit: {info['total_profit']:.2f} - Mean: {np.mean(history['total_profit']):.2f} - Std: {np.std(history['total_profit']):.2f}")
    print(f"Wallet value: {info['wallet_value']:.2f} - Mean: {np.mean(history['wallet_value']):.2f} - Std: {np.std(history['wallet_value']):.2f}")
    print(f"Total Reward: {info['total_reward']:.2f} - Mean: {np.mean(history['total_reward']):.2f} - Std: {np.std(history['total_reward']):.2f}")
    print(f"ROI: {info['roi']:.2f}% - Mean: {np.mean(history['roi']):.2f}% - Std: {np.std(history['roi']):.2f}%")
    print(f"{trade_num} trades executed. Trading distance: Mean: {np.mean(trading_distances):.3f} - Std: {np.std(trading_distances):.3f}")

    # Mostra il grafico (o i grafici) generati
    plt.show(block=True)  # Aspetta la chiusura della finestra del grafico


if __name__ == "__main__":
    main()
