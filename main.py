import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from custom_env import CustomStocksEnv
from dqn_agent import DQNAgent
from q_learning_agent import QLAgent
import utils as ut


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


def select_agent(model, state_size, action_size, batch_size, device, epsilon_decay, gamma, lr):

    if model == "DQN":
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            batch_size=batch_size,
            device=device,
            epsilon_decay=epsilon_decay,
            gamma=gamma,
            lr=lr
        )
    elif model == "QL":
        agent = QLAgent(
            action_size=action_size,
        )
    else:
        raise ValueError(f"Modello '{model}' non supportato.")
    return agent


def main():
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, choices=["DQN", "QL"], default="DQN", help="Modello da utilizzare (DQN/QL)")
    argparser.add_argument("--episodes", type=int, default=200, help="Numero di episodi per il training")
    argparser.add_argument("--initial-balance", type=int, default=10000, help="Saldo iniziale")
    argparser.add_argument("--seed", type=int, default=None, help="Seed per la riproducibilità")
    argparser.add_argument("--epsilon-decay", type=float, default=0.95, help="Fattore di decadimento del parametro epsilon")

    args = argparser.parse_args()

    model = args.model
    episodes = args.episodes
    initial_balance = args.initial_balance
    epsilon_decay = args.epsilon_decay
    symbols = ["AAPL", "NVDA", "TSLA", "RIOT", "UBER", "AMZN", "UAA", "INTC", "F", "GME", "QUBT"]
    window_size = 30
    
    print("Inizializzazione dell'ambiente di training...")
    env = setup_environment(
        symbols=symbols,
        start_date="2020-01-01",
        end_date="2024-12-30",
        window_size=window_size,
        initial_balance=initial_balance,
        model=model
    )

    print(f"Ambiente inizializzato. Prezzi shape: {env.prices.shape}, "
          f"Signal features shape: {env.signal_features.shape}")
   
    action_size = env.action_space.n

    # Se DQN, prepariamo device, state_size e batch
    if model == "DQN":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        print(f"State_size = {state_size}")
        batch_size = 128
        
        print(f"State size: {state_size}, Action size: {action_size}, "
              f"Batch size: {batch_size}, Device: {device}")
    else:
        # Per Q-Learning questi parametri non servono
        device = None
        state_size = None
        batch_size = None

    # Inizializza agente
    print("Inizializzazione dell'agente...")
    agent = select_agent(
        model=model,
        state_size=state_size,
        action_size=action_size,
        batch_size=batch_size,
        device=device,
        epsilon_decay = epsilon_decay,
        gamma=0.95,
        lr=1e-3
    )
    
    # Settiamo la render mode: "episode" durante il training
    agent.set_render_mode("episode")
    

    print(f"Inizio addestramento per {episodes} episodi...")
    agent.train_agent(env, episodes, seed=args.seed)
    print("Addestramento completato.")

    # Passiamo a valutazione con un nuovo dataset
    eval_symbols = ["MRVL"]
    print("Inizializzazione dell'ambiente di valutazione...")
    eval_env = setup_environment(
        symbols=eval_symbols,
        start_date="2020-01-01",
        end_date="2024-12-30",
        window_size=window_size,
        initial_balance=initial_balance,
        model=model
    )

    print("Inizio valutazione dell'agente.")
    print("Max possible profit: ", eval_env.max_possible_profit())
    agent.set_render_mode("step")
    info, history = agent.evaluate_agent(eval_env, seed=args.seed) 

    print("___ Valutazione ___")
    print(f"Total Profit: {info['total_profit']:.2f} - Mean: {np.mean(history['total_profit']):.2f} - Std: {np.std(history['total_profit']):.2f}")
    print(f"Wallet value: {info['wallet_value']:.2f} - Mean: {np.mean(history['wallet_value']):.2f} - Std: {np.std(history['wallet_value']):.2f}")
    print(f"Total Reward: {info['total_reward']:.2f} - Mean: {np.mean(history['total_reward']):.2f} - Std: {np.std(history['total_reward']):.2f}")
    print(f"ROI: {info['roi']:.2f}% - Mean: {np.mean(history['roi']):.2f}% - Std: {np.std(history['roi']):.2f}%")

    # Mostra il grafico (o i grafici) generati
    plt.show(block=True)  # Aspetta la chiusura della finestra del grafico


if __name__ == "__main__":
    main()
