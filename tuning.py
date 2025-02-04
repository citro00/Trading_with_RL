import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from custom_env import CustomStocksEnv
from dqn_agent import DQNAgent
import utils as ut
import torch
import torch.nn as nn
import json

# Parametri per il fine-tuning
PARAMETERS = {
    "batch_size": [128],
    # "epsilon_decay": [0.9, 0.95, 0.99, 0.995, 0.999],
    "epsilon_decay": [0.99],
    # "gamma": [0.9, 0.99, 0.999],
    "gamma": [0.999],
    # "lr": [1e-2, 1e-3, 5e-3, 5e-4],
    "lr": [5e-4],
    "loss_fn": [nn.SmoothL1Loss],
    # "loss_fn": [nn.MSELoss],
    "use_profit": [True],
    "net_hidden_dim": [320],
    "net_hidden_layers": [2]
}

def setup_environment(symbols, start_date, end_date, window_size, initial_balance) -> CustomStocksEnv:
    # Scarica i dati come dizionario
    data = ut.get_data_dict(start_date, end_date, symbols)
    
    # Crea i limiti del frame
    frame_bound = (window_size, len(data.get(list(data.keys())[0])))
    
    # Crea l'ambiente
    env = CustomStocksEnv(
        df=data,
        window_size=window_size,
        frame_bound=frame_bound,
        normalize=True,
        initial_balance=initial_balance
    )
    return env

def iterate_args_values(args: dict):
    """
    Generates all possible combinations of argument values from a dictionary.

    Args:
        args (dict): A dictionary where keys are argument names and values are lists of possible values for those arguments.

    Yields:
        dict: A dictionary representing one combination of argument values.
    """
    keys, values = zip(*args.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def rolling_average(data, window_size):
    """
    Calcola la media mobile di una serie temporale.
    """
    return np.convolve(data, np.ones(window_size), mode="valid") / window_size

def save_train_metrics(trains, labels, save_path):
    metrics = ['total_reward', 'roi', 'total_profit', 'deal_actions_num', 'deal_errors_num', 'drawdown_mean', 'performance']
    rolling_length = 50

    for metric in metrics:
        fig = plt.figure(figsize=(15,10))
        ax = fig.gca()
        for i, train_instance in enumerate(trains):
            label = labels[i]
            moving_avg = rolling_average(train_instance[metric], rolling_length)
            ax.plot(range(rolling_length, len(moving_avg) + rolling_length), moving_avg, label=label)

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Moving Average")
        ax.legend()
        fig.savefig(save_path/f"{metric}.png")
        plt.show(block=False)

        # salva tutti i dati
        df = pd.DataFrame(trains, index=labels)
        df.to_json(save_path / "data.json")

def save_eval_metrics(evals, labels, save_path):
    metrics = ['total_reward', 'roi', 'total_profit', 'deal_actions_num', 'deal_errors_num', 'drawdown']
    
    summary = pd.DataFrame(columns=["value", "avg", "std"])
    for metric in metrics:
        fig = plt.figure(figsize=(15,10))
        ax = fig.gca()
        for i, eval_instance in enumerate(evals):
            label = labels[i]
            ax.plot(range(len(eval_instance[metric])), eval_instance[metric], label=label)
            # ax.text(len(eval_instance[metric])-1, eval_instance[metric], f"mean: {avg}")
            summary.loc[label] = pd.Series({
                "value": eval_instance[metric][-1], 
                "avg": np.mean(eval_instance[metric]),
                "std": np.std(eval_instance[metric])
            })


        # Salva il sommario
        summary.to_csv(save_path/f"{metric}.csv")

        # Salva il plot
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Steps")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        fig.savefig(save_path/f"{metric}.png")

    # Salva tutti i dati
    df = pd.DataFrame(evals, index=labels)
    df.to_json(save_path / "data.json") 

def main(args):
    # symbols = ["IBM", "NVDA", "AAPL", "GOOGL", "AMZN", "MSFT", "INTC", "ORCL", "CSCO", "ADBE", "QCOM", "META"]
    symbols = ["AAPL", "NVDA", "TSLA", "RIOT", "UBER", "AMZN", "UAA", "INTC", "F", "GME", "QUBT", "TNDM", "JBHT", "TENB", "GSHD", "BRKR", "SNDR", "SNAP", "WBA", "PII", "APA", "SWTX"]
    window_size = 30
    initial_balance = args.initial_balance
    training_episodes = args.episodes
    testing_parameters = [k for k, v in PARAMETERS.items() if len(v) > 1]

    # Crea il percorso per il salvataggio dei risultati   
    output_folder = args.output_folder or '-'.join(testing_parameters)
    if args.tags:
        output_folder = output_folder + '__' + '-'.join(args.tags)

    folder = Path(f"./tuning/{output_folder}")
    training_folder = (folder/'training')
    models_folder = (folder/'models')
    testing_folder = (folder/'testing')
    
    print("Creazione dei percorsi...")
    training_folder.mkdir(parents=True, exist_ok=True)
    models_folder.mkdir(parents=True, exist_ok=True)
    testing_folder.mkdir(parents=True, exist_ok=True)

    with open(folder/"parameters.txt", "w") as fp:
        fp.write(f"Episodes: {training_episodes}\n")
        for k, v in PARAMETERS.items():
            vv = v[0] if len(v) == 1 else str(v)
            fp.write(f"{k}: {vv}\n")

    print("Inizializzazione dell'ambiente di training...")
    env = setup_environment(
        symbols=symbols,
        start_date="2020-01-01",
        end_date="2024-12-30",
        window_size=window_size,
        initial_balance=initial_balance
    )
    print(f"Ambiente inizializzato. Prezzi shape: {env.prices.shape}, "
          f"Signal features shape: {env.signal_features.shape}")
    
    eval_symbols = ["NFLX"]
    print("Inizializzazione dell'ambiente di valutazione...")
    eval_env = setup_environment(
        symbols=eval_symbols,
        start_date="2020-01-01",
        end_date="2024-12-30",
        window_size=window_size,
        initial_balance=initial_balance
    )
    print(f"Ambiente inizializzato. Prezzi shape: {eval_env.prices.shape}, "
          f"Signal features shape: {eval_env.signal_features.shape}")
    
    # Parametri dell'agente
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"State size: {state_size}, Action size: {action_size}, "
            "Device: {device}")

    print("Avvio del processo di ricerca dei parametri...")
    per_train_metrics = []
    per_eval_metrics = []
    labels = []

    total_iteration = len(list(iterate_args_values(PARAMETERS)))
    for params in tqdm(iterate_args_values(PARAMETERS), total=total_iteration, desc="Grid Search Progress", unit="iteration"):
        model_str = "_".join([f"{k}{v}" for k, v in params.items()])
        labels_lst = [f"{p}: {v}" for p, v in params.items() if p in testing_parameters]
        labels.append(" ".join(labels_lst))
        
        # Inizializza agente
        tqdm.write(f"Parametri: {params}")
        tqdm.write("Inizializzazione dell'agente...")
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            device=device,
            **params
        )
        # Disattiva il rendering delle metriche e dell'ambiente
        agent.set_render_mode("off")

        tqdm.write(f"Addestramento per {training_episodes} episodi...")
        info, per_step_metrics, per_episode_metrics = agent.train_agent(env, training_episodes, seed=42)
        per_train_metrics.append(per_episode_metrics)

        # Chiude l'ambiente di training
        env.close()

        tqdm.write("Valutazione agente...")
        metrics = ['total_reward', 'roi', 'total_profit', 'deal_actions_num', 'deal_errors_num', 'drawdown']
        info, history = agent.evaluate_agent(eval_env, seed=42)
        eval_data = dict(filter(lambda item: item[0] in metrics, history.items()))
        per_eval_metrics.append(eval_data)

        # Salva l'ultimo episodio
        eval_env.render_figure(title=model_str).savefig(testing_folder/f"eval_{model_str}.png")

        # Chiude l'ambiente di valutazione
        eval_env.close()

        # Salva il modello
        agent.save_model(models_folder, model_str)


    # Salva le metriche
    print("Salvataggio delle metriche...")
    save_train_metrics(per_train_metrics, labels, training_folder)
    save_eval_metrics(per_eval_metrics, labels, testing_folder)

    # Mostra il grafico (o i grafici) generati
    print("Completato.")

    if args.show_plots:
        plt.show(block=True)  # Aspetta la chiusura della finestra del grafico


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--episodes", type=int, default=200, help="Numero di episodi per il training")
    argparser.add_argument("--initial-balance", type=int, default=10000, help="Saldo iniziale")
    argparser.add_argument("--show-plots", help="Mostra i plot generati", action="store_true")
    argparser.add_argument("--output-folder", type=str, default=None, help="Cartella di output per i risultati")
    argparser.add_argument('--tags', type=str, nargs='+', help='Tags', required=False)
    args = argparser.parse_args()

    main(args)
