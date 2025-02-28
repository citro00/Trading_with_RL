import json
from collections import defaultdict
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

ql_train_data = json.load(open("tuning_ql/final_model__2000eps/training/data.json"))
ql_test_data = json.load(open("tuning_ql/final_model__2000eps/testing/data.json"))
dqn_train_data = json.load(open("tuning/epsilon_decay__2000eps-gamma999/training/data.json"))
dqn_test_data = json.load(open("tuning/epsilon_decay__2000eps-gamma999/testing/data.json"))

ql_data_key = ''
dqn_data_key = 'epsilon_decay: 0.99'

train_data = []
test_data = []

train_data.append({k: v[ql_data_key] for k, v in ql_train_data.items()})
train_data.append({k: v[dqn_data_key] for k, v in dqn_train_data.items()})
test_data.append({k: v[ql_data_key] for k, v in ql_test_data.items()})
test_data.append({k: v[dqn_data_key] for k, v in dqn_test_data.items()})

labels = ['QL', 'DQN']


folder = Path("./model_comparison/")
training_folder = (folder/'training')
testing_folder = (folder/'testing')

print("Creazione dei percorsi...")
training_folder.mkdir(parents=True, exist_ok=True)
testing_folder.mkdir(parents=True, exist_ok=True)

json.dump(train_data, open(training_folder / "data.json", "w"), indent=2)
json.dump(test_data, open(testing_folder / "data.json", "w"), indent=2)

save_train_metrics(train_data, labels, training_folder)
save_eval_metrics(test_data, labels, testing_folder)