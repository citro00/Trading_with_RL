import yfinance as yf
import pandas as pd
import numpy as np
from environment import CustomStocksEnv  # Importa l'ambiente personalizzato per il trading
from agent import QLearningAgent  # Importa l'agente Q-Learning
# from utils import make_state_hashable  # Non necessario nel main script
import test_dati


###MODIFICARE, UTILIZZARE MODULO DI MATTIA PER IL DATASET

# Scarica i dati storici di mercato per AAPL (Apple)
asset = "BTC-USD"
start_date = "2020-01-01"
end_date = "2023-12-30"
'''data = yf.download("AAPL", start=start_date, end=end_date, interval="1d")'''

# Prepara i dati per Gym-anytrading: seleziona le colonne rilevanti e rimuovi i valori nulli
'''data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
data.reset_index(drop=True, inplace=True)'''

###MODIFICARE, UTILIZZARE MODULO DI MATTIA PER IL DATASET
data = test_dati.cleaning(test_dati.download(asset, start_date, end_date))


# Crea l'ambiente Gym-anytrading con la classe personalizzata
env = CustomStocksEnv(df=data, window_size=10, frame_bound=(10, len(data)), render_fps=1)

# Inizializza l'agente Q-Learning
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = QLearningAgent(state_size=state_size, action_size=action_size, epsilon=1.0, epsilon_decay=0.97, epsilon_min=0.05)

# Addestramento
episodes = 100
for e in range(episodes):
    state, info = env.reset()
    # Non necessario: state = make_state_hashable(state)
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        # Non necessario: next_state = make_state_hashable(next_state)
        done = terminated or truncated
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        if done:
            break

    agent.decay_epsilon()

    # Calcolo del ROI alla fine di ogni episodio
    try:
        close_price = env.df['Close'].iloc[env.current_step].item()
        final_portfolio_value = float(env.cash + (env.shares * close_price))
    except AttributeError:
        # Se env.current_step non è disponibile o è fuori range
        final_portfolio_value = float(env.cash + (env.shares * env.df['Close'].iloc[-1].item()))

    roi = (final_portfolio_value - env.initial_cash) / env.initial_cash

    # Stampa dell'episodio, ricompensa totale, dettagli di portafoglio e ROI
    print(f"\nEpisode {e+1}/{episodes}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Valore finale del portafoglio: {final_portfolio_value:.2f} USD")
    print(f"Contante rimanente: {env.cash:.2f} USD")
    print(f"Azioni possedute: {env.shares}")
    print(f"Numero totale di operazioni di trading: {env.num_trades}")
    print(f"ROI: {roi:.2%}")

# Valutazione dell'Agente
print("\nInizio della valutazione dell'agente:")
state, info = env.reset()

while True:
    action = np.argmax(agent._get_q_values(state))
    next_state, reward, terminated, truncated, info = env.step(action)
    env.render()

    if env.current_step < len(env.df):
        prezzo_valutazione = float(env.df['Close'].iloc[env.current_step].item())
        if action == 1:
            print(f"[Valutazione] Acquisto al passo {env.current_step}, Prezzo: {prezzo_valutazione:.2f}")
        elif action == 2:
            print(f"[Valutazione] Vendita al passo {env.current_step}, Prezzo: {prezzo_valutazione:.2f}")
    else:
        prezzo_valutazione = float(env.df['Close'].iloc[-1].item())
        print(f"[Valutazione] Passo {env.current_step} oltre l'ultimo indice, Prezzo finale: {prezzo_valutazione:.2f}")

    state = next_state
    if terminated or truncated:
        break

# Calcolo del ROI alla fine della valutazione
try:
    close_price = env.df['Close'].iloc[env.current_step].item()
    final_portfolio_value = float(env.cash + (env.shares * close_price))
except AttributeError:
    final_portfolio_value = float(env.cash + (env.shares * env.df['Close'].iloc[-1].item()))
roi = (final_portfolio_value - env.initial_cash) / env.initial_cash

# Stampa finale dopo la valutazione
print("\nValutazione finale dell'agente:")
print(f"Valore finale del portafoglio: {final_portfolio_value:.2f} USD")
print(f"Contante rimanente: {env.cash:.2f} USD")
print(f"Azioni possedute: {env.shares}")
print(f"Numero totale di operazioni di trading durante la valutazione: {env.num_trades}")
print(f"ROI finale: {roi:.2%}")
