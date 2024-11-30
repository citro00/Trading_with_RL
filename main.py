from agent import Agent
import utils as ut
import torch
import plotly.graph_objects as go


#Download and clean data
start_date = "2020-01-01"
end_date = "2024-12-30"
symbol = "AAPL"

data =  ut.download(symbol, start_date, end_date)
close = ut.get_close_data(ut.cleaning(data))
print(close)
print(len(close))


#Initialize agent variables
initial_money = 10000
window_size = 30
#Lo skip è lo step, quindi una entry alla volta
skip = 1
batch_size = 32

# Determine the device to use (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the Agent
agent = Agent(state_size=window_size,
              window_size=window_size,
              trend=close,
              skip=skip,
              batch_size=batch_size,
              device=device)

#Train the agent
#Iteration è il numero di esecuzioni che l'agente deve effettuare su tutto il dataset (numero di epoche)
#Checkpoint è solamente un parametro per indicare ogni quante epoche restituire un check
agent.train(iterations=50, checkpoint=5, initial_money=initial_money)

#Evaluate the agent
states_buy, states_sell, total_gains, invest, shares_held = agent.buy(initial_money)
print(f"States_buy: {states_buy}\nStates_sell: {states_sell}\nTotal_gains: {total_gains}\nInvest: {invest}\nShere_held: {shares_held}")


starting_money = 100000

final_share_price = close[-1]  # Final share price
total_portfolio_value = starting_money + shares_held * final_share_price
total_gains = total_portfolio_value - starting_money

fig = go.Figure()

# Candlestick trace
fig.add_trace(go.Candlestick(x=data.index,
                             open=data['Open'],
                             high=data.loc['High'],
                             low=data.loc['Low'],
                             close=data.loc['Close']))

# Buy signals trace
fig.add_trace(go.Scatter(x=[data.index[i] for i in states_buy],
                         y=[close[i] for i in states_buy],
                         mode='markers',
                         name='Buy Signals',
                         marker=dict(symbol='triangle-up', size=10, color='green')))

# Sell signals trace
fig.add_trace(go.Scatter(x=[data.index[i] for i in states_sell],
                         y=[close[i] for i in states_sell],
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