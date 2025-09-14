import trueskillthroughtime as ttt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

import league_data_handler as ldh
from games_image_generator import get_web_games

NNLS6 = ldh.LeagueData('NNLSeason6')
S6df = get_web_games(NNLS6, edit_mode_file=True)

NNLS7 = ldh.LeagueData('NNLSeason7')
S7df = get_web_games(NNLS7, edit_mode_file=True)
df = pd.concat([S6df, S7df], ignore_index=True)
df = df.sort_values(by='date_time_end', ascending=True)

users = df['away_user'].to_list() + df['home_user'].to_list()
unique_users = list(set(users))

compositions = []

for _, row in df.iterrows():
    compositions.append([[row['winner_user']], [row['loser_user']]])

print(compositions)

history = ttt.History(compositions, gamma=0.5)
history.convergence()
learning_curves = history.learning_curves()
print(learning_curves)

print(vars(list(history.agents.values())[0].player))

def get_last_mu_sigma_df(learning_curves):
    data = []
    
    for player, curve in learning_curves.items():
        # Get the last entry in the player's curve
        last_entry = curve[-1]
        
        # Extract mu and sigma from the last entry
        mu = last_entry[1].mu
        sigma = last_entry[1].sigma
        
        # Add player name, mu, and sigma to the data list
        data.append({'Player': player, 'Mu': mu, 'Sigma': sigma})
    
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    
    df_sorted = df.sort_values(by='Mu', ascending=False)
    
    return df_sorted

# Call the function with your data
last_mu_sigma_df = get_last_mu_sigma_df(learning_curves)

# Output the DataFrame
print(last_mu_sigma_df)


def win_probability_from_names(player1, player2, df, beta=1.0):
    a_stats = df[df['Player'] == player1].iloc[0]
    b_stats = df[df['Player'] == player2].iloc[0]
    
    mu1, sigma1 = a_stats['Mu'], a_stats['Sigma']
    mu2, sigma2 = b_stats['Mu'], b_stats['Sigma']
    
    denom = np.sqrt(2 * beta**2 + sigma1**2 + sigma2**2)
    return norm.cdf((mu1 - mu2) / denom)

print(win_probability_from_names('MattGree', 'VicklessFalcon', last_mu_sigma_df))