from league_data_handler import LeagueData, get_web_games
from games_image_generator import GeneratorSettings, standings_generator, scorecard_generator, make_weekly_graphics
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from fuzzywuzzy import process
from pyRio.api_manager import APIManager
from pyRio.web_functions import game_mode_ladder
from games_image_generator import generate_stadings_df, remove_games_after_cutoff


manager = APIManager()
elo_map = pd.DataFrame(game_mode_ladder(manager, 'NNL Season 7')).T['rating'].to_dict()

NNLS7 = LeagueData('NNLSeason7')
df = get_web_games(NNLS7, edit_mode_file=True)

def match_player_to_team(player_name):
        best_match = process.extractOne(player_name.lower(), NNLS7.player_data.keys())
        if best_match:
            # Return the team information corresponding to the best match
            return best_match[0]
        return None

matchups = []
for week, games in NNLS7.matchups.items():
    for game in games:
        matchups.append(tuple(sorted([match_player_to_team(game[0]), match_player_to_team(game[1])])))  # Sort to handle order-independent comparison

# Now, extract the games that have already been played in your df
# Assuming your DataFrame df has columns 'home_team' and 'away_team'
played_games = []
for index, row in df.iterrows():
    played_games.append(tuple(sorted([row['home_user'].lower(), row['away_user'].lower()])))  # Sort for order-independent comparison

# Convert to set for faster lookup
matchups_set = set(matchups)
played_games_set = set(played_games)

# Find the remaining matchups (matchups that have not been played yet)
remaining_matchups = matchups_set - played_games_set

print(remaining_matchups)

def user_runs_for_and_against(user, df):
    user_games = df[(df['away_user'] == user) | (df['home_user'] == user)]
    user_runs_scored = user_games.apply(lambda row: row['away_score'] if row['away_user'] == user else row['home_score'], axis=1)
    user_runs_against = user_games.apply(lambda row: row['home_score'] if row['away_user'] == user else row['away_score'], axis=1)
    avg_runs_scored = user_runs_scored.mean()
    var_runs_scored = user_runs_scored.var()

    avg_runs_against = user_runs_against.mean()
    var_runs_against = user_runs_against.var()

    num_games = len(user_games)

    return avg_runs_scored, var_runs_scored, avg_runs_against, var_runs_against, num_games

def matchup_run_distirbution(user1, user2, games_df):
    user1_rpg, user1_var_rpg, user1_rapg, user1_var_rapg, user1_games  = user_runs_for_and_against(user1, games_df)
    user2_rpg, user2_var_rpg, user2_rapg, user2_var_rapg, user2_games = user_runs_for_and_against(user2, games_df)

    def estimate_neg_bin_params(mean, variance):
        if variance > mean:
            p = mean / variance
            r = (mean ** 2) / (variance - mean)
            return r, p
        else:
            return None, None
        
    def weighted_variance(var1, n1, var2, n2):
        return ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    r1_scored, p1_scored = estimate_neg_bin_params(user1_rpg, user1_var_rpg)
    r1_allowed, p1_allowed = estimate_neg_bin_params(user1_rapg, user1_var_rapg)

    r2_scored, p2_scored = estimate_neg_bin_params(user2_rpg, user2_var_rpg)
    r2_allowed, p2_allowed = estimate_neg_bin_params(user2_rapg, user2_var_rapg)

    # Average means and variances
    user1_runs_matchup = (user1_rpg + user2_rapg) / 2
    user1_runs_var_matchup = weighted_variance(user1_var_rpg, user1_games, user2_var_rapg, user2_games)

    user2_runs_matchup = (user2_rpg + user1_rapg) / 2
    user2_runs_var_matchup = weighted_variance(user2_var_rpg, user2_games, user1_var_rapg, user1_games)

    # Calculate new r and p for the combined distributions
    user1_r_matchup, user1_p_matchup = estimate_neg_bin_params(user1_runs_matchup, user1_runs_var_matchup)
    user2_r_matchup, user2_p_matchup = estimate_neg_bin_params(user2_runs_matchup, user2_runs_var_matchup)

    return user1_r_matchup, user1_p_matchup, user2_r_matchup, user2_p_matchup


def monte_carlo(matchup, df, elo_map, num_simulations = 10000, elo_scaling = 200, plot=False):
    user1_r, user1_p, user2_r, user2_p = matchup_run_distirbution(matchup[0], matchup[1], df)

    runs_user1 = stats.nbinom.rvs(user1_r, user1_p, size=num_simulations)
    runs_user2 = stats.nbinom.rvs(user2_r, user2_p, size=num_simulations)

    run_differential = runs_user1 - runs_user2

    if matchup[0] in elo_map and matchup[1] in elo_map:
        elo_diff = elo_map[matchup[0]] - elo_map[matchup[1]]
    else:
        elo_diff = 0  # Default to 0 if we don't have ELO data

    elo_adjustment = elo_diff / elo_scaling
    run_differential = run_differential + elo_adjustment

    # Ensure no ties (adjust if zero), and handle small positive or negative values
    run_differential += (run_differential == 0) * int(np.sign(np.random.uniform(-1, 1)))
    # Round the run differential to specific values:
    # Round [-0.5, 0) to -1, and (0, 0.5) to 1
    run_differential = np.where((run_differential > -0.5) & (run_differential < 0), -1, run_differential)
    run_differential = np.where((run_differential >= 0) & (run_differential < 0.5), 1, run_differential)
    run_differential = np.where((run_differential >= 10), 10, run_differential)
    run_differential = np.where((run_differential <= -10), -10, run_differential)

    run_differential = np.round(run_differential).astype(int)

    if plot:
        # Count wins for User 1 and User 2
        user1_wins = np.sum(run_differential > 0)
        user2_wins = np.sum(run_differential < 0)
        
        # Calculate win probabilities
        user1_win_prob = user1_wins / num_simulations
        user2_win_prob = user2_wins / num_simulations

        # Print out win probabilities
        print(f"User 1 Win Probability: {user1_win_prob:.4f}")
        print(f"User 2 Win Probability: {user2_win_prob:.4f}")

        # Plot results
        plt.figure(figsize=(10, 5))
        sns.histplot(run_differential, bins=range(min(run_differential), max(run_differential) + 1), kde=True, stat="density")
        plt.axvline(np.mean(run_differential), color='red', linestyle='dashed', label=f"Mean: {np.mean(run_differential):.2f}")
        plt.xlabel("Run Differential (User1 - User2)")
        plt.ylabel("Density")
        plt.title(f"Monte Carlo Simulation of Run Differential ({num_simulations} Simulations)")
        plt.legend()
        plt.show()
    
    return run_differential

monte_carlo(('MattGree', 'Cezarito'), df, elo_map, plot=True)

def monte_carlo_two_game_series(matchup, df, elo_map, num_simulations=10000, elo_scaling=200, plot=False):
    """Simulates a two-game series between two players using Monte Carlo simulations."""
    
    # Get run distribution parameters
    user1_r, user1_p, user2_r, user2_p = matchup_run_distirbution(matchup[0], matchup[1], df)

    # Simulate TWO games separately
    runs_user1_game1 = stats.nbinom.rvs(user1_r, user1_p, size=num_simulations)
    runs_user2_game1 = stats.nbinom.rvs(user2_r, user2_p, size=num_simulations)

    runs_user1_game2 = stats.nbinom.rvs(user1_r, user1_p, size=num_simulations)
    runs_user2_game2 = stats.nbinom.rvs(user2_r, user2_p, size=num_simulations)

    # Compute run differential for both games and sum
    run_differential = (runs_user1_game1 - runs_user2_game1) + (runs_user1_game2 - runs_user2_game2)

    # Apply Elo adjustment
    if matchup[0] in elo_map and matchup[1] in elo_map:
        elo_diff = elo_map[matchup[0]] - elo_map[matchup[1]]
    else:
        elo_diff = 0  

    elo_adjustment = (elo_diff / elo_scaling) * 2  # Scaling for 2 games
    run_differential = (run_differential + elo_adjustment).astype(int)

    # Ensure no ties and apply rounding adjustments
    run_differential += (run_differential == 0) * int(np.sign(np.random.uniform(-1, 1)))
    run_differential = np.where((run_differential > -0.5) & (run_differential < 0), -1, run_differential)
    run_differential = np.where((run_differential >= 0) & (run_differential < 0.5), 1, run_differential)
    run_differential = np.clip(run_differential, -20, 20)  # Bound values for 2-game total

    run_differential = np.round(run_differential).astype(int)

    if plot:
        # Compute win probabilities over the series
        user1_series_wins = np.sum(run_differential > 0)
        user2_series_wins = np.sum(run_differential < 0)
        
        user1_series_win_prob = user1_series_wins / num_simulations
        user2_series_win_prob = user2_series_wins / num_simulations

        print(f"User 1 Series Win Probability: {user1_series_win_prob:.4f}")
        print(f"User 2 Series Win Probability: {user2_series_win_prob:.4f}")

        # Plot results
        plt.figure(figsize=(10, 5))
        sns.histplot(run_differential, bins=range(min(run_differential), max(run_differential) + 1), kde=True, stat="density")
        plt.axvline(np.mean(run_differential), color='red', linestyle='dashed', label=f"Mean: {np.mean(run_differential):.2f}")
        plt.xlabel("Total Run Differential Over 2 Games")
        plt.ylabel("Density")
        plt.title(f"Monte Carlo Simulation of Two-Game Series ({num_simulations} Simulations)")
        plt.legend()
        plt.show()
    
    return run_differential

# monte_carlo_two_game_series(('Flatbread', 'DrWinkly'), df, elo_map, plot=True)

def simulate_season(games_df, elo_map, matchups, current_standings, games_per_matchup=2, num_simulations=10000, elo_scaling=200):
    rw_league_player_names = list(current_standings.index)
    standings_tracker = {team: np.zeros(len(rw_league_player_names)) for team in rw_league_player_names}

    unique_teams = set([team for game in matchups for team in game])
    team_name_map = {team: process.extractOne(team, rw_league_player_names)[0] for team in unique_teams}

    for _ in range(num_simulations):
        simulated_records = current_standings.copy()  # Start with actual standings

        # Simulate all remaining games
        print(_)
        for game in matchups:
            team1 = team_name_map[game[0]]
            team2 = team_name_map[game[1]]
            for i in range(games_per_matchup):
                run_differential = monte_carlo((team1, team2), games_df, elo_map, num_simulations=1, elo_scaling=elo_scaling)
                run_differential = int(run_differential[0])
                simulated_records.loc[team1, 'Games Played'] += 1
                simulated_records.loc[team2, 'Games Played'] += 1
                simulated_records.loc[team1, 'Run Differential'] += run_differential
                simulated_records.loc[team2, 'Run Differential'] -= run_differential
                if run_differential > 0:
                    simulated_records.loc[team1, 'Wins'] += 1
                    simulated_records.loc[team2, 'Losses'] += 1
                else:
                    simulated_records.loc[team2, 'Wins'] += 1
                    simulated_records.loc[team1, 'Losses'] += 1

        sorted_teams = simulated_records.sort_values(by=["Wins", "Run Differential"], ascending=[False, False])

        top_division = sorted_teams.iloc[0]["Division"]  # First place player's division
        # Find the highest-ranked team from the other division
        other_division_team = sorted_teams[sorted_teams["Division"] != top_division].iloc[0]  # Extract row

        # Reorder the standings
        sorted_teams = pd.concat([
            sorted_teams.iloc[:1],  # Keep the top team
            sorted_teams.loc[[other_division_team.name]],  # Move the top team from the other division to 2nd place
            sorted_teams.iloc[1:].drop(index=other_division_team.name)  # Remove them from their old position & keep the rest
        ])

        # Track frequency of each team's final placement
        for rank, team in enumerate(list(sorted_teams.index)):
            standings_tracker[team][rank] += 1

    # Convert frequencies to probabilities
    standings_df = pd.DataFrame(standings_tracker) / num_simulations * 100
    standings_df.index += 1  # Rank starts from 1
    return standings_df

# current_standings = generate_stadings_df(df, NNLS7)
# print(current_standings)
# simulation = simulate_season(df, elo_map, remaining_matchups, current_standings)
# simulation.to_excel("season_simulation_results.xlsx", index=True)


def elo_run_diff_regression(games_df):
    manager = APIManager()
    elo_map = pd.DataFrame(game_mode_ladder(manager, 'NNL Season 7')).T['rating'].to_dict()
    print(elo_map)

    games_df['winner_result_elo'] = games_df['winner_user'].map(elo_map)
    games_df['loser_result_elo'] = games_df['loser_user'].map(elo_map)

    regression_df = games_df[['winner_user', 'winner_result_elo', 'winner_score', 'loser_user', 'loser_result_elo', 'loser_score']].copy()
    regression_df['elo_diff'] = regression_df['winner_result_elo'] - regression_df['loser_result_elo']
    regression_df['run_diff'] = regression_df['winner_score'] - regression_df['loser_score']
    print(regression_df)

    # Independent (X) and dependent (y) variables
    X = regression_df['elo_diff']
    y = regression_df['run_diff']

    # Add constant for intercept in regression
    X = sm.add_constant(X)

    # Perform regression
    model = sm.OLS(y, X).fit()
    regression_df['predicted_run_diff'] = model.predict(X)

    # Print summary of regression results
    print(model.summary())

    # Plot results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=regression_df['elo_diff'], y=regression_df['run_diff'], alpha=0.6, label='Actual')
    sns.lineplot(x=regression_df['elo_diff'], y=regression_df['predicted_run_diff'], color='red', label='Regression Line')

    plt.xlabel("ELO Difference (Winner - Loser)")
    plt.ylabel("Run Differential (Winner - Loser)")
    plt.title("ELO Difference vs Run Differential")
    plt.legend()
    plt.show()
    
def negative_binomial():
    # Extract runs scored
    away_runs = df['away_score'].values
    home_runs = df['home_score'].values
    all_runs = np.concatenate([away_runs, home_runs])

    # Plot histogram of actual data
    plt.figure(figsize=(10,5))
    sns.histplot(all_runs, bins=range(0, max(all_runs) + 1), kde=True, stat="density", label="Empirical Data")
    plt.xlabel("Runs Scored Per Game")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # Fit Poisson distribution
    lambda_poisson = np.mean(all_runs)
    poisson_fit = stats.poisson.pmf(np.arange(0, max(all_runs) + 1), mu=lambda_poisson)

    # Fit Negative Binomial distribution
    mean_runs = np.mean(all_runs)
    var_runs = np.var(all_runs)
    p = mean_runs / var_runs
    r = mean_runs**2 / (var_runs - mean_runs)
    nbinom_fit = stats.nbinom.pmf(np.arange(0, max(all_runs) + 1), r, p)

    # Plot fitted distributions
    plt.figure(figsize=(10,5))
    sns.histplot(all_runs, bins=range(0, max(all_runs) + 1), kde=False, stat="density", label="Empirical Data")
    plt.plot(np.arange(0, max(all_runs) + 1), poisson_fit, 'r-', label="Poisson Fit")
    plt.plot(np.arange(0, max(all_runs) + 1), nbinom_fit, 'g-', label="Negative Binomial Fit")
    plt.xlabel("Runs Scored Per Game")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # Goodness-of-Fit Tests
    ks_poisson = stats.kstest(all_runs, 'poisson', args=(lambda_poisson,))
    ks_nbinom = stats.kstest(all_runs, 'nbinom', args=(r, p))

    print(f"Kolmogorov-Smirnov Test (Poisson): {ks_poisson}")
    print(f"Kolmogorov-Smirnov Test (Negative Binomial): {ks_nbinom}")

    # AIC/BIC Comparisons
    poisson_log_likelihood = np.sum(stats.poisson.logpmf(all_runs, lambda_poisson))
    nbinom_log_likelihood = np.sum(stats.nbinom.logpmf(all_runs, r, p))

    poisson_aic = -2 * poisson_log_likelihood + 2 * 1  # 1 parameter (lambda)
    nbinom_aic = -2 * nbinom_log_likelihood + 2 * 2  # 2 parameters (r, p)

    print(f"AIC (Poisson): {poisson_aic}")
    print(f"AIC (Negative Binomial): {nbinom_aic}")

def run_differential_normal():
    df['run_differential'] = df['home_score'] - df['away_score']

    # Plot the histogram with bins equal to 1 run
    plt.figure(figsize=(10, 5))

    # Set bins to be from the minimum run differential to the maximum, with a bin width of 1
    sns.histplot(df['run_differential'], kde=True, stat="density", color="blue", bins=range(int(df['run_differential'].min()), int(df['run_differential'].max()) + 2, 1))

    plt.title("Run Differential Distribution")
    plt.xlabel("Run Differential")
    plt.ylabel("Density")
    plt.show()

    # Q-Q plot to visually assess normality
    plt.figure(figsize=(8, 8))
    stats.probplot(df['run_differential'], dist="norm", plot=plt)
    plt.title("Q-Q Plot for Run Differential")
    plt.show()

def validate_model(games_df):
    # Add new columns to games_df
    games_df["predicted_differential"] = np.nan
    games_df["home_win_prob"] = np.nan
    games_df["away_win_prob"] = np.nan
    games_df["correct_prediction"] = False
    games_df["winner_is_home"] = False

    for idx, row in games_df.iterrows():
        user1, user2 = row['home_user'], row['away_user']
        
        # Get predicted run differential distribution
        r1, p1, r2, p2 = matchup_run_distirbution(user1, user2, games_df)
        simulated_differential = monte_carlo((user1, user2), games_df, elo_map, num_simulations=100000)

        # Compute win probability
        home_win_prob = np.mean(simulated_differential > 0)
        away_win_prob = 1 - home_win_prob  # Since it's a two-player game

        # Determine predicted winner
        predicted_winner = user1 if home_win_prob > away_win_prob else user2
        predicted_differential = np.mean(simulated_differential)

        # Store results in games_df
        games_df.at[idx, "predicted_differential"] = predicted_differential
        games_df.at[idx, "home_win_prob"] = home_win_prob
        games_df.at[idx, "away_win_prob"] = away_win_prob

        # Compare with actual winner
        actual_differential = row["home_score"] - row["away_score"]
        actual_winner = user1 if actual_differential > 0 else user2
        games_df.at[idx, "correct_prediction"] = predicted_winner == actual_winner

        # Assign predicted win probability of the winner
        if row['winner_user'] == user1:
            games_df.at[idx, "winner_win_prob"] = home_win_prob
            games_df.at[idx, "winner_is_home"] = True
        else:
            games_df.at[idx, "winner_win_prob"] = away_win_prob
            games_df.at[idx, "winner_is_home"] = False

    print(games_df)

    # Set plot style
    sns.set_style("whitegrid")

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=games_df["predicted_differential"], y=games_df["home_score"] - games_df["away_score"], alpha=0.7)

    # Add a reference line (y = x) for perfect predictions
    plt.axline((0, 0), slope=1, color='red', linestyle="dashed", label="Perfect Prediction")

    # Labels and title
    plt.xlabel("Predicted Run Differential")
    plt.ylabel("Actual Run Differential")
    plt.title("Predicted vs. Actual Run Differential")

    # Show correlation coefficient
    correlation = games_df["predicted_differential"].corr(games_df["home_score"] - games_df["away_score"])
    plt.figtext(0.15, 0.85, f"Correlation: {correlation:.2f}", fontsize=12, color="blue")

    X = games_df["predicted_differential"]
    y = games_df["home_score"] - games_df["away_score"]

    # Add a constant to the independent variable (for the intercept in the regression model)
    X = sm.add_constant(X)
    # Fit the model
    model = sm.OLS(y, X).fit()

    # Print out the summary
    print(model.summary())

    plt.legend()
    plt.show()

    # Compute prediction error
    games_df["prediction_error"] = games_df["predicted_differential"] - (games_df["home_score"] - games_df["away_score"])

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(games_df["prediction_error"], bins=20, kde=True, color="blue", alpha=0.7)

    # Add a vertical line at 0 (perfect prediction)
    plt.axvline(0, color="red", linestyle="dashed", label="Perfect Prediction (Error = 0)")

    # Labels and title
    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")

    # Show mean error
    mean_error = games_df["prediction_error"].mean()
    plt.figtext(0.15, 0.85, f"Mean Error: {mean_error:.2f}", fontsize=12, color="blue")

    plt.legend()
    plt.show()

    # Step 1: Create bins (0.0–0.1, 0.1–0.2, ..., 0.9–1.0)
    games_df['win_prob_bin'] = np.round(games_df['home_win_prob'], 1)  # Bin into 0.1 intervals

    # Step 2: Calculate actual win rate in each bin
    calibration_df = games_df.groupby('win_prob_bin').agg(
        predicted_win_rate=('win_prob_bin', 'mean'),  # Average predicted probability per bin
        actual_win_rate=('winner_is_home', 'mean'),  # Actual win % in that bin
        count=('correct_prediction', 'count')  # Number of games per bin
    ).reset_index()

    # Step 3: Plot Calibration Curve
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=calibration_df['predicted_win_rate'], y=calibration_df['actual_win_rate'], marker='o', label='Actual')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")  # Reference diagonal
    plt.xlabel("Predicted Win Probability")
    plt.ylabel("Actual Win Rate")
    plt.title("Win Probability Calibration")
    plt.legend()
    plt.show()

    # Print summary statistics
    print(calibration_df)

    # Filter games where the home team's predicted win probability is less than 20%
    low_win_prob_games = games_df[(games_df['home_win_prob'] < 0.2) | (games_df['away_win_prob'] < 0.25)]

    # Print the filtered dataframe
    print(low_win_prob_games[['home_user', 'home_score', 'away_user', 'away_score', 'home_win_prob', 'away_win_prob', 'win_prob_bin', 'winner_win_prob', 'predicted_differential', 'correct_prediction']])

validate_model(df)