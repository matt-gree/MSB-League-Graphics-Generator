import pyrio.stat_file_parser as sfp
import league_data_handler as ldh
from pyrio.lookup import LookupDicts
from datetime import datetime
import pandas as pd
import numpy as np
import os
import itertools
import json
import re

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

PATH_TO_STATS = ['LeagueData/NNLSeason5/StatFiles', 'LeagueData/NNLSeason6/StatFiles', 'LeagueData/NNLSeason7/StatFiles']

def create_state_string(event: sfp.EventObj):
    table_string = ''    
    for k in range(1,4):
        if event.bool_runner_on_base(k):
            table_string += f'{k}'
        else: table_string += '-'
    
    table_string += f'_{event.outs()}'
    return table_string

def create_run_expectancy_matrix(paths_to_stats):
    base_states = [
        '---',
        '1--',
        '-2-',
        '--3',
        '12-',
        '1-3',
        '-23',
        '123'  
    ]

    outs = [0, 1, 2]
    states = [f'{base}_{out}' for base, out in itertools.product(base_states, outs)]
    run_expectancy_matrix = pd.DataFrame(0, index=states, columns=['Runs', 'Occurances'])

    print(run_expectancy_matrix)
    for path in paths_to_stats:
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r') as f:
                stats = sfp.StatObj(json.load(f))
            events = sfp.EventSearch(stats)

            first_pitch_of_AB_events = events.firstPitchOfABEvents()

            counted_abs = 0
            for i in range(1,stats.inningsPlayed()+1):
                for j in range(0,2):
                    current_inning_events = events.inningEvents(i)
                    half_inning_events = events.halfInningEvents(j)

                    current_half_inning_events = current_inning_events.intersection(half_inning_events)
                    if not current_half_inning_events:
                        continue 
                    last_half_inning_event = sfp.EventObj(stats, max(current_half_inning_events))
                    half_inning_AB_start_events = current_half_inning_events.intersection(first_pitch_of_AB_events)

                    for event in half_inning_AB_start_events:
                        ab_event = sfp.EventObj(stats, event)

                        table_string = create_state_string(ab_event)

                        runs_scored_after_pa = last_half_inning_event.score(j) - ab_event.score(j)

                        run_expectancy_matrix.loc[table_string, 'Runs'] += runs_scored_after_pa
                        run_expectancy_matrix.loc[table_string, 'Occurances'] += 1

    run_expectancy_matrix['RE'] = run_expectancy_matrix['Runs'] / run_expectancy_matrix['Occurances']
    
    return run_expectancy_matrix

    run_expectancy_matrix = run_expectancy_matrix.reset_index()
    run_expectancy_matrix[['BaseState', 'Outs']] = run_expectancy_matrix['index'].str.split('_', expand=True)
    run_expectancy_matrix['Outs'] = run_expectancy_matrix['Outs'].astype(int)

    # Now pivot to get the matrix format
    rem = run_expectancy_matrix.pivot(index='BaseState', columns='Outs', values='RE')

    # Optional: Sort rows in logical base-running order
    base_order = ['---', '1--', '-2-', '--3', '12-', '1-3', '-23', '123']
    rem = rem.reindex(base_order)

    print(rem)

def create_linear_weights(paths_to_stats, rem):
    wOBA_df = pd.DataFrame(
        0.0,  # initialize with 0.0 (or use np.nan)
        index=['Single', 'Double', 'Triple', 'HR', 'Walk', 'Out', 'Error'],
        columns=['RE Change', 'Count']
    )
    if not isinstance(paths_to_stats, list):
        paths_to_stats = [paths_to_stats]

    for path in paths_to_stats:
        print(path)
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r') as f:
                stats = sfp.StatObj(json.load(f))
            events = sfp.EventSearch(stats)

            single_events = events.hitResultEvents(1)
            double_events = events.hitResultEvents(2)
            triple_events = events.hitResultEvents(3)
            hr_events = events.hitResultEvents(4)
            bb_events = events.walkResultEvents()
            out_events = events.allOutResultEvents()
            error_events = events.chemErrorResultEvents().union(events.inputErrorResultEvents())

            wOBA_events_dict = {
                'Single': single_events,
                'Double': double_events,
                'Triple': triple_events,
                'HR': hr_events,
                'Walk': bb_events,
                'Out': out_events,
                'Error': error_events
            }

            for event_type, event_nums in wOBA_events_dict.items():
                for event in event_nums:
                    event_of_result = sfp.EventObj(stats, event)
                    event_halfinning = event_of_result.half_inning()
                    current_state = create_state_string(event_of_result)
                    re_before = rem.loc[current_state, 'RE']

                    if event == stats.final_event():
                        # Ignore walkoff events, but handle last events cleanly
                        runs_scored = stats.score(event_halfinning) - event_of_result.score(event_halfinning)
                        if runs_scored == 0:
                            wOBA_df.loc[event_type, 'RE Change'] -= rem.loc[current_state, 'RE']
                            wOBA_df.loc[event_type, 'Count'] += 1
                            
                        continue

                    event_after_result = sfp.EventObj(stats, event+1)
                                                    
                    after_event_halfinning = event_after_result.half_inning()

                    if event_halfinning != after_event_halfinning:
                        re_after = 0
                    else:
                        new_state = create_state_string(event_after_result)
                        re_after = rem.loc[new_state, 'RE']

                    runs_scored = event_after_result.score(event_halfinning) - event_of_result.score(event_halfinning)

                    change_in_re = (re_after-re_before)+runs_scored

                    wOBA_df.loc[event_type, 'RE Change'] += change_in_re
                    wOBA_df.loc[event_type, 'Count'] += 1

    wOBA_df['Weight'] = wOBA_df['RE Change'] / wOBA_df['Count']
    print(wOBA_df)

    scaled_wOBA_df = wOBA_df['Weight'] + abs(wOBA_df.loc['Out', 'Weight'])
    print(scaled_wOBA_df)


    return scaled_wOBA_df


def calculate_game_stats(paths_to_stats):
    if not isinstance(paths_to_stats, list):
        paths_to_stats = [paths_to_stats]

    PAs_dict = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        "Games": 0
    }

    stadium_games_dict = {
        'Mario Stadium': 0,
        'Peach Garden': 0,
        'Wario Palace': 0,
        'Yoshi Park': 0,
        'DK Jungle': 0,
        'Bowser Castle': 0
    }

    star_swings_dict = {
        'Mario Stadium': 0,
        'Peach Garden': 0,
        'Wario Palace': 0,
        'Yoshi Park': 0,
        'DK Jungle': 0,
        'Bowser Castle': 0
    }

    for path in paths_to_stats:
        for filename in os.listdir(path):
            if "crash" in filename:
                continue
            filepath = os.path.join(path, filename)
            with open(filepath, 'r') as f:
                stats = sfp.StatObj(json.load(f))
            if stats.inningsPlayed() != 7:
                continue
            PAs_dict["Games"] += 1
            stadium_games_dict[stats.stadium()] += 1
            events = sfp.EventSearch(stats)
            PAs = events.firstPitchOfABEvents()
            for i in range(2):
                star_swings_dict[stats.stadium()] += stats.starHitsUsed(i)
            for eventNum in PAs:
                event = sfp.EventObj(stats, eventNum)
                PAs_dict[event.batter_roster_loc()] += 1

    return PAs_dict, stadium_games_dict, star_swings_dict

print(calculate_game_stats(PATH_TO_STATS))

def calculate_character_woba_wrc(paths_to_stats, weights, by_player=False, split_star=True, min_star_pa=40):
    if by_player:
        index = pd.MultiIndex.from_tuples([], names=["Player", "Character"])
    else:
        index = pd.Index([], name="Character")

    character_wOBA_df = pd.DataFrame(
        0.0,
        index=index,
        columns=['Single', 'Double', 'Triple', 'HR', 'Walk', 'PA', 'Error', 'Out']
    )

    total_runs = 0

    if not isinstance(paths_to_stats, list):
        paths_to_stats = [paths_to_stats]

    for path in paths_to_stats:
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r') as f:
                stats = sfp.StatObj(json.load(f))
            events = sfp.EventSearch(stats)

            total_runs += stats.score(0) + stats.score(1)

            result_events = set().union(
                events.hitResultEvents(1),
                events.hitResultEvents(2),
                events.hitResultEvents(3),
                events.hitResultEvents(4),
                events.walkResultEvents(),
                events.allOutResultEvents(),
            )

            for event_num in result_events:
                ab = sfp.EventObj(stats, event_num)
                result = ab.result_of_AB()
                batter = ab.batter()
                player = stats.player(ab.batting_team())

                # Adjust character labels
                if (batter == 'Bowser') & (ab.batter_hand() == 'Right'):
                    batter = 'Bowser(R)'
                if (batter in ['Toad(B)', 'Toad(G)', 'Toad(P)', 'Toad(Y)']) and (stats.endDate() < datetime(2024, 6, 1)):
                    batter += '_unfixed'

                # Tag star swings separately only if split_star=True
                if split_star and ab.type_of_swing() == 'Star' and ab.team_stars(ab.half_inning()) > 0:
                    batter += '_star'

                # Index key depends on mode
                if by_player:
                    idx = (player, batter)
                else:
                    idx = batter

                if idx not in character_wOBA_df.index:
                    character_wOBA_df.loc[idx, :] = [0.0] * len(character_wOBA_df.columns)

                character_wOBA_df.loc[idx, 'PA'] += 1

                if result in ['Single', 'Double', 'Triple', 'HR']:
                    character_wOBA_df.loc[idx, result] += 1
                elif result in ['Walk (BB)', 'Walk (HBP)']:
                    character_wOBA_df.loc[idx, 'Walk'] += 1
                elif result in ['Strikeout', 'Out', 'Caught', 'Caught line-drive',
                                'SacFly', 'Ground ball double Play', 'Foul catch']:
                    character_wOBA_df.loc[idx, 'Out'] += 1

    # Calculate wOBA for each row
    character_wOBA_df['wOBA'] = (
        (weights['Single'] * character_wOBA_df['Single'] +
        weights['Double'] * character_wOBA_df['Double'] +
        weights['Triple'] * character_wOBA_df['Triple'] +
        weights['HR'] * character_wOBA_df['HR']) /
        (character_wOBA_df['PA'] - character_wOBA_df['Walk'])
    )

    # --- NEW: calculate league-relative wRC/PA ---
    league_PA = character_wOBA_df['PA'].sum()
    league_wOBA = (character_wOBA_df['wOBA'] * character_wOBA_df['PA']).sum() / league_PA
    league_wRC_per_PA = total_runs / league_PA

    character_wOBA_df['wRC/PA'] = (
        (character_wOBA_df['wOBA'] - league_wOBA) + league_wRC_per_PA
    )

    if not split_star:
        # No split: just return plain wOBA table
        return character_wOBA_df, total_runs

    # --- Collapse star and non-star into one row per character ---
    records = []
    for idx in character_wOBA_df.index:
        char = idx[1] if by_player else idx
        is_star = "_star" in char
        base_char = char.replace("_star", "") if is_star else char

        pa = character_wOBA_df.loc[idx, "PA"]
        woba_val = character_wOBA_df.loc[idx, "wOBA"]
        wrc_pa_val = character_wOBA_df.loc[idx, "wRC/PA"]

        if is_star:
            # Always record PA
            records.append((base_char, "PA_star", pa))
            # wOBA_star
            if pa >= min_star_pa:
                records.append((base_char, "wOBA_star", woba_val))
                records.append((base_char, "wRC/PA_star", wrc_pa_val))
            else:
                records.append((base_char, "wOBA_star", np.nan))
                records.append((base_char, "wRC/PA_star", np.nan))
        else:
            records.append((base_char, "PA_nonstar", pa))
            records.append((base_char, "wOBA_nonstar", woba_val))
            records.append((base_char, "wRC/PA_nonstar", wrc_pa_val))

    wide = (
        pd.DataFrame(records, columns=["Character", "Metric", "Value"])
        .pivot(index="Character", columns="Metric", values="Value")
    )

    return wide, total_runs
    
rem = create_run_expectancy_matrix(PATH_TO_STATS)

print(rem)
weights = create_linear_weights(PATH_TO_STATS, rem)
woba, league_runs = calculate_character_woba_wrc(PATH_TO_STATS, weights, split_star=False)
woba_star, league_runs = calculate_character_woba_wrc(PATH_TO_STATS, weights, split_star=True)

print(woba)
print(woba_star)

prices_avg = {
    "Baby Luigi": 2.83,
    "Baby Mario": 2.17,
    "Birdo": 13.00,
    "Boo": 23.25,
    "Bro(B)": 32.83,
    "Bowser": 37.17,
    "Bowser(R)": 20.33,
    "Bowser Castle": 1.25,
    "Bowser Jr": 7.29,
    "Daisy": 5.93,
    "Diddy": 2.82,
    "Dixie": 4.60,
    "DK Jungle": 4.83,
    "DK": 33.85,
    "Dry Bones(B)": 2.33,
    "Dry Bones(Gy)": 2.33,
    "Dry Bones(G)": 7.83,
    "Dry Bones(R)": 3.00,
    "Bro(F)": 27.17,
    "Goomba": 1.17,
    "Bro(H)": 30.67,
    "King Boo": 33.83,
    "Paratroopa(G)": 6.85,
    "Paratroopa(R)": 2.25,
    "Koopa(G)": 1.27,
    "Koopa(R)": 2.00,
    "Luigi": 10.99,
    "Magikoopa(B)": 22.67,
    "Magikoopa(G)": 22.83,
    "Magikoopa(R)": 23.83,
    "Magikoopa(Y)": 22.17,
    "Mario": 10.71,
    "Mario Stadium": 4.08,
    "Monty": 2.57,
    "Noki(B)": 3.42,
    "Noki(G)": 13.33,
    "Noki(R)": 4.25,
    "Paragoomba": 1.17,
    "Peach": 11.48,
    "Peach Garden": 1.17,
    "Petey": 21.32,
    "Pianta(B)": 14.50,
    "Pianta(R)": 23.67,
    "Pianta(Y)": 14.50,
    "Shy Guy(Bk)": 2.00,
    "Shy Guy(B)": 1.50,
    "Shy Guy(G)": 3.17,
    "Shy Guy(R)": 2.93,
    "Shy Guy(Y)": 2.50,
    "Toad(B)": 1.83,
    "Toad(G)": 1.33,
    "Toad(P)": 2.10,
    "Toad(R)": 4.43,
    "Toad(Y)": 1.67,
    "Toadette": 7.27,
    "Toadsworth": 11.50,
    "Waluigi": 16.67,
    "Wario": 9.77,
    "Wario Palace": 4.18,
    "Yoshi": 18.44,
    "Yoshi Park": 1.43
}

avg_prices = pd.Series(prices_avg)
woba_per_dollar = woba['wOBA'] / avg_prices
wrc= woba[['wRC/PA', 'PA']]
wrc = wrc[wrc["PA"]>=40]

woba_split, runs = calculate_character_woba_wrc(PATH_TO_STATS[-1], weights, by_player=True, split_star=False)

# Restrict to just MattGree
matt_df = woba_split.xs("MattGree", level="Player").copy()

print(matt_df[["wRC/PA", "PA"]])

with pd.option_context('display.max_rows', None):
    print(woba_per_dollar.sort_values(ascending=False))
    print(wrc.sort_values(by="wRC/PA", ascending=False))

def optimize_lineup_with_stars_greedy(woba_df, avg_prices, signed_players=None, budget=120,
                                      lineup_size=9, projected_PA=79, season_star_swing=109,
                                      scaling_top=1.0, scaling_bottom=0.85):
    """
    Greedy lineup optimizer that guarantees exactly lineup_size players,
    allocates star swings proportionally, and applies PA scaling.

    Parameters
    ----------
    woba_df : pd.DataFrame
        Must contain 'wRC/PA_nonstar' and 'wRC/PA_star'.
    avg_prices : dict
        {player: price}.
    signed_players : dict, optional
        {player: price} for already-signed players.
    budget : float
        Total budget.
    lineup_size : int
        Number of players in lineup.
    projected_PA : int
        Baseline PA per player.
    season_star_swing : int
        Total star swings for the team.
    scaling_top : float
        Top lineup PA scaling.
    scaling_bottom : float
        Bottom lineup PA scaling.

    Returns
    -------
    full_roster : dict
        {player: price}.
    roster_df : pd.DataFrame
        Detailed per-player contributions.
    """

    signed_players = signed_players or {}
    remaining_budget = budget - sum(signed_players.values())

    # --------------------------
    # Step 1: candidate scoring (value per dollar)
    candidates = woba_df.copy()
    candidates = candidates[candidates.index.isin(avg_prices.keys())]
    candidates = candidates.drop(index=signed_players.keys(), errors='ignore')

    # Approximate combined value per PA including star swings
    # Star factor normalized by max star contribution
    max_star = candidates['wRC/PA_star'].fillna(0).max()
    star_factor = candidates['wRC/PA_star'].fillna(0) / max_star if max_star > 0 else 0
    candidates['ValuePerPA'] = candidates['wRC/PA_nonstar'].fillna(0) + star_factor

    # Value per dollar
    candidates['ValuePerDollar'] = candidates['ValuePerPA'] / candidates.index.map(avg_prices)

    # --------------------------
    # Step 2: greedy pick until lineup_size
    chosen_players = {}
    for p, row in candidates.sort_values('ValuePerDollar', ascending=False).iterrows():
        price = avg_prices[p]
        if len(chosen_players) + len(signed_players) >= lineup_size:
            break
        if price <= remaining_budget:
            chosen_players[p] = price
            remaining_budget -= price

    # Fill remaining spots with cheapest players if needed
    if len(chosen_players) + len(signed_players) < lineup_size:
        remaining_slots = lineup_size - len(chosen_players) - len(signed_players)
        remaining_candidates = candidates.drop(index=chosen_players.keys())
        remaining_candidates = remaining_candidates.sort_values('ValuePerDollar', ascending=False)
        for p, row in remaining_candidates.iterrows():
            if remaining_slots <= 0:
                break
            price = avg_prices[p]
            if price <= remaining_budget:
                chosen_players[p] = price
                remaining_budget -= price
                remaining_slots -= 1

    full_roster = {**signed_players, **chosen_players}
    roster_df = woba_df.loc[full_roster.keys()].copy()
    roster_df['Price'] = [full_roster[p] for p in roster_df.index]

    # --------------------------
    # Step 3: allocate star swings proportionally
    star_total = roster_df['wRC/PA_star'].fillna(0).sum()
    if star_total > 0:
        roster_df['AllocatedStarPA'] = roster_df['wRC/PA_star'].fillna(0) / star_total * season_star_swing
    else:
        roster_df['AllocatedStarPA'] = 0.0

    # Remaining PA
    roster_df['NonStarPA'] = projected_PA - roster_df['AllocatedStarPA']

    # Total expected wRC per player
    roster_df['ExpectedWRC'] = (
        roster_df['NonStarPA'] * roster_df['wRC/PA_nonstar'].fillna(0) +
        roster_df['AllocatedStarPA'] * roster_df['wRC/PA_star'].fillna(0)
    )

    # --------------------------
    # Step 4: apply lineup PA scaling
    order = roster_df.sort_values('ExpectedWRC', ascending=False)
    scaling = np.linspace(scaling_top, scaling_bottom, lineup_size)
    order['ScaledWRC'] = order['ExpectedWRC'].values * scaling

    # --------------------------
    # Step 5: print results
    print("Optimal Lineup (with PA scaling and star swings):")
    for i, (p, row) in enumerate(order.iterrows()):
        print(f"#{i+1}: {p:12s} | Cost: ${row['Price']:5.2f} | "
              f"Raw wRC: {row['ExpectedWRC']:7.2f} | Scaled: {row['ScaledWRC']:7.2f} | "
              f"Star PA: {row['AllocatedStarPA']:5.1f} | Non-Star PA: {row['NonStarPA']:5.1f}")

    total_cost = order['Price'].sum()
    total_raw = order['ExpectedWRC'].sum()
    total_scaled = order['ScaledWRC'].sum()

    print(f"\nTotal cost: ${total_cost:.2f} / {budget}")
    print(f"Raw projected team wRC: {total_raw:.2f}")
    print(f"Scaled projected team wRC: {total_scaled:.2f}")

    return full_roster, order

signed = {}

def team_top9_wrc_scaled(teams, woba_df, woba_combined_df, season_PA=79, season_star_swing=109):
    """
    Compute team totals by summing top 9 characters' wRC contributions with PA scaling
    (from 1.0 down to 0.78) and proportional star swing allocation.

    Parameters
    ----------
    teams : dict
        {"Owner": ["Char1", ..., "Char10"], ...}
    woba_df : DataFrame
        Must have columns: "wRC/PA_nonstar", "wRC/PA_star",
                           "PA_nonstar", "PA_star",
                           optionally "wOBA_nonstar", "wOBA_star".
    season_PA : int
        Baseline PA per lineup slot before scaling.
    season_star_swing : int
        Total star PAs available for the team to distribute.
    """
    results = []

    for owner, roster in teams.items():
        df = pd.DataFrame({
            "Character": roster,
            "wRC_non": [woba_df.at[char, "wRC/PA_nonstar"] if char in woba_df.index else np.nan for char in roster],
            "wRC_star": [woba_df.at[char, "wRC/PA_star"] if char in woba_df.index else np.nan for char in roster],
            "PA_non":  [woba_df.at[char, "PA_nonstar"] if char in woba_df.index else np.nan for char in roster],
            "PA_star": [woba_df.at[char, "PA_star"] if char in woba_df.index else np.nan for char in roster],
            "wOBA_non": [woba_df.at[char, "wOBA_nonstar"] if "wOBA_nonstar" in woba_df.columns and char in woba_df.index else np.nan for char in roster],
            "wOBA_star": [woba_df.at[char, "wOBA_star"] if "wOBA_star" in woba_df.columns and char in woba_df.index else np.nan for char in roster],
        }).dropna(subset=["wRC_non"])  # need at least non-star data

        # Map combined wRC/PA to team roster
        df["wRC_combined"] = [woba_combined_df.at[char, "wRC/PA"] if char in woba_combined_df.index else np.nan for char in df["Character"]]

        # Drop lowest one if roster > 9
        dropped = None
        if len(df) > 9:
            df = df.sort_values("wRC_combined", ascending=False, kind="mergesort")
            dropped = df.iloc[-1]["Character"]
            df = df.iloc[:9]

        # PA scaling: 1.0 -> 0.78
        scaling = np.linspace(1.0, 0.78, len(df))
        df["Scaled_PA"] = season_PA * scaling

        # Star allocation proportional to star wRC/PA * eligibility
        star_power = df["wRC_star"].fillna(0).sum()
        if star_power > 0:
            df["StarShare"] = df["wRC_star"].fillna(0) / star_power
        else:
            df["StarShare"] = 0.0

        # Assign star PAs
        df["Star_PA"] = df["StarShare"] * season_star_swing

        # Non-star PAs = remaining scaled PAs
        df["NonStar_PA"] = df["Scaled_PA"] - df["Star_PA"]
        df["NonStar_PA"] = df["NonStar_PA"].clip(lower=0)

        # Contributions (ignoring NaN safely)
        df["wRC_contrib"] = (
            df["Star_PA"] * df["wRC_star"].fillna(0) +
            df["NonStar_PA"] * df["wRC_non"].fillna(0)
        )

        total = df["wRC_contrib"].sum()
        results.append({
            "Owner": owner,
            "Scaled_wRC_total": total,
            "Dropped": dropped
        })

    # Leaderboard
    scores_df = pd.DataFrame(results).sort_values("Scaled_wRC_total", ascending=False).reset_index(drop=True)
    return scores_df

season7_player_data = ldh.LeagueData("NNLSeason7").player_data
season8_player_data = ldh.LeagueData("NNLSeason8").player_data

combined_seasons_teams = {}

for name, data in season7_player_data.items():
    combined_seasons_teams[f'{name} S7'] = data['Drafted Team']

for name, data in season8_player_data.items():
    combined_seasons_teams[f'{name} S8'] = data['Drafted Team']


print(team_top9_wrc_scaled(combined_seasons_teams, woba_star, woba))

full_roster, scaled_wRC = optimize_lineup_with_stars_greedy(
    woba_df=woba_star,
    avg_prices=prices_avg,
    signed_players=signed,
    budget=115,
    lineup_size=9
)


# --- Prepare league data ---
plot_df = pd.DataFrame({
    "wRC/PA": woba["wRC/PA"],
    "Price": avg_prices
}).dropna()

# --- Prepare personal data ---
avg_prices_named = avg_prices.copy()
avg_prices_named.name = "Price"  # assign column name for join
matt_df = matt_df.join(avg_prices_named, how="inner")  # attach prices

# --- Fit exponential curve ---
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

xdata, ydata = plot_df["wRC/PA"].values, plot_df["Price"].values
try:
    popt, _ = curve_fit(exp_func, xdata, ydata, maxfev=10000)
    xfit = np.linspace(min(xdata), max(xdata), 200)
    yfit = exp_func(xfit, *popt)
    ypred = exp_func(xdata, *popt)
    r2 = r2_score(ydata, ypred)
    fit_label = f"Exp fit (R²={r2:.3f})"
except RuntimeError:
    xfit, yfit, fit_label = [], [], "Fit failed"

# --- Plot ---
plt.figure(figsize=(10,6))

# League scatter
plt.scatter(plot_df["wRC/PA"], plot_df["Price"], alpha=0.6, label="League players")

# Your personal performance
plt.scatter(matt_df["wRC/PA"], matt_df["Price"], 
            color="red", marker="D", s=80, label="MattGree performance")

# Label your characters
for char, row in matt_df.iterrows():
    plt.text(row["wRC/PA"], row["Price"]+0.5, char, fontsize=9, ha="center", color="red")

# Plot exponential fit
if len(xfit) > 0:
    plt.plot(xfit, yfit, "k-", linewidth=2, label=fit_label)

plt.xlabel("wRC/PA")
plt.ylabel("Price ($)")
plt.title("League Prices vs Production (with MattGree overlay)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()