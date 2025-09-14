import pandas as pd
import json
from fuzzywuzzy import process
from pathlib import Path
from pyRio import endpoint_handling, web_functions, api_manager, web_caching


class LeagueData():
    def __init__(self, tag):
        self.tag = tag
        self.path = Path(__file__).parent / "LeagueData" / tag

        with open(self.path / 'teams.json', 'r') as json_file:
            data = json.load(json_file)
        
        self.player_data = data['Players']
        self.league_name = data['League Name']
        self.font = data['Font']
        self.matchups = data['Matchups']

        self.json_path = self.path / "edit_games.json"

def get_web_games(settings: LeagueData, edit_mode_file=True):
    manager = api_manager.APIManager()
    api_response = web_functions.games_endpoint(manager, settings.tag, limit_games=500)
    web_cache = web_caching.CompleterCache(manager)
    games_df = endpoint_handling.games_endpoint(api_response, web_cache)
    
    if edit_mode_file:
        with open(settings.json_path, 'r') as f:
            data = json.load(f)
        if data.get('Add'):
            add_games_df = pd.DataFrame(data['Add'])
            add_games_df['date_time_start'] = pd.to_datetime(add_games_df['date_time_start'], utc=True, unit='ms').dt.tz_convert('US/Eastern')
            add_games_df['date_time_end'] = pd.to_datetime(add_games_df['date_time_end'], utc=True, unit='ms').dt.tz_convert('US/Eastern')
            games_df = pd.concat([games_df, add_games_df], ignore_index=True)
        if data.get('Delete'):
            games_df = games_df[~games_df['game_id'].isin(data['Delete'])]

    unique_names = list(settings.player_data.keys())
    print(unique_names)

    def match_player_to_team(player_name):
        best_match = process.extractOne(player_name, unique_names)
        if best_match:
            # Return the team information corresponding to the best match
            return best_match[0]
        return None
    
    games_df["Week"] = ""

    for week, matches_list in settings.matchups.items():
        print(matches_list)
        for match in matches_list:
            #print(match)
            team1 = match_player_to_team(match[0].lower())
            team2 = match_player_to_team(match[1].lower())
            #print(team1, team2)

            mask = ((games_df["away_user"].str.lower() == team1) & (games_df["home_user"].str.lower() == team2)) | \
                    ((games_df["away_user"].str.lower() == team2) & (games_df["home_user"].str.lower() == team1))
            
            games_df.loc[mask, "Week"] = week

    return games_df