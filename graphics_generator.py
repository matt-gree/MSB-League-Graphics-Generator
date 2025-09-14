from NNL_config import NNLConfig
from games_image_generator import standings_generator, scorecard_generator, make_weekly_graphics
from league_data_handler import get_web_games

# Pick season here
config = NNLConfig("NNLSeason8")

games_df = get_web_games(config.league, edit_mode_file=True)
print(games_df)

# Standings
standings_generator(games_df, settings=config.standings_settings)
standings_generator(
    games_df,
    config.standings_insert_settings,
    size_input="weekly graphic",
    title="Standings",
    output_name="standingsInsert.png"
)

# Scorecards & weekly graphics
# scorecard_generator(games_df, config.default_settings, limit_games=4, size_input="large")
make_weekly_graphics(config.default_settings, games_df, "")

# Optional: player-specific scorecard
# scorecard_generator(games_df, config.default_settings, size_input="large", player_filter="Cezarito", output_name="Cezarito.png")