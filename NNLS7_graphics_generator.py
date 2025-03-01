from games_image_generator import LeagueData, GeneratorSettings, make_standings_graphic, make_graphic, make_weekly_graphics, get_web_games

NNLS7 = LeagueData('NNLSeason7')

standings_font = {
    'title': 96,
    'subtitle': 64,
    'team': 0,
    'player': 28,
    'score': 28
}

NNLS7StandingsSettings = GeneratorSettings('NNLSeason7', font_dict=standings_font)
NNLS7StandingsSettings.user_box_height = 32
NNLS7StandingsSettings.user_box_width = 325
NNLS7StandingsSettings.drawing_x_pos = 0
NNLS7StandingsSettings.drawing_y_pos = 0
NNLS7StandingsSettings.logo_padding = 2
NNLS7StandingsSettings.gradient_factor = 0.3
NNLS7StandingsSettings.logo_player_padding = 25
NNLS7StandingsSettings.user_box_score_right_padding = 30

games_df = get_web_games(NNLS7, edit_mode_file=True)
make_standings_graphic(NNLS7StandingsSettings, games_df, edit_mode_file=True)

NNLS7Settings = GeneratorSettings('NNLSeason7')
make_graphic(NNLS7Settings, games_df, edit_mode_file=True)
make_weekly_graphics(NNLS7Settings, games_df, 'Week 2')