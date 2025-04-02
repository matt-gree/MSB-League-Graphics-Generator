from games_image_generator import LeagueData, GeneratorSettings, standings_generator, scorecard_generator, make_weekly_graphics, get_web_games

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

standings_insert = {
    'title': 96,
    'subtitle': 64,
    'team': 0,
    'player': 46,
    'score': 46
}

NNLS7StandingsInsert = GeneratorSettings('NNLSeason7', font_dict=standings_insert)
NNLS7StandingsInsert.user_box_height = 57
NNLS7StandingsInsert.user_box_width = 600
NNLS7StandingsInsert.drawing_x_pos = 0
NNLS7StandingsInsert.drawing_y_pos = 0
NNLS7StandingsInsert.logo_padding = 5
NNLS7StandingsInsert.gradient_factor = 0.3
NNLS7StandingsInsert.logo_player_padding = 40
NNLS7StandingsInsert.user_box_score_right_padding = 50
NNLS7StandingsInsert.title_padding = 10

games_df = get_web_games(NNLS7, edit_mode_file=True)
standings_generator(games_df, NNLS7StandingsSettings)
standings_generator(games_df, NNLS7StandingsInsert, size_input='weekly graphic', title='Standings', output_name='standingsInsert.png')

NNLS7Settings = GeneratorSettings('NNLSeason7')
scorecard_generator(games_df, NNLS7Settings, limit_games=4, size_input='large')
make_weekly_graphics(NNLS7Settings, games_df, 'Week 5')