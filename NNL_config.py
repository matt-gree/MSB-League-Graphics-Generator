from games_image_generator import GeneratorSettings
from league_data_handler import LeagueData

class NNLConfig:
    def __init__(self, season: str):
        self.season = season
        self.league = LeagueData(season)

        # Shared font dicts
        self.standings_font = {
            'title': 96,
            'subtitle': 64,
            'team': 0,
            'player': 28,
            'score': 28
        }

        self.standings_insert_font = {
            'title': 96,
            'subtitle': 64,
            'team': 0,
            'player': 46,
            'score': 46
        }

        # Prebuild GeneratorSettings objects
        self.standings_settings = self._make_standings_settings()
        self.standings_insert_settings = self._make_standings_insert_settings()
        self.default_settings = GeneratorSettings(season)

    def _make_standings_settings(self):
        settings = GeneratorSettings(self.season, font_dict=self.standings_font)
        settings.user_box_height = 32
        settings.user_box_width = 325
        settings.drawing_x_pos = 0
        settings.drawing_y_pos = 0
        settings.logo_padding = 2
        settings.gradient_factor = 0.3
        settings.logo_player_padding = 25
        settings.user_box_score_right_padding = 30
        return settings

    def _make_standings_insert_settings(self):
        settings = GeneratorSettings(self.season, font_dict=self.standings_insert_font)
        settings.user_box_height = 57
        settings.user_box_width = 600
        settings.drawing_x_pos = 0
        settings.drawing_y_pos = 0
        settings.logo_padding = 5
        settings.gradient_factor = 0.3
        settings.logo_player_padding = 40
        settings.user_box_score_right_padding = 50
        settings.title_padding = 10
        return settings