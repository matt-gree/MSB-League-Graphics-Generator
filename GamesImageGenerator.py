import requests
from PIL import Image, ImageDraw, ImageFont, ImageColor
from datetime import datetime, timedelta
from pyRio import endpoint_handling, web_functions, api_manager, web_caching
import pandas as pd
import pytz
import json
from fuzzywuzzy import process

class GeneratorSettings():
    def __init__(self, tag, font_dict=False):
        self.tag = tag
        self.path = f'LeagueData/{tag}/'

        with open(f'{self.path}teams.json', 'r') as json_file:
            data = json.load(json_file)
        
        self.player_data = data['Players']
        self.league_name = data['League Name']
        self.font = data['Font']
        self.matchups = data['Matchups']

        font_path = f'{self.path}{self.font}'

        if font_dict:
            self.title_font = ImageFont.truetype(font_path, font_dict['title']) if font_dict['title'] else 0
            self.subtitle_font = ImageFont.truetype(font_path, font_dict['subtitle']) if font_dict['subtitle'] else 0
            self.team_font = ImageFont.truetype(font_path, font_dict['team']) if font_dict['team'] else 0
            self.player_font = ImageFont.truetype(font_path, font_dict['player']) if font_dict['player'] else 0
            self.score_font = ImageFont.truetype(font_path, font_dict['score']) if font_dict['score'] else 0
        else:
            self.title_font = ImageFont.truetype(font_path, 96)
            self.subtitle_font = ImageFont.truetype(font_path, 124)
            self.team_font = ImageFont.truetype(font_path, 32)
            self.player_font = ImageFont.truetype(font_path, 28)
            self.score_font = ImageFont.truetype(font_path, 72)

        self.drawing_x_pos = 35
        self.drawing_y_pos = 0
        self.user_box_width = 500
        self.user_box_height = 100
        self.x_padding_users = 20
        self.y_padding_games = 20
        self.x_padding_columns = 80
        self.logo_padding = 10
        self.title_padding = 30
        self.score_centered_bounding_box = 100
        self.gradient_factor = 1.5
        self.logo_player_padding = 10
        self.user_box_score_right_padding = 50


def create_box(player_display_name, team_name, box_color, score, logo_path, starting_x_pos, starting_y_pos, settings: GeneratorSettings, image, draw):

    def create_gradient_mask(start_color, width, height):
        gradient = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        gradient_draw = ImageDraw.Draw(gradient)
        start_color = ImageColor.getrgb(start_color)

        for i in range(width):
            if (i / width) < 0.5:
                alpha = 255
            else:
                alpha = int(255 - (((i / width)-0.5) * 255)* settings.gradient_factor)
            current_color = start_color + (alpha,)
            gradient_draw.line([(i, 0), (i, height)], fill=current_color)

        return gradient


    gradient_mask = create_gradient_mask(box_color, settings.user_box_width, settings.user_box_height)
    # Paste the gradient mask onto the image
    image.paste(gradient_mask, (starting_x_pos, starting_y_pos), gradient_mask)
    use_team_name = True
    use_player_name = True

    if settings.team_font == 0:
        use_team_name = False
        team_height = 0
        team_top_font_padding = 0
    else:
        team_height = settings.team_font.getmask(team_name).size[1]
        team_top_font_padding = settings.team_font.getbbox(team_name)[1]

    if settings.player_font == 0:
        use_player_name = False
        team_height = 0
        player_top_font_padding = 0
    else:
        player_height = settings.player_font.getmask('MattGree').size[1]
        player_top_font_padding = settings.player_font.getbbox('MattGree')[1]
    
    if settings.team_font == 0:
        player_text_y = starting_y_pos - player_top_font_padding + (settings.user_box_height - player_height)/2
    else:
        total_team_and_player_height = team_height + player_height + team_top_font_padding + player_top_font_padding
        team_text_y = starting_y_pos + (settings.user_box_height - total_team_and_player_height)/2 - team_top_font_padding/2
        player_text_y = team_text_y + team_height + team_top_font_padding

    if use_team_name:
        draw.text((starting_x_pos + settings.user_box_height + settings.logo_player_padding, team_text_y),
                    team_name,
                    font=settings.team_font,
                    fill='white',
                    stroke_width=2,
                    stroke_fill='black'
        )
    
    if use_player_name:
        draw.text((starting_x_pos + settings.user_box_height + settings.logo_player_padding, player_text_y),
                    player_display_name,
                    font=settings.player_font,
                    fill='white',
                    stroke_width=2,
                    stroke_fill='black'
        )

    try:
        player_logo = Image.open(logo_path).convert('RGBA')
    except:
        player_logo = Image.new('RGBA', (100,100), (255, 255, 255, 0))  # Transparent

    original_width, original_height = player_logo.size
    player_logo_height = settings.user_box_height - 2*settings.logo_padding
    ratio = player_logo_height / original_height
    resized_logo_width = int(original_width * ratio)
    new_resized_logo_height = player_logo_height
    resized_logo = player_logo.resize((resized_logo_width, new_resized_logo_height), Image.LANCZOS)

    paste_position = (starting_x_pos + settings.logo_padding + (player_logo_height-resized_logo_width)//2, starting_y_pos + settings.logo_padding)
    image.paste(resized_logo, paste_position, resized_logo)

    score_bbox = settings.score_font.getbbox(score)
    score_text_width, score_text_height = score_bbox[2], score_bbox[3]+score_bbox[1]

    score_text_width = settings.score_font.getmask(score).size[0]

    # Place the center of the text on the (box width - right padding) value
    score_x_start = starting_x_pos + settings.user_box_width - settings.user_box_score_right_padding - score_text_width/2

    centered_scoring_start = (settings.user_box_height - score_text_height)/2
    score_pos = settings.user_box_width - settings.score_centered_bounding_box + (settings.score_centered_bounding_box - score_text_width)-20
    draw.text((score_x_start, starting_y_pos+centered_scoring_start), score, font=settings.score_font, fill='white', stroke_width=2, stroke_fill='black')

def generator(games_df: pd.DataFrame, settings: GeneratorSettings, limit_days_ago=-1, limit_games=-1, subtitle=False, size='large', output_name = 'Graphic.png'):

    use_logo = True
    use_background = True
    games_of_interest = games_df.sort_values(by='date_time_end', ascending=False)

    if limit_games != -1:
        games_of_interest = games_of_interest.iloc[range(limit_games)]
        print(games_of_interest)
    elif limit_days_ago != -1:
        eastern = pytz.timezone('US/Eastern')
        midnight_x_days_ago = (datetime.now(eastern) - timedelta(days=limit_days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
        games_of_interest = games_of_interest[games_of_interest['date_time_start'] > midnight_x_days_ago]
        print(games_of_interest)

    games_of_interest = games_of_interest.reset_index()

    if size == 'large':
        graphic_width = 2190
        graphic_height = 1800
    elif size == 'small':
        graphic_width = 1090
        graphic_height = 1200
    else:
        graphic_width = settings.user_box_width
        graphic_height = 2*settings.user_box_height + 2*settings.y_padding_games
        use_logo = False
        use_background = False

    image = Image.new('RGBA', (graphic_width, graphic_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    drawing_x_pos = settings.drawing_x_pos
    drawing_y_pos = settings.drawing_y_pos
    
    if use_background:
        background_image = Image.open(f'{settings.path}background.png').convert('RGBA')
        resized_background = background_image.resize((3633,1800))
        image.paste(resized_background, (0,0))

    # If want league name in text in future
    # text = settings.league_name
    # text_width, text_height = settings.title_font.getbbox(text)[2:]
    # center_x = (image.width - text_width) // 2
    # center_y = 50

    # Resize league logo to 200 px tall

    if use_logo:
        drawing_y_pos = 200
        league_logo = Image.open(f"{settings.path}League.png").convert('RGBA')
        league_logo_height = 200
        original_width, original_height = league_logo.size
        ratio = league_logo_height / original_height
        resized_logo_width = int(original_width * ratio)
        new_resized_logo_height = league_logo_height
        resized_logo = league_logo.resize((resized_logo_width, new_resized_logo_height), Image.LANCZOS)

        if subtitle:
            text_width, text_height = settings.subtitle_font.getbbox(subtitle)[2:]
            logo_centered = (graphic_width - (text_width + resized_logo_width + settings.title_padding)) // 2
            paste_position = (logo_centered, settings.title_padding)
            image.paste(resized_logo, paste_position, resized_logo)
            text_centered = (logo_centered + settings.title_padding + resized_logo_width, (new_resized_logo_height - text_height) // 2)
            draw.text(text_centered, subtitle, font=settings.subtitle_font, fill='white', stroke_width=4, stroke_fill='black')


        else:
            paste_position = ((graphic_width-resized_logo_width)//2, 10)
            image.paste(resized_logo, paste_position, resized_logo)
    
    def new_row_drawing(y_drawing_pos):
        y_drawing_pos += 2*settings.user_box_height + settings.y_padding_games
        return y_drawing_pos
    
    def reset_drawing_y_pos_after_graphic_height_exceeded(x, y):
        next_scoreboard_bottom_px = y + 2*(settings.user_box_height + settings.y_padding_games)
        if next_scoreboard_bottom_px >= graphic_height:
            y = 200 
            x = settings.drawing_x_pos + 2*(settings.user_box_width) + settings.x_padding_users + settings.x_padding_columns
            old_weekday = ''
            new_row = True
    
        return x, y
    
    old_weekday = ''
    old_home = ''
    old_away = ''
    new_row = True

    for index, game in games_of_interest.iterrows():
        print(game['home_user'], game['away_user'])
        new_weekday = game['date_time_start'].strftime('%A, %B %-d, %Y')
        if new_weekday != old_weekday:
            new_row=True
            if index !=0:
                drawing_y_pos = new_row_drawing(drawing_y_pos)
                drawing_x_pos, drawing_y_pos = reset_drawing_y_pos_after_graphic_height_exceeded(drawing_x_pos, drawing_y_pos)
            draw.text((drawing_x_pos, drawing_y_pos), new_weekday, font=settings.team_font, fill='white', stroke_width=2, stroke_fill='black')
            drawing_y_pos += settings.team_font.getbbox(new_weekday)[3]
            old_weekday = new_weekday
        elif new_row and index != 0:
            drawing_y_pos = new_row_drawing(drawing_y_pos)
            drawing_x_pos, drawing_y_pos = reset_drawing_y_pos_after_graphic_height_exceeded(drawing_x_pos, drawing_y_pos)

        new_away = game['away_user'].lower()
        new_home = game['home_user'].lower()

        away_display_name = settings.player_data[new_away]['Display Name']
        away_team_name = settings.player_data[new_away]['Team Name']
        away_color = settings.player_data[new_away]['Color']
        away_score = str(int(game['away_score']))
        away_logo = settings.path + f'Logos/{new_away}.png'
        
        home_display_name = settings.player_data[new_home]['Display Name']
        home_team_name = settings.player_data[new_home]['Team Name']
        home_color = settings.player_data[new_home]['Color']
        home_score = str(int(game['home_score']))
        home_drawing_y_pos = drawing_y_pos + settings.user_box_height
        home_logo = settings.path + f'Logos/{new_home}.png'

        column_2_drawing_x_pos = drawing_x_pos + settings.user_box_width + settings.x_padding_users
        
        if new_row:
            create_box(away_display_name, away_team_name, away_color, away_score, away_logo, drawing_x_pos, drawing_y_pos, settings, image, draw)
            create_box(home_display_name, home_team_name, home_color, home_score, home_logo, drawing_x_pos, home_drawing_y_pos, settings, image, draw)
            new_row = False
        else:
            create_box(away_display_name, away_team_name, away_color, away_score, away_logo, column_2_drawing_x_pos, drawing_y_pos, settings, image, draw)
            create_box(home_display_name, home_team_name, home_color, home_score, home_logo, column_2_drawing_x_pos, home_drawing_y_pos, settings, image, draw)
            new_row = True

        old_home = new_home
        old_away = new_away

    image.save(output_name)

    if use_background:
        background_image = Image.open(f"{settings.path}background.png").convert('RGB')
        resized_background = background_image.resize((graphic_width, graphic_height))  # Example resize
        remove_transparency = Image.open(output_name).convert('RGBA')
        resized_background.paste(remove_transparency, (0, 0), remove_transparency)  # Using the alpha channel as mask
        resized_background.save(output_name)

def standings_generator(games_df: pd.DataFrame, settings: GeneratorSettings, output_name = 'StandingsGraphic.png'):
    
    use_logo = True
    games_of_interest = games_df.sort_values(by='date_time_end', ascending=False)

    games_of_interest = games_of_interest.reset_index()

    graphic_width = 325
    graphic_height = 512

    image = Image.new('RGBA', (graphic_width, graphic_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    drawing_x_pos = settings.drawing_x_pos
    drawing_y_pos = settings.drawing_y_pos
    
    standings = {}

    for index, game in games_df.iterrows():
        winner, loser = game["winner_user"], game["loser_user"]
        winner_score, loser_score = game["winner_score"], game["loser_score"]

        # Update winner stats
        if winner not in standings:
            standings[winner] = {"Wins": 0, "Losses": 0, "Games Played": 0, "Run Differential": 0}
        standings[winner]["Wins"] += 1
        standings[winner]["Games Played"] += 1
        standings[winner]["Run Differential"] += (winner_score - loser_score)

        # Update loser stats
        if loser not in standings:
            standings[loser] = {"Wins": 0, "Losses": 0, "Games Played": 0, "Run Differential": 0}
        standings[loser]["Losses"] += 1
        standings[loser]["Games Played"] += 1
        standings[loser]["Run Differential"] += (loser_score - winner_score)

    # Convert standings to a DataFrame
    standings_df = pd.DataFrame.from_dict(standings, orient="index")

    # Calculate Win Percentage
    standings_df["Win%"] = standings_df["Wins"] / standings_df["Games Played"]

    # Sort by Win%, Games Played, and Run Differential
    standings_df = standings_df.sort_values(by=["Win%", "Wins", "Losses", "Run Differential"], ascending=[False, False, True, False])

    # Display standings
    print(standings_df)

    for player, standings in standings_df.iterrows():
        player = player.lower()
        player_display_name = settings.player_data[player]['Display Name']
        player_team_name = settings.player_data[player]['Team Name']
        player_color = settings.player_data[player]['Color']
        player_WL = f'{int(standings["Wins"])}-{int(standings["Losses"])}'
        player_logo = settings.path + f'Logos/{player}.png'
    
        create_box(player_display_name, player_team_name, player_color, player_WL, player_logo, drawing_x_pos, drawing_y_pos, settings, image, draw)

        drawing_y_pos += settings.user_box_height
        
    image.save('StandingsGraphic.png')


    image = Image.new('RGBA', (graphic_width, graphic_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    remove_transparencey = Image.open("StandingsGraphic.png").convert('RGBA')
    image.paste(remove_transparencey, (0,0))

    image.save(output_name)

def make_graphic(settings: GeneratorSettings, edit_mode_file=False):
    manager = api_manager.APIManager()
    api_response = web_functions.games_endpoint(manager, settings.tag, limit_games=500)
    web_cache = web_caching.CompleterCache(manager)
    games_df = endpoint_handling.games_endpoint(api_response, web_cache)
    if edit_mode_file:
        add_games_df = pd.read_json('output.json', orient='records')
        print(add_games_df)
        add_games_df['date_time_start'] = pd.to_datetime(add_games_df['date_time_start'], utc=True, unit='ms').dt.tz_convert('US/Eastern')
        add_games_df['date_time_end'] = pd.to_datetime(add_games_df['date_time_end'], utc=True, unit='ms').dt.tz_convert('US/Eastern')
        print(add_games_df)
        games_df = pd.concat([games_df, add_games_df], ignore_index=True)

    print(games_df)

    generator(games_df, settings, limit_days_ago=1, size='small')


def make_standings_graphic(settings: GeneratorSettings, edit_mode_file=False):
    manager = api_manager.APIManager()
    api_response = web_functions.games_endpoint(manager, settings.tag, limit_games=500)
    web_cache = web_caching.CompleterCache(manager)
    games_df = endpoint_handling.games_endpoint(api_response, web_cache)
    if edit_mode_file:
        add_games_df = pd.read_json('output.json', orient='records')
        print(add_games_df)
        add_games_df['date_time_start'] = pd.to_datetime(add_games_df['date_time_start'], utc=True, unit='ms').dt.tz_convert('US/Eastern')
        add_games_df['date_time_end'] = pd.to_datetime(add_games_df['date_time_end'], utc=True, unit='ms').dt.tz_convert('US/Eastern')
        print(add_games_df)
        games_df = pd.concat([games_df, add_games_df], ignore_index=True)

    print(games_df)

    standings_generator(games_df, settings)


def make_individual_games(settings: GeneratorSettings):
    manager = api_manager.APIManager()
    api_response = web_functions.games_endpoint(manager, settings.tag, limit_games=500)
    web_cache = web_caching.CompleterCache(manager)
    games_df = endpoint_handling.games_endpoint(api_response, web_cache)

    for index, game in games_df.iterrows():
        game_df = games_df.iloc[[index]].reset_index(drop=True)
        generator(game_df, settings, size='solo', output_name=f'graphic_{index}.png')

def make_weekly_graphics(settings: GeneratorSettings, matchup: str, edit_mode_file=True):
    manager = api_manager.APIManager()
    api_response = web_functions.games_endpoint(manager, settings.tag, limit_games=500)
    web_cache = web_caching.CompleterCache(manager)
    games_df = endpoint_handling.games_endpoint(api_response, web_cache)

    if edit_mode_file:
        add_games_df = pd.read_json('output.json', orient='records')
        add_games_df['date_time_start'] = pd.to_datetime(add_games_df['date_time_start'], utc=True, unit='ms').dt.tz_convert('US/Eastern')
        add_games_df['date_time_end'] = pd.to_datetime(add_games_df['date_time_end'], utc=True, unit='ms').dt.tz_convert('US/Eastern')
        games_df = pd.concat([games_df, add_games_df], ignore_index=True)

    week_matchups = settings.matchups[matchup]

    def match_player_to_team(player_name):
        best_match = process.extractOne(player_name.lower(), settings.player_data.keys())
        if best_match:
            # Return the team information corresponding to the best match
            return best_match[0]
        return None
    
    converted_matchups = []

    for single_matchup in week_matchups:
        converted_matchups.append(tuple(sorted([match_player_to_team(single_matchup[0]), match_player_to_team(single_matchup[1])])))

    print(converted_matchups)

    print(games_df)

    # Apply the function to the DataFrame
    filtered_games_df = games_df[
    games_df.apply(lambda row: tuple(sorted([row['away_user'].lower(), row['home_user'].lower()])) in converted_matchups, axis=1)
    ]

    print(filtered_games_df)

    generator(filtered_games_df, settings, subtitle=matchup, size='large', output_name=f'WeeklyGraphic.png')


pjfriendsfont = {
    'title': 96,
    'subtitle': 64,
    'team': 20,
    'player': 30,
    'score': 72
}

PJFriendsSettings = GeneratorSettings('PJandFriendsTeamClassic2', font_dict=pjfriendsfont)
#NNLSeason6Settings = GeneratorSettings('NNLSeason6')
#NNLS7TrainingSettings = GeneratorSettings('NNLS7Training')
NNLS7Settings = GeneratorSettings('NNLSeason7')

make_weekly_graphics(NNLS7Settings, 'Interleague Kickoff')

PJFriendsIndividualGames = GeneratorSettings('PJandFriendsTeamClassic2', font_dict=pjfriendsfont)
PJFriendsIndividualGames.drawing_x_pos = 0
PJFriendsIndividualGames.gradient_factor = 0.75

#make_graphic(NNLS7Settings, edit_mode_file=True)
# make_individual_games(PJFriendsIndividualGames)

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
NNLS7StandingsSettings.gradient_factor = 0.5
NNLS7StandingsSettings.logo_player_padding = 25
NNLS7StandingsSettings.user_box_score_right_padding = 30

#make_standings_graphic(NNLS7StandingsSettings, edit_mode_file=True)