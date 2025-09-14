from PIL import Image, ImageDraw, ImageFont, ImageColor
from datetime import datetime, timedelta
import pandas as pd
import pytz
import json
from fuzzywuzzy import process
from pathlib import Path

from pyrio import endpoint_handling, web_functions, api_manager, web_caching
import league_data_handler as ldh

class GeneratorSettings(ldh.LeagueData):
    def __init__(self, tag, font_dict=False):
        super().__init__(tag)

        font_path = self.path / self.font

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

        # Starting position for adding elements within the image
        self.drawing_x_pos = 35
        self.drawing_y_pos = 0

        # Size of the box for a single user
        self.user_box_width = 500
        self.user_box_height = 100

        # Padding between games
        self.x_padding_users = 20
        self.y_padding_games = 20

        # Padding between columns (a column is made of two games left to right)
        self.x_padding_columns = 80

        # Top/Bottom padding on a logo & minimum left padding
        self.logo_padding = 10

        # Padding between the logo and the subtitle
        self.title_padding = 30

        # Affects how fast the color in the boxes fade to transparent
        self.gradient_factor = 1.5

        # Padding between logo and player/team names
        self.logo_player_padding = 10

        # Location of the center of the score in a box from the right side of the box
        self.user_box_score_right_padding = 50

        self.graphic_sizes = {
            'large': (2190, 1800),
            'small': (1090, 1200),
            'individual': (self.user_box_width,  2*self.user_box_height + 2*self.y_padding_games)
        }

        self.standigs_graphic_sizes = {
            'stream': (325, 512),
            'weekly graphic': (600, 912),
        }


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
        player_logo = Image.new('RGBA', (100,100), (255, 255, 255, 0))

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
    draw.text((score_x_start, starting_y_pos+centered_scoring_start), score, font=settings.score_font, fill='white', stroke_width=2, stroke_fill='black')

def scorecard_generator(games_df: pd.DataFrame, settings: GeneratorSettings, limit_days_ago=-1, limit_games=-1, subtitle=False, size_input='large', output_name = 'Graphic.png', player_filter = False):
    use_logo = True
    use_background = True
    games_of_interest = games_df.sort_values(by='date_time_end', ascending=False, ignore_index=True)

    if limit_games != -1:
        games_of_interest = games_of_interest.iloc[range(limit_games)]
        print(games_of_interest)
    elif limit_days_ago != -1:
        eastern = pytz.timezone('US/Eastern')
        midnight_x_days_ago = (datetime.now(eastern) - timedelta(days=limit_days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
        games_of_interest = games_of_interest[games_of_interest['date_time_start'] > midnight_x_days_ago]
        print(games_of_interest)

    games_of_interest = games_of_interest.reset_index()

    if player_filter:
        games_of_interest = games_of_interest[(games_of_interest['away_user'] == player_filter) | (games_of_interest['home_user'] == player_filter)]
        subtitle = f'{player_filter} NNL Season 7 Games'

    graphic_size = settings.graphic_sizes[size_input]

    image = Image.new('RGBA', graphic_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    drawing_x_pos = settings.drawing_x_pos
    drawing_y_pos = settings.drawing_y_pos
    
    if use_background:
        background_image = Image.open(settings.path / 'background.png').convert('RGBA')
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
        league_logo = Image.open(settings.path / 'League.png').convert('RGBA')
        league_logo_height = 200
        original_width, original_height = league_logo.size
        ratio = league_logo_height / original_height
        resized_logo_width = int(original_width * ratio)
        new_resized_logo_height = league_logo_height
        resized_logo = league_logo.resize((resized_logo_width, new_resized_logo_height), Image.LANCZOS)

        if subtitle:
            text_width, text_height = settings.subtitle_font.getbbox(subtitle)[2:]
            logo_centered = (graphic_size[0] - (text_width + resized_logo_width + settings.title_padding)) // 2
            paste_position = (logo_centered, settings.title_padding)
            image.paste(resized_logo, paste_position, resized_logo)
            text_centered = (logo_centered + settings.title_padding + resized_logo_width, (new_resized_logo_height - text_height) // 2)
            draw.text(text_centered, subtitle, font=settings.subtitle_font, fill='white', stroke_width=4, stroke_fill='black')


        else:
            paste_position = ((graphic_size[0]-resized_logo_width)//2, 10)
            image.paste(resized_logo, paste_position, resized_logo)
    
    def new_row_drawing(y_drawing_pos):
        y_drawing_pos += 2*settings.user_box_height + settings.y_padding_games
        return y_drawing_pos
    
    def reset_drawing_y_pos_after_graphic_height_exceeded(x, y, new_row, old_weekday):
        next_scoreboard_bottom_px = y + 2*(settings.user_box_height + settings.y_padding_games)
        if next_scoreboard_bottom_px >= graphic_size[1]:
            y = 200 
            x = settings.drawing_x_pos + 2*(settings.user_box_width) + settings.x_padding_users + settings.x_padding_columns
            old_weekday = ''
            new_row = True
    
        return x, y, new_row, old_weekday
    
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
                drawing_x_pos, drawing_y_pos, new_row, old_weekday = reset_drawing_y_pos_after_graphic_height_exceeded(drawing_x_pos, drawing_y_pos, new_row, old_weekday)
            draw.text((drawing_x_pos, drawing_y_pos), new_weekday, font=settings.team_font, fill='white', stroke_width=2, stroke_fill='black')
            drawing_y_pos += settings.team_font.getbbox(new_weekday)[3]
            old_weekday = new_weekday
        elif new_row and index != 0:
            drawing_y_pos = new_row_drawing(drawing_y_pos)
            drawing_x_pos, drawing_y_pos, new_row, old_weekday = reset_drawing_y_pos_after_graphic_height_exceeded(drawing_x_pos, drawing_y_pos, new_row, old_weekday)
            draw.text((drawing_x_pos, drawing_y_pos), new_weekday, font=settings.team_font, fill='white', stroke_width=2, stroke_fill='black')
            drawing_y_pos += settings.team_font.getbbox(new_weekday)[3]
            old_weekday = new_weekday

        new_away = game['away_user'].lower()
        new_home = game['home_user'].lower()

        away_display_name = settings.player_data[new_away]['Display Name']
        away_team_name = settings.player_data[new_away]['Team Name']
        away_color = settings.player_data[new_away]['Color']
        away_score = str(int(game['away_score']))
        away_logo = settings.path / 'Logos' / f'{new_away}.png'
        
        home_display_name = settings.player_data[new_home]['Display Name']
        home_team_name = settings.player_data[new_home]['Team Name']
        home_color = settings.player_data[new_home]['Color']
        home_score = str(int(game['home_score']))
        home_drawing_y_pos = drawing_y_pos + settings.user_box_height
        home_logo = settings.path / 'Logos' / f'{new_home}.png'

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

    image.save(settings.path / output_name)

    if use_background:
        background_image = Image.open(settings.path / 'background.png').convert('RGB')
        resized_background = background_image.resize(graphic_size)  # Example resize
        remove_transparency = Image.open(settings.path / output_name).convert('RGBA')
        resized_background.paste(remove_transparency, (0, 0), remove_transparency)  # Using the alpha channel as mask
        resized_background.save(settings.path / output_name)

def generate_stadings_df(games_df, settings: ldh.LeagueData):
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

    player_keys = list(settings.player_data.keys())

    # Function to find the best fuzzy match
    def fuzzy_match(name, choices, threshold=80):
        match, score = process.extractOne(name, choices)
        return match if score >= threshold else None  # Only return if confidence is high

    # Apply fuzzy matching to standings index
    standings_df["Matched Username"] = standings_df.index.map(lambda x: fuzzy_match(x, player_keys))

    # Drop rows where no match was found
    standings_df = standings_df.dropna(subset=["Matched Username"])
    standings_df["Division"] = standings_df["Matched Username"].map(lambda x: settings.player_data[x]["Division"])

    top_division = standings_df.iloc[0]["Division"]  # First place player's division
    # Find the highest-ranked team from the other division
    other_division_team = standings_df[standings_df["Division"] != top_division].iloc[0]  # Extract row

    # Reorder the standings
    standings_df = pd.concat([
        standings_df.iloc[:1],  # Keep the top team
        standings_df.loc[[other_division_team.name]],  # Move the top team from the other division to 2nd place
        standings_df.iloc[1:].drop(index=other_division_team.name)  # Remove them from their old position & keep the rest
    ])
        # Display standings
    # print(standings_df)

    return standings_df


def standings_generator(games_df: pd.DataFrame, settings: GeneratorSettings, size_input='stream', title=False, output_name = 'StandingsGraphic.png'):
    games_of_interest = games_df.sort_values(by='date_time_end', ascending=False)
    games_of_interest = games_of_interest.reset_index()

    graphic_width, graphic_height = settings.standigs_graphic_sizes[size_input]
    
    if title:
        title_top_font_padding = settings.title_font.getmask(title).size[1]
        title_width = settings.title_font.getbbox(title)[2]
        graphic_height+=title_top_font_padding+settings.title_padding

    image = Image.new('RGBA', (graphic_width, graphic_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    drawing_x_pos = settings.drawing_x_pos
    drawing_y_pos = settings.drawing_y_pos

    if title:
        x_title_centered = (graphic_width - drawing_x_pos - title_width)/2
        
        draw.text((x_title_centered, drawing_y_pos - settings.title_font.getbbox(title)[1]+2),
                    title,
                    font=settings.title_font,
                    fill='white',
                    stroke_width=2,
                    stroke_fill='black'
        )

        drawing_y_pos += title_top_font_padding +settings.title_padding

    standings_df = generate_stadings_df(games_df, settings)

    for player, standings in standings_df.iterrows():
        player = player.lower()
        player_display_name = settings.player_data[player]['Display Name']
        player_team_name = settings.player_data[player]['Team Name']
        player_color = settings.player_data[player]['Color']
        player_WL = f'{int(standings["Wins"])}-{int(standings["Losses"])}'
        player_logo = settings.path  / 'Logos' / f'{player}.png'
    
        create_box(player_display_name, player_team_name, player_color, player_WL, player_logo, drawing_x_pos, drawing_y_pos, settings, image, draw)

        drawing_y_pos += settings.user_box_height
        
    image.save(settings.path / output_name)

def remove_games_after_cutoff(settings: ldh.LeagueData, games_df: pd.DataFrame, cutoff: str):
    ordered_weeks = list(settings.matchups.keys())
    if cutoff not in ordered_weeks:
        raise ValueError(f"'{cutoff}' not found in the list.")
    
    index = ordered_weeks.index(cutoff)  # Find index of the item
    ordered_weeks = ordered_weeks[:index]

    print(ordered_weeks)

    games_df = games_df[games_df["Week"].isin(ordered_weeks)]

    return games_df

def make_individual_games(settings: GeneratorSettings, games_df: pd.DataFrame, edit_mode_file):
    games_df = ldh.get_web_games(settings, edit_mode_file)

    for index, game in games_df.iterrows():
        game_df = games_df.iloc[[index]].reset_index(drop=True)
        scorecard_generator(game_df, settings, size_input='solo', output_name=f'graphic_{index}.png')

def make_weekly_graphics(settings: GeneratorSettings, games_df: pd.DataFrame, matchup: str):
    filtered_games_df = games_df[games_df['Week'] == matchup]

    scorecard_generator(filtered_games_df, settings, subtitle=matchup, size_input='large', output_name=f'WeeklyGraphic.png')