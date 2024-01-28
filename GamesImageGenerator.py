import requests
from PIL import Image, ImageDraw, ImageFont, ImageColor
from datetime import datetime, timedelta
import RioStatsConverter
import pandas as pd
import json

url = "https://api.projectrio.app/games/?tag=NNLSeason5"

def parse_games_endpoint(url):
    export_list = pd.DataFrame(columns=["Winner User", "Winner Captain", "Winner Score", "Winner Side", "Winner Elo Before", "Winner Elo After",
                    "Loser User", "Loser Captain", "Loser Score", "Loser Side", "Loser Elo Before", "Loser Elo After",
                    "Innings Played", "Game ID", "Game Start Date", "Game End Date", "Game Time Length (Minutes)", "Stadium"])
    
    print(url)
    r = requests.get(url)
    jsonObj = r.json()

    print("RioWeb Request Complete")
    for event in jsonObj["games"]:
        if event["away_score"] > event["home_score"]:
            winner, loser = "away", "home"
        else:
            winner, loser = "home", "away"

        new_row = {
            'Home User': event["home_user"],
            'Home Captain': event["home_captain"],
            'Home Score': event["home_score"],
            'Winner Elo Before': event["winner_incoming_elo"],
            'Winner Elo After': event["winner_result_elo"],
            'Away User': event["away_user"],
            'Away Captain': event["away_captain"],
            'Away Score': event["away_score"],
            'Loser Elo Before': event["loser_incoming_elo"],
            'Loser Elo After': event["loser_result_elo"],
            'Innings Played': event["innings_played"],
            'Game ID': event["game_id"],
            'Game Start Date': datetime.fromtimestamp(event["date_time_start"]),
            'Game End Date': datetime.fromtimestamp(event["date_time_end"]),
            'Game Length (Minutes)': str((event["date_time_end"] - event["date_time_start"])/60),
            'Stadium': RioStatsConverter.stadium_id(event["stadium"]),
        }
        
        export_list = export_list.append(new_row, ignore_index=True)


    print(export_list)
    export_list.to_json('output.json', orient='records', lines=True)
    return export_list

def generator(games_df, limit_days_ago=-1, limit_games=-1, subtitle=False):
    with open('NNLS5Players.json', 'r') as json_file:
        player_data = json.load(json_file)

    if limit_games != -1:
        games_of_interest = games_df.iloc[range(limit_games)]
        print(games_of_interest)
    elif limit_days_ago != -1:
        midnight_x_days_ago = (datetime.now() - timedelta(days=limit_days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
        games_of_interest = games_df[games_df['Game Start Date'].apply(lambda x: pd.to_datetime(x) > midnight_x_days_ago)]
        print(games_of_interest)

    graphic_width = 2180
    graphic_height = 1800
    image = Image.new('RGBA', (graphic_width, graphic_height), color='#00B3FF')
    draw = ImageDraw.Draw(image)

    background_image = Image.open("standings_backdrop.png").convert('RGBA')
    resized_background = background_image.resize((3633,1800))
    image.paste(resized_background, (0,0))

    # Load the local font Lalezar
    font_path = "Lalezar-Regular.ttf"  # Replace with the actual path to your Lalezar font file
    title_font = ImageFont.truetype(font_path, 96)
    subtitle_font = ImageFont.truetype(font_path, 64)
    player_font = ImageFont.truetype(font_path, 32)
    small_player_font = ImageFont.truetype(font_path, 24)
    score_font = ImageFont.truetype(font_path, 72)

    text = "National Netplay League"
    text_width, text_height = title_font.getbbox(text)[2:]
    center_x = (image.width - text_width) // 2
    center_y = 50

    # draw.text((center_x, center_y), text, font=title_font, fill='white', stroke_width=5, stroke_fill='black')
    #Subtitle
    league_logo = Image.open("NationalNetplayLeagueLogo.png").convert('RGBA')
    league_logo_size = 200
    new_size = (league_logo_size, league_logo_size)
    resized_logo = league_logo.resize(new_size)

    drawing_x_pos = 50
    drawing_y_pos = 200
    user_box_width = 500
    user_box_height = 100
    x_padding_users = 10
    y_padding_games = 25
    x_padding_columns = 50
    logo_padding = 10
    title_padding = 30
    score_centered_bounding_box = 100

    if subtitle:
        text_width, text_height = subtitle_font.getbbox(subtitle)[2:]
        logo_centered = (graphic_width - (text_width + league_logo_size + title_padding)) // 2
        paste_position = (logo_centered, title_padding)
        image.paste(resized_logo, paste_position, resized_logo)
        text_centered = (logo_centered + title_padding + league_logo_size, (league_logo_size - text_height) // 2)
        draw.text(text_centered, subtitle, font=subtitle_font, fill='white', stroke_width=4, stroke_fill='black')
    else:
        paste_position = ((graphic_width-200)//2, 10)
        image.paste(resized_logo, paste_position, resized_logo)

    def create_gradient_mask(start_color, width, height):
        gradient = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        gradient_draw = ImageDraw.Draw(gradient)

        for i in range(width):
            if (i / width) < 0.5:
                alpha = 255
            else:
                alpha = int(255 - (((i / width)-0.5) * 255)*2*.75)
            current_color = start_color + (alpha,)
            gradient_draw.line([(i, 0), (i, height)], fill=current_color)

        return gradient


    def create_box(player, score, starting_x_pos, starting_y_pos):
        start_color = ImageColor.getrgb(player_data[player]['Color'])

        # Create a gradient mask
        gradient_mask = create_gradient_mask(start_color, user_box_width, user_box_height)

        # Paste the gradient mask onto the image
        image.paste(gradient_mask, (starting_x_pos, starting_y_pos), gradient_mask)
        
        player_text_height = player_font.getbbox(player)[3]
        draw.text((starting_x_pos + user_box_height, starting_y_pos + (user_box_height - player_text_height) // 3-3), player_data[player]['Team Name'], font=player_font, fill='white', stroke_width=2, stroke_fill='black')
        draw.text((starting_x_pos + user_box_height, starting_y_pos + (user_box_height - player_text_height)*2 // +3), player, font=small_player_font, fill='white', stroke_width=2, stroke_fill='black')
        # Open another PNG image
        winner_image_path = f"Season5TeamLogos/{player}.png"  # Replace with the path to your PNG image
        winner_image = Image.open(winner_image_path).convert('RGBA')
        new_size = (user_box_height - 2*logo_padding, user_box_height - 2*logo_padding)
        resized_winner = winner_image.resize(new_size)
        paste_position = (starting_x_pos + logo_padding, starting_y_pos + logo_padding)
        image.paste(resized_winner, paste_position, resized_winner)

        score_text_width, score_text_height = score_font.getbbox(score)[2], score_font.getbbox(score)[3]
        score_pos = user_box_width - score_centered_bounding_box + (score_centered_bounding_box - score_text_width)-20
        draw.text((starting_x_pos + score_pos, starting_y_pos), score, font=score_font, fill='white', stroke_width=2, stroke_fill='black')
    
    old_weekday = ''
    for index, game in games_of_interest.iterrows():

        new_weekday = game['Game Start Date'].strftime('%A, %B %d, %Y')
        if new_weekday != old_weekday:
            draw.text((drawing_x_pos, drawing_y_pos), new_weekday, font=player_font, fill='white', stroke_width=2, stroke_fill='black')
            drawing_y_pos += player_font.getbbox(new_weekday)[3]
            old_weekday = new_weekday

        away = game['Away User'].lower()
        home = game['Home User'].lower()

        create_box(away, str(int(game['Away Score'])), drawing_x_pos, drawing_y_pos)
        create_box(home, str(int(game['Home Score'])), drawing_x_pos + user_box_width + x_padding_users, drawing_y_pos)
        
        drawing_y_pos += user_box_height + y_padding_games
        print(drawing_y_pos + user_box_height)

        if drawing_y_pos + user_box_height + y_padding_games >= graphic_height:
            drawing_y_pos = 200 
            drawing_x_pos += 2*(user_box_width)+ x_padding_users + x_padding_columns
            old_weekday = ''

    # Save the image
    image.save('Graphic.png')
    

def make_graphic(edit_mode_file=False):
    if edit_mode_file:
        games_df = pd.read_json('output.json', orient='records', lines=True)
    else:
        games_df = parse_games_endpoint(url)

    generator(games_df, limit_days_ago=116, subtitle='Week 2 Results')

make_graphic()