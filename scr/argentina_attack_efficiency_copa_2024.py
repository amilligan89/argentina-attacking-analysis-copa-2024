# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:34:28 2025

@author: A.milligan
"""

#%% Imports
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html


#%% Load events and lineups

events_dir = r"C:/Users/a.milligan/Documents/My Files/soccer_analytics_project/events"
lineups_dir = r"C:/Users/a.milligan/Documents/My Files/soccer_analytics_project/lineups"

# Load event files
event_dfs = []
for file in os.listdir(events_dir):
    if file.endswith(".json"):
        with open(os.path.join(events_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.json_normalize(data)
        event_dfs.append(df)

all_events = pd.concat(event_dfs, ignore_index=True)

# Load lineup files
lineup_dfs = []
for file in os.listdir(lineups_dir):
    if file.endswith(".json"):
        with open(os.path.join(lineups_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        for team in data:
            team_name = team["team_name"]
            df = pd.json_normalize(team["lineup"])
            df["team_name"] = team_name
            lineup_dfs.append(df)

all_lineups = pd.concat(lineup_dfs, ignore_index=True)

# Ensure string types for merging
all_events['player.id'] = all_events['player.id'].astype(str)
all_lineups['player_id'] = all_lineups['player_id'].astype(str)

# Merge events with lineups
events_with_lineups = all_events.merge(
    all_lineups,
    left_on=['player.id', 'team.name'],
    right_on=['player_id', 'team_name'],
    how='left'
)

#%% Subsetting

# Filter shots with key_pass_id
shots_with_key_pass = all_events[all_events['type.name'] == 'Shot'][['id', 'shot.statsbomb_xg', 'shot.key_pass_id']].copy()

# Drop rows where key_pass_id is NaN
shots_with_key_pass = shots_with_key_pass[shots_with_key_pass['shot.key_pass_id'].notna()].copy()

# Ensure key_pass_id is string for merging (keep UUIDs as strings)
shots_with_key_pass['shot.key_pass_id'] = shots_with_key_pass['shot.key_pass_id'].astype(str)

# Filter passes
passes = all_events[all_events['type.name'] == 'Pass'][['id', 'player.id']].copy()

# Ensure pass id is string (also UUID)
passes['id'] = passes['id'].astype(str)

# Merge passes with shots using key_pass_id
passes_with_xa = passes.merge(
    shots_with_key_pass,
    left_on='id',                 
    right_on='shot.key_pass_id',  
    how='inner'  # keep only matches
)
# Assign xA = shot xG if pass led to a shot, else 0
passes_with_xa['xA'] = passes_with_xa['shot.statsbomb_xg'].fillna(0)

# Aggregate xA per player
player_xA = passes_with_xa.groupby('player.id')['xA'].sum().reset_index()

#%% Filter Argentina events

arg_events = events_with_lineups[events_with_lineups["team.name"] == "Argentina"].copy()

# Mark shots, assists, key passes
arg_events["is_shot"] = arg_events["type.name"].fillna('') == "Shot"
arg_events["is_assist"] = arg_events["pass.goal_assist"].fillna(False)
arg_events["is_key_pass"] = arg_events["pass.shot_assist"].fillna(False)

# Chance possessions
chance_possessions = set(
    arg_events.loc[arg_events["is_shot"] & (arg_events["shot.statsbomb_xg"].fillna(0) >= 0.10), "possession"]
)
chance_possessions |= set(arg_events.loc[arg_events["is_key_pass"], "possession"])
arg_events["chance_possession"] = arg_events["possession"].isin(chance_possessions)

# Shots subset (exclude penalty shootout shots)
shots = arg_events[
    (arg_events["is_shot"]) &
    ~( (arg_events["shot.type.name"] == "Penalty") & (arg_events["period"] == 5) )
].copy()

# Totals per player
total_shots = shots.groupby("player.name")["id"].count().rename("total_shots")
total_assisted_shots = arg_events.groupby("player.name")["is_key_pass"].sum().rename("total_assisted_shots")
total_assists = arg_events.groupby("player.name")["is_assist"].sum().rename("total_assists")
total_goals = shots[shots["shot.outcome.name"].fillna('')=="Goal"].groupby("player.name")["id"].count().rename("total_goals")
possessions_played = arg_events.groupby("player.name")["possession"].nunique().rename("possessions_played")
total_xG = shots.groupby("player.name")["shot.statsbomb_xg"].sum().rename("total_xG")

# Ensure player ID types match
player_xA['player.id'] = player_xA['player.id'].fillna(0).astype(float).astype(int)
all_lineups['player_id'] = all_lineups['player_id'].astype(int)

# Map player names to xA
player_id_to_name = all_lineups.set_index('player_id')['player_name'].to_dict()
player_xA['player.name'] = player_xA['player.id'].map(player_id_to_name)
total_xA = player_xA.groupby('player.name')['xA'].sum().rename('total_xA')

# Build player attack involvement dataframe
player_attack_involvement = pd.concat([
    total_shots, total_assisted_shots, total_goals, total_assists, possessions_played, total_xG
], axis=1).fillna(0)

# Merge xA explicitly by player.name
player_attack_involvement = player_attack_involvement.merge(
    total_xA.reset_index(),
    on='player.name',
    how='left'
).fillna({'total_xA': 0})

# Compute total chances
player_attack_involvement["total_chances"] = (
    player_attack_involvement["total_shots"] + player_attack_involvement["total_assisted_shots"]
)

# Create combined columns for plotting and filter out players that had minimal contribution in goals and assists
player_attack_involvement["xG_plus_xA"] = player_attack_involvement["total_xG"] + player_attack_involvement["total_xA"]
player_attack_involvement = player_attack_involvement[player_attack_involvement["xG_plus_xA"] > 1]
player_attack_involvement["goals_plus_assists"] = player_attack_involvement["total_goals"] + player_attack_involvement["total_assists"]

# Efficiency % (actual vs expected)
player_attack_involvement["efficiency_%"] = (
    (player_attack_involvement["total_goals"] + player_attack_involvement["total_assists"]) /
    (player_attack_involvement["total_xG"] + player_attack_involvement["total_xA"])
) * 100

# Avoid inf/nan if denominator = 0
player_attack_involvement["efficiency_%"] = player_attack_involvement["efficiency_%"].replace([np.inf, -np.inf], 0).fillna(0)

# Conversion % (goals+assists per chance)
player_attack_involvement["conversion_%"] = (
    (player_attack_involvement["total_goals"] + player_attack_involvement["total_assists"]) /
    player_attack_involvement["total_chances"]
) * 100

player_attack_involvement["conversion_%"] = player_attack_involvement["conversion_%"].replace([np.inf, -np.inf], 0).fillna(0)

# Manual name mapping for short names
manual_names = {
    "Lionel Andrés Messi Cuccittini": "Messi",       
    "Ángel Fabián Di María Hernández": "Di María",                     
    "Rodrigo Javier De Paul": "De Paul",                    
    "Giovani Lo Celso": "Lo Celso",                 
    "Leandro Paredes": "Paredes",
    "Alexis Mac Allister": "MacAllister",
    "Lisandro Martínez": "Li. Martínez",
    "Julián Álvarez": "Álvarez",
    "Lautaro Javier Martínez": "La. Martínez",
    "Damián Emiliano Martínez": "E. Martínez",
    "Naheul Molina Lucero": "Molina",
    "Nicolás Iván González": "González",
    "Nicolás Alejandro Tagliafico": "Tagliafico",
    "Enzo Fernandez": "Enzo",
    "Gonzalo Ariel Montiel": "Montiel",
    "Cristian Gabriel Romero": "Romero"
}

def shorten_name(full_name):
    if full_name in manual_names:
        return manual_names[full_name]
    else:
        return full_name.split()[-1]

player_attack_involvement["short_name"] = player_attack_involvement["player.name"].apply(shorten_name)

# Reset index for plotting
player_attack_involvement = player_attack_involvement.reset_index(drop=True)

#%% xT modelling

# Function to map locations to pitch zones safely
def get_zone(location, x_bins=12, y_bins=8, pitch_x=120, pitch_y=80):
    if isinstance(location, (list, tuple)) and len(location) == 2:
        x, y = location
        x_zone = min(int(x / (pitch_x / x_bins)), x_bins - 1)
        y_zone = min(int(y / (pitch_y / y_bins)), y_bins - 1)
        return x_zone, y_zone
    else:
        return None  # handle missing or malformed locations


# Build simplified xT grid
x_bins, y_bins = 12, 8
xT_matrix = np.zeros((x_bins, y_bins))

for i in range(x_bins):
    for j in range(y_bins):
        x_norm = i / (x_bins - 1)
        y_norm = abs(j - (y_bins / 2)) / (y_bins / 2)

        # Heuristic: more threat closer to opponent goal + central zones
        xT_matrix[i, j] = 0.05 + 0.45 * x_norm + 0.25 * (1 - y_norm)

# Normalize values to 0–1
xT_matrix = xT_matrix / xT_matrix.max()

def get_end_location(row):
    for col in ['pass.end_location', 'carry.end_location', 'shot.end_location', 'goalkeeper.end_location']:
        if col in row and isinstance(row[col], (list, tuple)) and len(row[col]) == 2:
            return row[col]
    return None

# Unify end_location
all_events['end_location'] = all_events.apply(get_end_location, axis=1)


# Map zones (start & end)
all_events['start_zone'] = all_events['location'].apply(get_zone)
all_events['end_zone']   = all_events['end_location'].apply(get_zone)


# Assign xT values to start & end
def get_xT_from_zone(zone):
    if zone is None:
        return np.nan
    return xT_matrix[zone[0], zone[1]]

all_events['xT_start'] = all_events['start_zone'].apply(get_xT_from_zone)
all_events['xT_end']   = all_events['end_zone'].apply(get_xT_from_zone)


# Compute xT contribution per action
def compute_xT(row):
    if pd.isna(row['xT_start']) or pd.isna(row['xT_end']):
        return 0.0

    if row['type.name'] in ['Pass', 'Carry']:
        return row['xT_end'] - row['xT_start']

    elif row['type.name'] == 'Shot':
        if 'shot.statsbomb_xg' in row and pd.notna(row['shot.statsbomb_xg']):
            return row['shot.statsbomb_xg'] - row['xT_start']

    return 0.0

all_events['xT_value'] = all_events.apply(compute_xT, axis=1)


# Aggregate per player
player_xT = (
    all_events.groupby('player.name')['xT_value']
    .sum()
    .reset_index()
    .rename(columns={'xT_value': 'xT'})
    .sort_values('xT', ascending=False)
)

#%% Player involvement stats for xT chart

# Assign unique possession IDs
all_events['team_shift'] = all_events['possession_team.name'] != all_events['possession_team.name'].shift(1)
all_events['possession_id'] = all_events['team_shift'].cumsum()
all_events = all_events.copy()  # defragmented copy

# Aggregate player stats
player_stats = all_events.groupby('player.name').agg(
    total_possessions=('possession_id', 'nunique'),
    total_touches=('player.name', 'count')  # total events = touches
).reset_index()

# Average touches per possession
player_stats['avg_touches_per_possession'] = (
    player_stats['total_touches'] / player_stats['total_possessions']
)

# Merge with xT
player_stats = player_stats.merge(player_xT, on='player.name')

# Filter for players with enough possessions
player_stats = player_stats[player_stats['total_possessions'] >= 100]

player_stats["short_name"] = player_stats["player.name"].apply(shorten_name)

player_stats = player_stats[player_stats["short_name"] != "Emiliano Martínez"]

# Total team possessions & touches
team_total_possessions = player_stats['total_possessions'].sum()
team_total_touches = player_stats['total_touches'].sum()

# % of total possessions
player_stats['pct_possessions'] = (player_stats['total_possessions'] / team_total_possessions) * 100

# % of team total touches
player_stats['pct_touches'] = (player_stats['total_touches'] / team_total_touches) * 100

#%% Dash App
app = Dash(__name__)

# Chart 1 - Player Efficiency vs xG+xA
player_attack_involvement["size_uniform"] = 25

fig1 = px.scatter(
    player_attack_involvement,
    x="xG_plus_xA",
    y="goals_plus_assists",
    size="size_uniform",
    color="efficiency_%",
    text="short_name",
    color_continuous_scale=['blue', 'red']
)

fig1.update_traces(
    textposition="top center",
    marker=dict(line=dict(width=1, color="black")),
    hovertemplate=(
        "<b>%{customdata[0]}</b><br><br>"
        "Total Shots: %{customdata[1]}<br>"
        "xG: %{customdata[2]:.2f}<br>"
        "Goals: %{customdata[3]}<br><br>"
        "Assisted Shots: %{customdata[4]}<br>"
        "xA: %{customdata[5]:.2f}<br>"
        "Assists: %{customdata[6]}<br><br>"
        "Total Chances: %{customdata[7]}<br>"
        "Chance Conversion: %{customdata[8]:.1f}%<br>"
        "Efficiency: %{customdata[9]:.1f}%<extra></extra>"
    ),
    customdata=np.stack([
        player_attack_involvement["player.name"],
        player_attack_involvement["total_shots"],
        player_attack_involvement["total_xG"],
        player_attack_involvement["total_goals"],
        player_attack_involvement["total_assisted_shots"],
        player_attack_involvement["total_xA"],
        player_attack_involvement["total_assists"],
        player_attack_involvement["total_chances"],
        player_attack_involvement["conversion_%"],
        player_attack_involvement["efficiency_%"]
    ], axis=-1)
)

fig1.update_layout(
    title_text="Player Efficiency vs xG+xA (Copa America 2024)",
    title_x=0.5,
    font=dict(size=13),
    xaxis_title="xG + xA",
    yaxis_title="Goals + Assists",
    coloraxis_colorbar=dict(title="Efficiency %")
)

# Chart 2 - Possessions vs xT
player_stats['size_uniform'] = 25

fig2 = px.scatter(
    player_stats,
    x='avg_touches_per_possession',
    y='total_possessions',
    text='short_name',
    color='xT',
    color_continuous_scale=['blue', 'red'],
    size="size_uniform"
)

fig2.update_traces(
    marker=dict(sizemode='area'),
    textposition='top center',
    hovertemplate=(
        "<b>%{customdata[0]}</b><br><br>"
        "xT: %{customdata[1]:.2f}<br><br>"
        "Total Possessions Involved: %{customdata[2]}<br>"
        "Average Touches Per Possession: %{customdata[3]:.2f}<br><br>"
        "Percentage of Total Team Possessions: %{customdata[4]:.1f}%<br>"
        "Percentage of Total Player Touches: %{customdata[5]:.1f}%<extra></extra>"
    ),
    customdata=np.stack([
        player_stats['player.name'],
        player_stats['xT'],
        player_stats['total_possessions'],
        player_stats['avg_touches_per_possession'],
        player_stats['pct_possessions'],
        player_stats['pct_touches']
    ], axis=-1)
)

fig2.update_layout(
    title="Player Involvement vs xT",
    title_x=0.5,
    font=dict(size=13),
    xaxis_title="Average Touches Per Possession",
    yaxis_title="Total Possessions",
    coloraxis_colorbar=dict(title="xT Score")
)

#%% Dash formatting

# Narrative text
narrative1 = """
**Overall Team Threat and Efficiency**  
While Argentina showed solid attacking potential across the Copa América, a closer look at the underlying metrics reveals a divergence between expected contributions and actual output. This highlights the need to go beyond simple goal and assist counts and consider xG, xA, and xT to fully understand player efficiency and attacking influence.  
"""
narrative2 = """
**xG+xA Insights**  
Lionel Messi recorded just 1 goal and 1 assist over the six games, yet his combined xG and xA totaled 4.17. This gave him an efficiency of 48% and a very low conversion rate of 7.7%, indicating that he underperformed relative to the chances he created and received, making him largely unproductive in terms of final output. Lautaro Martínez, on the other hand, scored 5 goals, outperforming his xG of 2.89 to achieve an efficiency of 173%, with a team-best conversion rate of 45% of his shots finding the net. Alexis MacAllister quietly excelled as well, surpassing his combined xG and xA of 1.42 with 2 assists, yielding an efficiency of 141% and a conversion rate of 22%.
"""
narrative3 = """
**xT Insights**  
Focusing on xT contributions, Nicolás Tagliafico led the team with 20.75 xT. Despite only being involved in 8.1% of team possessions, he consistently looked to move the ball into dangerous areas, outperforming his teammates in terms of threat creation. Conversely, outside of his goal-scoring, Martínez contributed little to advancing attacking opportunities, registering -2.51 xT while being involved in just 4.8% of the team’s possessions.
"""
app.layout = html.Div([
    html.Div([
        html.H1(
            "Argentina Player Efficiency:",
            style={'textAlign': 'center',
                   'color': 'black',
                   'fontWeight': 'bold',
                   'WebkitTextStroke': '0.5px gold',
                   'fontSize': '35px',
                   'margin': '0',
                   'padding': '0px',
                   'fontFamily': '"Oswald", "Arial Black", sans-serif'}
        ),
        html.H2(
            "Actual vs Expected Performance",
            style={'textAlign': 'center',
                   'color': 'black',
                   'fontWeight': 'bold',
                   'WebkitTextStroke': '0.5px gold',
                   'fontSize': '35px',
                   'margin': '0',
                   'padding': '0px',
                   'fontFamily': '"Oswald", "Arial Black", sans-serif'}
        ),
        html.H3(
            "Copa America 2024",
            style={'textAlign': 'center',
                   'color': ' solid black',
                   'WebkitTextStroke': '0.5px black',
                   'fontSize': '25px',
                   'margin': '0',
                   'padding': '10px',
                   'fontFamily': '"Oswald", "Arial Black", sans-serif'}
        ),
    ], style={'borderBottom': '5px solid black',
              'backgroundImage': 'repeating-linear-gradient(90deg, lightblue 0, lightblue 51px, white 51px, white 102px)', 
              'padding': '10px','marginBottom': '20px'}
    ),
    dcc.Markdown(narrative1),
    dcc.Markdown(narrative2),
    dcc.Graph(figure=fig1),
    dcc.Markdown(narrative3),
    dcc.Graph(figure=fig2),
])

# Run App
if __name__ == "__main__":
    app.run(debug=True)