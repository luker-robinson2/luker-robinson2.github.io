#!/usr/bin/env python3
"""
Round-Based Interactive Map Visualizer
======================================

Enhanced visualizer with round-by-round analysis and precise time scrubbing.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from pathlib import Path

# Load position data and round information
def load_data():
    data_dir = Path("example_position_data")
    csv_files = list(data_dir.glob("*_positions_timeseries.csv"))
    
    if not csv_files:
        print("No position data files found!")
        return {}, {}
    
    position_data = {}
    round_data = {}
    
    for csv_file in csv_files[:1]:  # Load just one file for now
        demo_name = csv_file.stem.replace('_positions_timeseries', '')
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['X', 'Y', 'player_name'])
        position_data[demo_name] = df
        
        # Extract round information from existing data
        round_info = extract_rounds(df, demo_name)
        round_data[demo_name] = round_info
        
        print(f"Loaded {len(df)} records for {demo_name}")
        print(f"Detected {len(round_info)} rounds")
    
    return position_data, round_data

def extract_rounds(df, demo_name):
    """
    Extract actual round information using event data and round boundaries.
    Falls back to time-based estimation if no event data is available.
    """
    rounds = []
    
    try:
        # Try to load actual round data
        bomb_df = pd.read_csv('bomb_events_cleaned.csv')
        deaths_df = pd.read_csv('deaths_cleaned.csv')
        rounds_df = pd.read_csv('rounds_cleaned.csv')
        
        # Filter for this specific match
        match_bombs = bomb_df[bomb_df['match_id'] == demo_name].copy()
        match_deaths = deaths_df[deaths_df['match_id'] == demo_name].copy()
        match_rounds = rounds_df[rounds_df['match_id'] == demo_name].copy()
        
        if not match_rounds.empty:
            print(f"Using actual round data for {demo_name}")
            
            # Convert ticks to timestamps (assuming 128 tick rate)
            tick_rate = 128
            match_bombs['timestamp'] = match_bombs['tick'] / tick_rate
            match_deaths['timestamp'] = match_deaths['tick'] / tick_rate
            
            # Use bomb plants as round markers
            bomb_plants = match_bombs[match_bombs['event_type'] == 'planted'].sort_values('timestamp')
            
            if len(bomb_plants) > 0:
                for i, (_, plant) in enumerate(bomb_plants.iterrows()):
                    # Estimate round start (60s before bomb plant)
                    round_start = max(0, plant['timestamp'] - 60)
                    
                    # Estimate round end (look for next plant or use duration)
                    if i < len(bomb_plants) - 1:
                        next_plant = bomb_plants.iloc[i + 1]
                        round_end = next_plant['timestamp'] - 30
                    else:
                        round_end = min(plant['timestamp'] + 45, df['timestamp'].max())
                    
                    # Get round metadata
                    round_num = i + 1
                    round_info = match_rounds[match_rounds['round_num'] == round_num]
                    
                    if round_end > round_start:  # Valid round
                        round_data = {
                            'round_num': round_num,
                            'start_time': round_start,
                            'end_time': round_end,
                            'duration': round_end - round_start,
                            'players': df[(df['timestamp'] >= round_start) & 
                                        (df['timestamp'] <= round_end)]['player_name'].nunique()
                        }
                        
                        # Add metadata if available
                        if not round_info.empty:
                            round_data.update({
                                'winning_team': round_info.iloc[0]['winning_team'],
                                'round_end_reason': round_info.iloc[0]['round_end_reason'],
                                'bomb_planted': round_info.iloc[0]['bomb_planted']
                            })
                        
                        rounds.append(round_data)
            
            if rounds:
                print(f"Detected {len(rounds)} rounds using bomb plant markers")
                return rounds
        
    except FileNotFoundError:
        print("Event data files not found, using time-based estimation")
    except Exception as e:
        print(f"Error loading event data: {e}, using time-based estimation")
    
    # Fallback: time-based round estimation
    print("Using time-based round estimation")
    total_time = df['timestamp'].max() - df['timestamp'].min()
    estimated_round_length = 120  # 2 minutes average
    num_rounds = max(1, int(total_time / estimated_round_length))
    
    start_time = df['timestamp'].min()
    
    for i in range(num_rounds):
        round_start = start_time + (i * estimated_round_length)
        round_end = min(start_time + ((i + 1) * estimated_round_length), df['timestamp'].max())
        
        # Get data for this round
        round_df = df[(df['timestamp'] >= round_start) & (df['timestamp'] <= round_end)]
        
        if len(round_df) > 0:
            rounds.append({
                'round_num': i + 1,
                'start_time': round_start,
                'end_time': round_end,
                'duration': round_end - round_start,
                'players': round_df['player_name'].nunique(),
                'winning_team': 'Unknown',
                'round_end_reason': 'estimated',
                'bomb_planted': False
            })
    
    return rounds

# Initialize
position_data, round_data = load_data()
if not position_data:
    print("No data loaded!")
    exit(1)

demo_name = list(position_data.keys())[0]
df = position_data[demo_name]
rounds = round_data[demo_name]

# Create app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("CS2/CS:GO Round-by-Round Analysis", style={'textAlign': 'center', 'color': 'white'}),
    
    # Round selector
    html.Div([
        html.Div([
            html.Label("Select Round:", style={'color': 'white', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='round-dropdown',
                options=[{'label': f"Round {r['round_num']} ({r['duration']:.0f}s)", 'value': i} 
                        for i, r in enumerate(rounds)],
                value=0,
                style={'marginBottom': '10px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
        
        html.Div([
            html.Button('Play Round', id='play-button', n_clicks=0,
                       style={'marginRight': '10px', 'backgroundColor': '#4CAF50', 'color': 'white', 
                             'border': 'none', 'padding': '8px 16px', 'borderRadius': '4px'}),
            html.Button('Pause', id='pause-button', n_clicks=0,
                       style={'marginRight': '10px', 'backgroundColor': '#f44336', 'color': 'white', 
                             'border': 'none', 'padding': '8px 16px', 'borderRadius': '4px'}),
            html.Button('Reset', id='reset-button', n_clicks=0,
                       style={'backgroundColor': '#008CBA', 'color': 'white', 
                             'border': 'none', 'padding': '8px 16px', 'borderRadius': '4px'})
        ], style={'width': '40%', 'display': 'inline-block', 'textAlign': 'right'})
    ], style={'margin': '20px', 'padding': '10px', 'backgroundColor': '#2E2E2E', 'borderRadius': '5px'}),
    
    # Round info display
    html.Div(id='round-info', style={'margin': '10px 20px', 'color': 'white'}),
    
    # Time slider for selected round
    html.Div([
        html.Label("Round Timeline:", style={'color': 'white', 'marginBottom': '10px'}),
        dcc.Slider(
            id='time-slider',
            min=0,
            max=100,
            value=0,
            marks={},
            tooltip={"placement": "bottom", "always_visible": True},
            step=1
        )
    ], style={'margin': '20px'}),
    
    # Map plot
    dcc.Graph(id='map-plot', style={'height': '600px'}),
    
    # Auto-update interval for playback
    dcc.Interval(
        id='interval-component',
        interval=500,  # Update every 500ms
        n_intervals=0,
        disabled=True
    ),
    
    # Store for playback state
    dcc.Store(id='playback-state', data={'playing': False})
    
], style={'backgroundColor': '#1E1E1E', 'minHeight': '100vh', 'padding': '20px'})

# Callback to update time slider when round is selected
@app.callback(
    [Output('time-slider', 'min'),
     Output('time-slider', 'max'),
     Output('time-slider', 'marks'),
     Output('time-slider', 'value'),
     Output('round-info', 'children')],
    Input('round-dropdown', 'value')
)
def update_round_slider(round_index):
    if round_index is None or round_index >= len(rounds):
        round_index = 0
    
    selected_round = rounds[round_index]
    start_time = selected_round['start_time']
    end_time = selected_round['end_time']
    duration = selected_round['duration']
    
    # Create marks for the round timeline
    marks = {}
    num_marks = min(10, int(duration / 10))  # Mark every 10 seconds, max 10 marks
    if num_marks > 1:
        for i in range(num_marks + 1):
            rel_time = (duration * i / num_marks)
            abs_time = start_time + rel_time
            marks[abs_time] = f"{rel_time:.0f}s"
    else:
        marks[start_time] = "0s"
        marks[end_time] = f"{duration:.0f}s"
    
    # Round info with enhanced metadata
    winner_info = ""
    if 'winning_team' in selected_round and selected_round['winning_team'] != 'Unknown':
        winner_info = f" | Winner: {selected_round['winning_team']}"
    
    reason_info = ""
    if 'round_end_reason' in selected_round and selected_round['round_end_reason'] != 'estimated':
        reason_info = f" | End: {selected_round['round_end_reason']}"
    
    bomb_info = ""
    if 'bomb_planted' in selected_round and selected_round['bomb_planted']:
        bomb_info = " | 💣 Bomb"
    
    round_info = html.Div([
        html.H4(f"Round {selected_round['round_num']}", style={'color': '#4CAF50', 'margin': '0'}),
        html.P(f"Duration: {duration:.1f}s | Players: {selected_round['players']}{winner_info}{reason_info}{bomb_info}", 
               style={'margin': '5px 0', 'color': '#CCCCCC'})
    ])
    
    return start_time, end_time, marks, start_time, round_info

# Callback for playback controls
@app.callback(
    [Output('interval-component', 'disabled'),
     Output('playback-state', 'data')],
    [Input('play-button', 'n_clicks'),
     Input('pause-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('playback-state', 'data')]
)
def control_playback(play_clicks, pause_clicks, reset_clicks, playback_state):
    ctx = callback_context
    
    if not ctx.triggered:
        return True, playback_state
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'play-button':
        playback_state['playing'] = True
        return False, playback_state  # Enable interval
    elif button_id == 'pause-button':
        playback_state['playing'] = False
        return True, playback_state   # Disable interval
    elif button_id == 'reset-button':
        playback_state['playing'] = False
        return True, playback_state   # Disable interval, value will be reset by other callback
    
    return True, playback_state

# Callback for auto-advancing time slider during playback
@app.callback(
    Output('time-slider', 'value', allow_duplicate=True),
    [Input('interval-component', 'n_intervals'),
     Input('reset-button', 'n_clicks'),
     Input('round-dropdown', 'value')],
    [State('time-slider', 'value'),
     State('time-slider', 'max'),
     State('time-slider', 'min'),
     State('playback-state', 'data')],
    prevent_initial_call=True
)
def update_time_value(n_intervals, reset_clicks, round_index, current_time, max_time, min_time, playback_state):
    ctx = callback_context
    
    if not ctx.triggered:
        return current_time or min_time or 0
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Reset to start when round changes or reset button clicked
    if trigger_id in ['reset-button', 'round-dropdown']:
        return min_time or 0
    
    # Auto-advance during playback
    if trigger_id == 'interval-component' and playback_state and playback_state.get('playing', False):
        if current_time is None or max_time is None or min_time is None:
            return min_time or 0
        
        new_time = current_time + 1.0  # Advance by 1 second
        if new_time > max_time:
            return min_time  # Loop back to start of round
        return new_time
    
    return current_time or min_time or 0

# Main callback for updating the map visualization
@app.callback(
    Output('map-plot', 'figure'),
    [Input('time-slider', 'value'),
     Input('round-dropdown', 'value')]
)
def update_map(timestamp, round_index):
    if timestamp is None:
        timestamp = df['timestamp'].min()
    
    # Get data for this timestamp (within 5 second window)
    frame_data = df[abs(df['timestamp'] - timestamp) <= 5.0]
    
    if frame_data.empty:
        # If no data at timestamp, get data from start of match
        frame_data = df[df['timestamp'] <= df['timestamp'].min() + 10]
    
    # Get latest position for each player
    if not frame_data.empty:
        frame_data = frame_data.groupby('player_name').last().reset_index()
    
    # Create figure
    fig = go.Figure()
    
    if not frame_data.empty:
        # Add players by team
        teams = frame_data['team_name'].unique() if 'team_name' in frame_data.columns else ['Unknown']
        colors = ['gold', 'royalblue', 'red', 'green']
        
        for i, team in enumerate(teams):
            if 'team_name' in frame_data.columns:
                team_data = frame_data[frame_data['team_name'] == team]
            else:
                team_data = frame_data
                
            if team_data.empty:
                continue
            
            # Map team names
            team_display = 'Terrorists' if team in ['T', 'TERRORIST'] else 'Counter-Terrorists' if team in ['CT', 'COUNTER_TERRORIST'] else str(team)
                
            fig.add_trace(go.Scatter(
                x=team_data['X'],
                y=team_data['Y'],
                mode='markers+text',
                marker=dict(
                    size=15, 
                    color=colors[i % len(colors)],
                    line=dict(width=3, color='white')
                ),
                text=team_data['player_name'],
                textposition="top center",
                textfont=dict(color='white', size=12),
                name=team_display,
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    f"Team: {team_display}<br>" +
                    "Position: (%{x:.0f}, %{y:.0f})<br>" +
                    "<extra></extra>"
                )
            ))
    else:
        # Add a dummy trace to ensure the plot shows
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=1, color='transparent'),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_annotation(
            x=0, y=0,
            text="Loading player data...",
            showarrow=False,
            font=dict(color='white', size=16)
        )
    
    # Get round info for title
    round_info = ""
    if round_index is not None and round_index < len(rounds):
        selected_round = rounds[round_index]
        round_info = f"Round {selected_round['round_num']} - "
    
    fig.update_layout(
        title=f"{round_info}Player Positions at {timestamp:.1f}s",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        plot_bgcolor='darkgreen',
        paper_bgcolor='black',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)"),
        height=600
    )
    
    return fig

if __name__ == '__main__':
    print("Starting simple visualizer...")
    print("Go to: http://localhost:8051")
    print("If dots don't appear, try refreshing the page or clearing browser cache")
    app.run(debug=False, host='127.0.0.1', port=8051, dev_tools_hot_reload=False)
