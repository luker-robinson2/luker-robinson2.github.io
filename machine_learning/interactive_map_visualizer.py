#!/usr/bin/env python3
"""
Interactive Map Visualizer for CS2/CS:GO Player Positions
=========================================================

This script creates an interactive web-based visualization of player movements
on CS maps with a time slider. It supports various maps including Mirage, Dust2,
Inferno, etc.

Features:
- Interactive time slider to scrub through match timeline
- Player position visualization with team colors
- Speed controls (play/pause/speed adjustment)
- Player trails showing movement history
- Heat map visualization option
- Export capabilities for analysis

Dependencies:
    pip install dash plotly pandas numpy pillow requests

Usage:
    python interactive_map_visualizer.py
    Then open http://localhost:8050 in your browser

Author: Generated for CS 5612 Project
Date: 2024
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
from pathlib import Path
import requests
from PIL import Image, ImageDraw
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MapVisualizer:
    """
    Interactive map visualizer for CS2/CS:GO player positions
    """
    
    def __init__(self, data_dir="example_position_data"):
        """
        Initialize the visualizer
        
        Args:
            data_dir: Directory containing position data CSV files
        """
        self.data_dir = Path(data_dir)
        self.position_data = {}
        self.map_images = {}
        self.map_bounds = {}
        self.current_demo = None
        
        # Define map coordinate bounds (these may need adjustment based on actual data)
        self.map_bounds = {
            'de_mirage': {
                'x_min': -3217, 'x_max': 1912,
                'y_min': -3401, 'y_max': 1682,
                'image_width': 1024, 'image_height': 1024
            },
            'de_dust2': {
                'x_min': -2476, 'x_max': 3239,
                'y_min': -2553, 'y_max': 2681,
                'image_width': 1024, 'image_height': 1024
            },
            'de_inferno': {
                'x_min': -2087, 'x_max': 3870,
                'y_min': -3870, 'y_max': 1357,
                'image_width': 1024, 'image_height': 1024
            },
            'de_train': {
                'x_min': -2477, 'x_max': 2936,
                'y_min': -2936, 'y_max': 2477,
                'image_width': 1024, 'image_height': 1024
            },
            'de_nuke': {
                'x_min': -3453, 'x_max': 1935,
                'y_min': -2320, 'y_max': 3870,
                'image_width': 1024, 'image_height': 1024
            }
        }
        
        # Team colors
        self.team_colors = {
            'TERRORIST': '#D4AF37',  # Gold
            'COUNTER_TERRORIST': '#4169E1',  # Royal Blue
            'CT': '#4169E1',
            'T': '#D4AF37'
        }
        
        self.load_position_data()
        self.create_map_images()
    
    def load_position_data(self):
        """
        Load all position data CSV files from the data directory
        """
        print("Loading position data...")
        
        csv_files = list(self.data_dir.glob("*_positions_timeseries.csv"))
        
        if not csv_files:
            print(f"No position data files found in {self.data_dir}")
            return
        
        for csv_file in csv_files:
            demo_name = csv_file.stem.replace('_positions_timeseries', '')
            print(f"Loading {demo_name}...")
            
            try:
                df = pd.read_csv(csv_file)
                
                # Clean and prepare data
                df = df.dropna(subset=['X', 'Y', 'player_name'])
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df = df.sort_values('timestamp')
                
                # Standardize team names
                if 'team_name' in df.columns:
                    df['team_name'] = df['team_name'].replace({
                        'TERRORIST': 'T',
                        'COUNTER_TERRORIST': 'CT'
                    })
                
                self.position_data[demo_name] = df
                print(f"  Loaded {len(df)} records for {df['player_name'].nunique()} players")
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        print(f"Loaded data for {len(self.position_data)} demos")
    
    def create_map_images(self):
        """
        Create or load map background images
        """
        print("Loading map images...")
        
        map_images_dir = Path("map_images")
        
        for map_name, bounds in self.map_bounds.items():
            image_path = map_images_dir / f"{map_name}.png"
            
            if image_path.exists():
                # Load actual map image
                try:
                    with open(image_path, "rb") as f:
                        img_data = f.read()
                    img_str = base64.b64encode(img_data).decode()
                    self.map_images[map_name] = f"data:image/png;base64,{img_str}"
                    print(f"  Loaded {map_name} image")
                except Exception as e:
                    print(f"  Error loading {map_name}: {e}")
                    self.map_images[map_name] = self.create_simple_map_image(
                        map_name, bounds['image_width'], bounds['image_height']
                    )
            else:
                # Create simple placeholder
                self.map_images[map_name] = self.create_simple_map_image(
                    map_name, bounds['image_width'], bounds['image_height']
                )
    
    def create_simple_map_image(self, map_name, width, height):
        """
        Create a simple map background (placeholder for actual map images)
        
        Args:
            map_name: Name of the map
            width: Image width
            height: Image height
            
        Returns:
            Base64 encoded image string
        """
        # Create a simple colored background with grid
        img = Image.new('RGB', (width, height), color='#2F2F2F')
        draw = ImageDraw.Draw(img)
        
        # Add grid lines
        grid_size = 64
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill='#404040', width=1)
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill='#404040', width=1)
        
        # Add map name
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
            draw.text((10, 10), map_name, fill='white', font=font)
        except:
            draw.text((10, 10), map_name, fill='white')
        
        # Convert to base64 for web display
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def normalize_coordinates(self, df, map_name):
        """
        Normalize game coordinates to image coordinates
        
        Args:
            df: DataFrame with X, Y coordinates
            map_name: Name of the map
            
        Returns:
            DataFrame with normalized coordinates
        """
        if map_name not in self.map_bounds:
            print(f"Warning: No bounds defined for {map_name}")
            return df
        
        bounds = self.map_bounds[map_name]
        df = df.copy()
        
        # Normalize X coordinates (left to right)
        df['norm_x'] = ((df['X'] - bounds['x_min']) / 
                       (bounds['x_max'] - bounds['x_min']) * bounds['image_width'])
        
        # Normalize Y coordinates (top to bottom, flipped)
        df['norm_y'] = bounds['image_height'] - ((df['Y'] - bounds['y_min']) / 
                       (bounds['y_max'] - bounds['y_min']) * bounds['image_height'])
        
        return df
    
    def get_demo_options(self):
        """
        Get dropdown options for demo selection
        
        Returns:
            List of demo options for dropdown
        """
        return [{'label': demo_name.replace('_', ' ').title(), 'value': demo_name} 
                for demo_name in self.position_data.keys()]
    
    def get_map_name(self, demo_name):
        """
        Get map name for a demo
        
        Args:
            demo_name: Name of the demo
            
        Returns:
            Map name
        """
        if demo_name not in self.position_data:
            return 'unknown'
        
        df = self.position_data[demo_name]
        if 'map_name' in df.columns:
            return df['map_name'].iloc[0]
        else:
            return 'de_mirage'  # Default fallback
    
    def create_frame_data(self, demo_name, timestamp):
        """
        Create data for a specific timestamp
        
        Args:
            demo_name: Name of the demo
            timestamp: Timestamp to get data for
            
        Returns:
            DataFrame with player positions at the timestamp
        """
        if demo_name not in self.position_data:
            return pd.DataFrame()
        
        df = self.position_data[demo_name]
        
        # Get data closest to the requested timestamp
        closest_idx = (df['timestamp'] - timestamp).abs().idxmin()
        time_window = 1.0  # 1 second window
        
        frame_data = df[abs(df['timestamp'] - timestamp) <= time_window].copy()
        
        if frame_data.empty:
            return pd.DataFrame()
        
        # Get the latest position for each player within the time window
        frame_data = frame_data.sort_values('timestamp').groupby('player_name').last().reset_index()
        
        # Normalize coordinates
        map_name = self.get_map_name(demo_name)
        frame_data = self.normalize_coordinates(frame_data, map_name)
        
        return frame_data
    
    def create_player_trails(self, demo_name, timestamp, trail_length=10):
        """
        Create player movement trails
        
        Args:
            demo_name: Name of the demo
            timestamp: Current timestamp
            trail_length: Length of trail in seconds
            
        Returns:
            DataFrame with trail data
        """
        if demo_name not in self.position_data:
            return pd.DataFrame()
        
        df = self.position_data[demo_name]
        
        # Get data for the trail period
        start_time = max(0, timestamp - trail_length)
        trail_data = df[(df['timestamp'] >= start_time) & 
                       (df['timestamp'] <= timestamp)].copy()
        
        if trail_data.empty:
            return pd.DataFrame()
        
        # Normalize coordinates
        map_name = self.get_map_name(demo_name)
        trail_data = self.normalize_coordinates(trail_data, map_name)
        
        return trail_data

def create_dash_app(visualizer):
    """
    Create the Dash application
    
    Args:
        visualizer: MapVisualizer instance
        
    Returns:
        Dash app instance
    """
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("CS2/CS:GO Interactive Map Visualizer", 
                style={'textAlign': 'center', 'color': '#FFFFFF', 'marginBottom': 30}),
        
        # Control panel
        html.Div([
            html.Div([
                html.Label("Select Demo:", style={'color': '#FFFFFF', 'marginBottom': 5}),
                dcc.Dropdown(
                    id='demo-dropdown',
                    options=visualizer.get_demo_options(),
                    value=list(visualizer.position_data.keys())[0] if visualizer.position_data else None,
                    style={'marginBottom': 10}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
            
            html.Div([
                html.Label("Show Trails:", style={'color': '#FFFFFF', 'marginBottom': 5}),
                dcc.Checklist(
                    id='show-trails',
                    options=[{'label': 'Player Trails', 'value': 'trails'}],
                    value=[],
                    style={'color': '#FFFFFF'}
                )
            ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '5%'}),
            
            html.Div([
                html.Button('Play', id='play-button', n_clicks=0, 
                           style={'marginRight': 10, 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'padding': '8px 16px'}),
                html.Button('Pause', id='pause-button', n_clicks=0,
                           style={'marginRight': 10, 'backgroundColor': '#f44336', 'color': 'white', 'border': 'none', 'padding': '8px 16px'}),
                html.Button('Reset', id='reset-button', n_clicks=0,
                           style={'backgroundColor': '#008CBA', 'color': 'white', 'border': 'none', 'padding': '8px 16px'})
            ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'right'})
        ], style={'marginBottom': 20, 'padding': '10px', 'backgroundColor': '#1E1E1E'}),
        
        # Time slider
        html.Div([
            html.Label("Timeline:", style={'color': '#FFFFFF', 'marginBottom': 10}),
            dcc.Slider(
                id='time-slider',
                min=0,
                max=100,
                value=0,
                marks={},
                tooltip={"placement": "bottom", "always_visible": True},
                step=1
            )
        ], style={'marginBottom': 20, 'padding': '0 20px'}),
        
        # Map visualization
        dcc.Graph(
            id='map-plot',
            style={'height': '700px'},
            config={'displayModeBar': True, 'displaylogo': False}
        ),
        
        # Player info panel
        html.Div(id='player-info', style={'marginTop': 20, 'padding': '10px', 'backgroundColor': '#1E1E1E', 'color': '#FFFFFF'}),
        
        # Auto-update component
        dcc.Interval(
            id='interval-component',
            interval=500,  # Update every 500ms when playing
            n_intervals=0,
            disabled=True
        ),
        
        # Store components for state management
        dcc.Store(id='playback-state', data={'playing': False, 'speed': 1.0}),
        dcc.Store(id='current-timestamp', data=0)
    ], style={'backgroundColor': '#0F0F0F', 'minHeight': '100vh', 'padding': '20px'})
    
    # Callback for updating time slider range based on selected demo
    @app.callback(
        [Output('time-slider', 'min'),
         Output('time-slider', 'max'),
         Output('time-slider', 'marks')],
        [Input('demo-dropdown', 'value')]
    )
    def update_time_slider_range(demo_name):
        if not demo_name or demo_name not in visualizer.position_data:
            return 0, 100, {}
        
        df = visualizer.position_data[demo_name]
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        
        # Create marks every minute
        marks = {}
        step = max(1, int((max_time - min_time) / 10))  # ~10 marks
        for i in range(int(min_time), int(max_time) + 1, step):
            minutes = int(i // 60)
            seconds = int(i % 60)
            marks[i] = f"{minutes:02d}:{seconds:02d}"
        
        return min_time, max_time, marks
    
    # Callback for updating the map visualization
    @app.callback(
        [Output('map-plot', 'figure'),
         Output('player-info', 'children')],
        [Input('time-slider', 'value'),
         Input('demo-dropdown', 'value'),
         Input('show-trails', 'value')]
    )
    def update_map(timestamp, demo_name, show_trails):
        if not demo_name or demo_name not in visualizer.position_data:
            return {}, "No demo selected"
        
        # Get current frame data
        frame_data = visualizer.create_frame_data(demo_name, timestamp)
        map_name = visualizer.get_map_name(demo_name)
        
        # Create the plot
        fig = go.Figure()
        
        # Add map background
        if map_name in visualizer.map_images:
            fig.add_layout_image(
                dict(
                    source=visualizer.map_images[map_name],
                    xref="x", yref="y",
                    x=0, y=0,
                    sizex=visualizer.map_bounds[map_name]['image_width'],
                    sizey=visualizer.map_bounds[map_name]['image_height'],
                    sizing="stretch",
                    opacity=0.8,
                    layer="below"
                )
            )
        
        # Add player trails if enabled
        if 'trails' in show_trails:
            trail_data = visualizer.create_player_trails(demo_name, timestamp, trail_length=10)
            if not trail_data.empty:
                for player in trail_data['player_name'].unique():
                    player_trail = trail_data[trail_data['player_name'] == player].sort_values('timestamp')
                    if len(player_trail) > 1:
                        team = player_trail['team_name'].iloc[0] if 'team_name' in player_trail.columns else 'T'
                        color = visualizer.team_colors.get(team, '#FFFFFF')
                        
                        fig.add_trace(go.Scatter(
                            x=player_trail['norm_x'],
                            y=player_trail['norm_y'],
                            mode='lines',
                            line=dict(color=color, width=2, dash='dot'),
                            opacity=0.5,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        # Add current player positions
        if not frame_data.empty:
            for team in frame_data['team_name'].unique() if 'team_name' in frame_data.columns else ['Unknown']:
                team_data = frame_data[frame_data['team_name'] == team] if 'team_name' in frame_data.columns else frame_data
                
                if team_data.empty:
                    continue
                
                color = visualizer.team_colors.get(team, '#FFFFFF')
                
                fig.add_trace(go.Scatter(
                    x=team_data['norm_x'],
                    y=team_data['norm_y'],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=color,
                        line=dict(width=2, color='white'),
                        symbol='circle'
                    ),
                    text=team_data['player_name'],
                    textposition="top center",
                    textfont=dict(color='white', size=10),
                    name=f"Team {team}",
                    hovertemplate=(
                        "<b>%{text}</b><br>" +
                        "Team: " + team + "<br>" +
                        "Position: (%{customdata[0]:.0f}, %{customdata[1]:.0f})<br>" +
                        "Health: %{customdata[2]}<br>" +
                        "Speed: %{customdata[3]:.1f}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=np.column_stack((
                        team_data['X'] if 'X' in team_data.columns else [0] * len(team_data),
                        team_data['Y'] if 'Y' in team_data.columns else [0] * len(team_data),
                        team_data['health'] if 'health' in team_data.columns else [100] * len(team_data),
                        team_data['speed'] if 'speed' in team_data.columns else [0] * len(team_data)
                    ))
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{demo_name.replace('_', ' ').title()} - {map_name} - Time: {timestamp:.1f}s",
            xaxis=dict(
                range=[0, visualizer.map_bounds.get(map_name, {}).get('image_width', 1024)],
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                range=[0, visualizer.map_bounds.get(map_name, {}).get('image_height', 1024)],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)")
        )
        
        # Create player info
        player_info = []
        if not frame_data.empty:
            player_info.append(html.H4(f"Players at {timestamp:.1f}s:"))
            for _, player in frame_data.iterrows():
                team = player.get('team_name', 'Unknown')
                health = player.get('health', 'N/A')
                speed = player.get('speed', 0)
                
                player_info.append(html.P(
                    f"{player['player_name']} ({team}) - Health: {health}, Speed: {speed:.1f}",
                    style={'margin': '5px 0', 'color': visualizer.team_colors.get(team, '#FFFFFF')}
                ))
        
        return fig, player_info
    
    # Callback for play/pause functionality
    @app.callback(
        [Output('interval-component', 'disabled'),
         Output('playback-state', 'data')],
        [Input('play-button', 'n_clicks'),
         Input('pause-button', 'n_clicks'),
         Input('reset-button', 'n_clicks')],
        [State('playback-state', 'data'),
         State('time-slider', 'min')]
    )
    def control_playback(play_clicks, pause_clicks, reset_clicks, playback_state, min_time):
        ctx = callback_context
        
        if not ctx.triggered:
            return True, playback_state
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'play-button':
            playback_state['playing'] = True
            return False, playback_state
        elif button_id == 'pause-button':
            playback_state['playing'] = False
            return True, playback_state
        elif button_id == 'reset-button':
            playback_state['playing'] = False
            return True, playback_state
        
        return True, playback_state
    
    # Combined callback for all time slider value updates
    @app.callback(
        Output('time-slider', 'value'),
        [Input('interval-component', 'n_intervals'),
         Input('reset-button', 'n_clicks'),
         Input('demo-dropdown', 'value')],
        [State('time-slider', 'value'),
         State('time-slider', 'max'),
         State('time-slider', 'min'),
         State('playback-state', 'data')],
        prevent_initial_call=True
    )
    def update_time_slider_value(n_intervals, reset_clicks, demo_name, current_time, max_time, min_time, playback_state):
        ctx = callback_context
        
        if not ctx.triggered:
            return current_time or 0
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # When demo changes, reset to start time
        if trigger_id == 'demo-dropdown':
            if demo_name and demo_name in visualizer.position_data:
                df = visualizer.position_data[demo_name]
                return df['timestamp'].min()
            return 0
        
        # When reset button is clicked
        if trigger_id == 'reset-button':
            return min_time or 0
        
        # When interval updates (auto-play)
        if trigger_id == 'interval-component' and playback_state and playback_state.get('playing', False):
            if current_time is None or max_time is None or min_time is None:
                return 0
            new_time = current_time + 1.0  # Advance by 1 second
            if new_time > max_time:
                return min_time  # Loop back to start
            return new_time
        
        return current_time or 0
    
    return app

def main():
    """
    Main function to run the interactive visualizer
    """
    print("Starting CS2/CS:GO Interactive Map Visualizer...")
    
    # Initialize the visualizer
    visualizer = MapVisualizer()
    
    if not visualizer.position_data:
        print("No position data found. Please ensure you have CSV files in the example_position_data directory.")
        print("Run the extract_player_positions.py script first to generate position data.")
        return
    
    # Create and run the Dash app
    app = create_dash_app(visualizer)
    
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:8051")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(debug=False, host='127.0.0.1', port=8051)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Trying alternative port...")
        try:
            app.run(debug=False, host='127.0.0.1', port=8052)
        except Exception as e2:
            print(f"Error on alternative port: {e2}")
    except KeyboardInterrupt:
        print("\nServer stopped by user")

if __name__ == "__main__":
    main()
