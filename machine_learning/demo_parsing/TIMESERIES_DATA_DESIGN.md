# Time Series Data Design for Player Position Analysis

## Overview

This document outlines the design of time series data structures for CS2/CS:GO player position analysis and provides comprehensive strategies for data transformation and analysis.

## Demo File Format Research

### CS:GO Demo Files (.dem)
- **Header**: `HL2DEMO` - Source Engine demo format
- **Tick Rate**: Typically 64 or 128 ticks per second
- **Content**: Player positions, events, game state at each tick
- **Parser**: Best supported by `awpy` library

### CS2 Demo Files (.dem)
- **Header**: `PBDEMS2` or `DEMS2` - New CS2 demo format
- **Tick Rate**: 64 or 128 ticks per second (subtick system)
- **Content**: Enhanced player data with subtick precision
- **Parser**: Best supported by `demoparser2` library

## Time Series Data Structure

### Core Schema

```python
{
    # Temporal Information
    'timestamp': float,           # Time in seconds from demo start
    'tick': int,                 # Game tick number
    'game_time': timedelta,      # Pandas timedelta object
    
    # Player Identification
    'player_name': str,          # Player's name/identifier
    'player_id': int,            # Unique player ID
    'team_name': str,            # Team name (CT/T)
    'team_num': int,             # Team number (2/3)
    
    # Spatial Coordinates (3D Position)
    'X': float,                  # X coordinate (left-right)
    'Y': float,                  # Y coordinate (forward-backward)
    'Z': float,                  # Z coordinate (up-down)
    
    # Movement Vectors
    'velocity_X': float,         # X-axis velocity
    'velocity_Y': float,         # Y-axis velocity
    'velocity_Z': float,         # Z-axis velocity
    
    # View Direction
    'view_angle_X': float,       # Pitch (up-down look)
    'view_angle_Y': float,       # Yaw (left-right look)
    
    # Player State
    'health': int,               # Health points (0-100)
    'armor_value': int,          # Armor points (0-100)
    'is_alive': bool,            # Alive status
    'is_crouching': bool,        # Crouching state
    'is_walking': bool,          # Walking (quiet movement)
    
    # Equipment
    'active_weapon_name': str,   # Current weapon
    
    # Metadata
    'demo_file': str,            # Source demo filename
    'map_name': str,             # Map name (e.g., de_dust2)
    'round_number': int,         # Current round number
}
```

### Derived Features (Calculated)

```python
{
    # Movement Analysis
    'speed': float,              # 3D speed (units/second)
    'speed_2d': float,           # 2D speed (ignoring Z)
    'distance_moved': float,     # Distance since last tick
    'acceleration': float,       # Speed change rate
    'direction_change': float,   # Angular change in movement
    
    # Positional Features
    'distance_to_bombsite_A': float,  # Distance to bomb site A
    'distance_to_bombsite_B': float,  # Distance to bomb site B
    'area_control': str,         # Map area/zone
    'elevation': float,          # Relative elevation
    
    # Tactical Features
    'exposure_score': float,     # How exposed player is
    'cover_availability': float, # Available cover nearby
    'teammate_proximity': float, # Distance to nearest teammate
    'enemy_proximity': float,    # Distance to nearest enemy (if visible)
    
    # Temporal Features
    'time_in_area': float,       # Time spent in current area
    'time_since_last_kill': float,
    'time_since_last_death': float,
    'round_time_remaining': float,
}
```

## Data Transformation Strategies

### 1. Temporal Transformations

#### A. Resampling Strategies
```python
# High-frequency analysis (every 100ms)
df_100ms = df.resample('100ms').mean()

# Standard analysis (every 1 second)
df_1s = df.resample('1S').mean()

# Strategic analysis (every 5 seconds)
df_5s = df.resample('5S').mean()

# Round-based analysis
df_rounds = df.groupby('round_number').agg({
    'X': ['mean', 'std', 'min', 'max'],
    'Y': ['mean', 'std', 'min', 'max'],
    'speed': ['mean', 'max'],
    'health': ['mean', 'min']
})
```

#### B. Window Functions
```python
# Moving averages for smoothing
df['speed_ma_3s'] = df.groupby('player_name')['speed'].rolling('3S').mean()

# Exponential smoothing for trend analysis
df['position_trend_X'] = df.groupby('player_name')['X'].ewm(span=10).mean()

# Change detection
df['position_change'] = df.groupby('player_name')[['X', 'Y', 'Z']].diff()
```

### 2. Spatial Transformations

#### A. Coordinate System Normalization
```python
# Map-specific normalization (0-1 scale)
def normalize_coordinates(df, map_bounds):
    df['X_norm'] = (df['X'] - map_bounds['x_min']) / (map_bounds['x_max'] - map_bounds['x_min'])
    df['Y_norm'] = (df['Y'] - map_bounds['y_min']) / (map_bounds['y_max'] - map_bounds['y_min'])
    return df

# Polar coordinates (distance and angle from map center)
df['distance_from_center'] = np.sqrt(df['X_norm']**2 + df['Y_norm']**2)
df['angle_from_center'] = np.arctan2(df['Y_norm'], df['X_norm'])
```

#### B. Zone-Based Analysis
```python
# Map area classification
def classify_map_areas(df, map_name):
    if map_name == 'de_dust2':
        conditions = [
            (df['X'] < -500) & (df['Y'] > 1000),  # A Site
            (df['X'] > 1000) & (df['Y'] < -500),  # B Site
            # ... more conditions
        ]
        choices = ['A_Site', 'B_Site', 'Mid', 'T_Spawn', 'CT_Spawn']
        df['area'] = np.select(conditions, choices, default='Other')
    return df
```

### 3. Feature Engineering for Time Series Analysis

#### A. Lag Features
```python
# Previous positions for trajectory analysis
for lag in [1, 3, 5, 10]:
    df[f'X_lag_{lag}'] = df.groupby('player_name')['X'].shift(lag)
    df[f'Y_lag_{lag}'] = df.groupby('player_name')['Y'].shift(lag)
```

#### B. Statistical Features (Rolling Windows)
```python
# Rolling statistics for behavioral patterns
window_sizes = ['5S', '10S', '30S']
for window in window_sizes:
    df[f'speed_mean_{window}'] = df.groupby('player_name')['speed'].rolling(window).mean()
    df[f'speed_std_{window}'] = df.groupby('player_name')['speed'].rolling(window).std()
    df[f'direction_changes_{window}'] = df.groupby('player_name')['direction_change'].rolling(window).sum()
```

#### C. Interaction Features
```python
# Team coordination metrics
def calculate_team_features(df):
    # Team centroid
    team_centroids = df.groupby(['timestamp', 'team_name'])[['X', 'Y']].mean().reset_index()
    team_centroids.columns = ['timestamp', 'team_name', 'team_center_X', 'team_center_Y']
    
    # Merge back to get distance from team center
    df = df.merge(team_centroids, on=['timestamp', 'team_name'])
    df['distance_from_team_center'] = np.sqrt(
        (df['X'] - df['team_center_X'])**2 + 
        (df['Y'] - df['team_center_Y'])**2
    )
    
    # Team spread (how spread out the team is)
    team_spread = df.groupby(['timestamp', 'team_name'])['distance_from_team_center'].std().reset_index()
    team_spread.columns = ['timestamp', 'team_name', 'team_spread']
    df = df.merge(team_spread, on=['timestamp', 'team_name'])
    
    return df
```

### 4. Advanced Time Series Transformations

#### A. Fourier Transform for Pattern Detection
```python
from scipy.fft import fft, fftfreq

def analyze_movement_patterns(player_data):
    # FFT on X and Y coordinates to find periodic movement
    x_fft = fft(player_data['X'].values)
    y_fft = fft(player_data['Y'].values)
    
    # Dominant frequencies
    freqs = fftfreq(len(player_data), d=1.0)  # Assuming 1 second intervals
    
    return {
        'dominant_freq_x': freqs[np.argmax(np.abs(x_fft[1:len(freqs)//2])) + 1],
        'dominant_freq_y': freqs[np.argmax(np.abs(y_fft[1:len(freqs)//2])) + 1]
    }
```

#### B. Trajectory Clustering
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def cluster_movement_patterns(df):
    # Create trajectory vectors
    trajectory_features = []
    for player in df['player_name'].unique():
        player_data = df[df['player_name'] == player]
        
        # Extract movement features for clustering
        features = [
            player_data['speed'].mean(),
            player_data['speed'].std(),
            player_data['direction_change'].sum(),
            player_data['X'].std(),
            player_data['Y'].std()
        ]
        trajectory_features.append(features)
    
    # Cluster similar movement patterns
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(trajectory_features)
    
    clusterer = DBSCAN(eps=0.5, min_samples=2)
    clusters = clusterer.fit_predict(features_scaled)
    
    return dict(zip(df['player_name'].unique(), clusters))
```

#### C. State Transition Analysis
```python
def analyze_state_transitions(df):
    # Define states based on speed and area
    def get_state(row):
        if row['speed'] < 50:
            return 'stationary'
        elif row['speed'] < 200:
            return 'walking'
        else:
            return 'running'
    
    df['movement_state'] = df.apply(get_state, axis=1)
    
    # Transition matrix
    transitions = df.groupby('player_name').apply(
        lambda x: x['movement_state'].shift().fillna('start').astype(str) + '_to_' + x['movement_state'].astype(str)
    ).reset_index()
    
    return transitions['movement_state'].value_counts()
```

### 5. Multi-Player Analysis Transformations

#### A. Relative Position Analysis
```python
def calculate_relative_positions(df):
    # For each player, calculate position relative to all other players
    relative_data = []
    
    for timestamp in df['timestamp'].unique():
        frame = df[df['timestamp'] == timestamp]
        
        for i, player1 in frame.iterrows():
            for j, player2 in frame.iterrows():
                if i != j:
                    relative_data.append({
                        'timestamp': timestamp,
                        'player1': player1['player_name'],
                        'player2': player2['player_name'],
                        'relative_x': player2['X'] - player1['X'],
                        'relative_y': player2['Y'] - player1['Y'],
                        'distance': np.sqrt((player2['X'] - player1['X'])**2 + (player2['Y'] - player1['Y'])**2),
                        'same_team': player1['team_name'] == player2['team_name']
                    })
    
    return pd.DataFrame(relative_data)
```

#### B. Formation Analysis
```python
def analyze_team_formations(df):
    # Calculate team formations at each timestamp
    formations = []
    
    for timestamp in df['timestamp'].unique():
        for team in df['team_name'].unique():
            team_players = df[(df['timestamp'] == timestamp) & (df['team_name'] == team)]
            
            if len(team_players) >= 2:
                # Calculate formation metrics
                center_x = team_players['X'].mean()
                center_y = team_players['Y'].mean()
                
                # Spread metrics
                spread_x = team_players['X'].std()
                spread_y = team_players['Y'].std()
                
                # Formation compactness
                distances = []
                for i, p1 in team_players.iterrows():
                    for j, p2 in team_players.iterrows():
                        if i < j:
                            dist = np.sqrt((p1['X'] - p2['X'])**2 + (p1['Y'] - p2['Y'])**2)
                            distances.append(dist)
                
                formations.append({
                    'timestamp': timestamp,
                    'team': team,
                    'center_x': center_x,
                    'center_y': center_y,
                    'spread_x': spread_x,
                    'spread_y': spread_y,
                    'avg_inter_distance': np.mean(distances) if distances else 0,
                    'formation_compactness': 1 / (np.mean(distances) + 1) if distances else 0
                })
    
    return pd.DataFrame(formations)
```

## Analysis Applications

### 1. Player Behavior Analysis
- **Movement Patterns**: Identify aggressive vs. defensive play styles
- **Positioning**: Analyze optimal positioning for different situations
- **Reaction Times**: Measure response to game events

### 2. Team Coordination Analysis
- **Formation Dynamics**: How teams move together
- **Communication Patterns**: Inferred from coordinated movements
- **Strategic Execution**: Analyze execution of planned strategies

### 3. Performance Prediction
- **Outcome Prediction**: Predict round outcomes based on positions
- **Player Performance**: Predict individual player performance
- **Risk Assessment**: Identify high-risk positions and movements

### 4. Map Control Analysis
- **Territory Control**: Which team controls which areas over time
- **Chokepoint Analysis**: Movement through key map areas
- **Rotation Patterns**: How teams rotate between bomb sites

## Implementation Recommendations

### Data Pipeline
1. **Raw Demo Parsing** → Extract basic position data
2. **Time Series Conversion** → Resample and structure data
3. **Feature Engineering** → Calculate derived features
4. **Data Validation** → Check for anomalies and missing data
5. **Storage** → Save in efficient format (Parquet recommended)

### Performance Considerations
- Use vectorized operations (NumPy/Pandas)
- Consider chunking for large datasets
- Implement parallel processing for multiple demos
- Use appropriate data types (float32 vs float64)

### Visualization Strategies
- **Trajectory Plots**: 2D movement paths
- **Heat Maps**: Position density analysis
- **Time Series Plots**: Speed, health, etc. over time
- **Interactive Dashboards**: Real-time exploration

This comprehensive framework provides the foundation for sophisticated analysis of player position data from CS2/CS:GO demo files, enabling insights into player behavior, team dynamics, and strategic patterns.
