# Interactive Map Visualizer for CS2/CS:GO Player Positions

## Overview

This interactive web-based visualizer allows you to analyze player movements on CS2/CS:GO maps with a time slider. It provides real-time visualization of player positions, movement trails, and tactical analysis capabilities.

## Features

### 🎮 Core Functionality
- **Interactive Time Slider**: Scrub through match timeline to see player positions at any moment
- **Real-time Playback**: Play/pause/reset controls for automated timeline progression
- **Multi-Demo Support**: Switch between different demo files
- **Team Visualization**: Color-coded players by team (T/CT)

### 📍 Position Tracking
- **Accurate Positioning**: Player positions mapped to actual map coordinates
- **Movement Trails**: Optional trails showing recent player movements
- **Speed Visualization**: Real-time speed information in tooltips
- **Health Status**: Player health displayed in hover information

### 🗺️ Map Support
- **Multiple Maps**: Support for Mirage, Dust2, Inferno, Train, Nuke
- **Custom Map Images**: Load actual map backgrounds or use generated layouts
- **Coordinate Mapping**: Automatic transformation from game coordinates to map pixels

### 📊 Analysis Tools
- **Player Information Panel**: Real-time stats for all players
- **Team Coordination**: Visual analysis of team formations
- **Movement Patterns**: Trail visualization for tactical analysis
- **Export Ready**: Data structure suitable for further analysis

## Installation

### Prerequisites
```bash
# Ensure you have Python 3.8+ and the virtual environment activated
source /path/to/your/venv/bin/activate

# Install required packages
pip install dash plotly pandas numpy pillow
```

### Setup
1. **Extract Position Data** (if not done already):
   ```bash
   python3 extract_player_positions.py
   ```

2. **Download/Create Map Images**:
   ```bash
   python3 download_map_images.py
   ```

3. **Run the Visualizer**:
   ```bash
   python3 interactive_map_visualizer.py
   ```

4. **Open Browser**: Navigate to `http://localhost:8050`

## Usage Guide

### Basic Controls

#### Demo Selection
- Use the dropdown to select which demo file to analyze
- The time slider automatically adjusts to the demo's duration

#### Time Navigation
- **Slider**: Drag to jump to any point in the timeline
- **Play Button**: Start automatic timeline progression
- **Pause Button**: Stop automatic progression
- **Reset Button**: Jump back to the beginning

#### Visualization Options
- **Show Trails**: Toggle player movement trails
- **Team Colors**: Automatic color coding (Gold for T, Blue for CT)

### Understanding the Visualization

#### Player Markers
- **Circles**: Current player positions
- **Colors**: Team affiliation (Gold = Terrorist, Blue = Counter-Terrorist)
- **Labels**: Player names above markers
- **Size**: Consistent marker size for visibility

#### Movement Trails
- **Dotted Lines**: Recent movement paths (when enabled)
- **Opacity**: Trails fade to show temporal progression
- **Color**: Matches team colors

#### Map Background
- **Actual Maps**: High-quality map images when available
- **Generated Maps**: Simplified layouts with labeled areas
- **Scale**: Coordinates properly mapped to image dimensions

### Hover Information
When hovering over player markers, you'll see:
- Player name
- Team affiliation
- Current position coordinates
- Health status
- Movement speed

### Player Information Panel
The bottom panel shows:
- Current timestamp
- List of all active players
- Team, health, and speed for each player
- Color-coded by team

## Map Coordinate System

### Coordinate Mapping
The visualizer automatically converts game coordinates to image pixels:

```python
# Example for Mirage
game_bounds = {
    'x_min': -3217, 'x_max': 1912,    # Game X range
    'y_min': -3401, 'y_max': 1682,    # Game Y range
}

# Converted to image coordinates (0-1024 pixels)
image_x = (game_x - x_min) / (x_max - x_min) * image_width
image_y = image_height - ((game_y - y_min) / (y_max - y_min) * image_height)
```

### Supported Maps
- **de_mirage**: Complete layout with labeled areas
- **de_dust2**: Basic coordinate mapping
- **de_inferno**: Basic coordinate mapping  
- **de_train**: Basic coordinate mapping
- **de_nuke**: Basic coordinate mapping

## Customization

### Adding New Maps

1. **Define Coordinates** in `interactive_map_visualizer.py`:
   ```python
   self.map_bounds['de_newmap'] = {
       'x_min': -2000, 'x_max': 2000,
       'y_min': -2000, 'y_max': 2000,
       'image_width': 1024, 'image_height': 1024
   }
   ```

2. **Add Map Image** to `map_images/de_newmap.png`

3. **Update Map Creator** in `download_map_images.py` (optional)

### Modifying Visualization

#### Colors
```python
self.team_colors = {
    'T': '#FF4444',        # Red for Terrorists
    'CT': '#4444FF',       # Blue for Counter-Terrorists
}
```

#### Update Frequency
```python
dcc.Interval(
    id='interval-component',
    interval=1000,  # Update every 1000ms (1 second)
    n_intervals=0,
    disabled=True
)
```

#### Trail Length
```python
trail_data = visualizer.create_player_trails(
    demo_name, timestamp, 
    trail_length=15  # 15 seconds of trail
)
```

## Performance Optimization

### Large Datasets
- The visualizer handles large datasets efficiently
- Position data is sampled at 1-second intervals by default
- Memory usage scales with number of players and timeline length

### Smooth Playback
- Automatic frame interpolation for missing data points
- Optimized coordinate transformation
- Efficient data filtering by timestamp

### Browser Performance
- Uses Plotly's optimized rendering
- Minimal DOM updates during playback
- Responsive design for different screen sizes

## Troubleshooting

### Common Issues

1. **"No demo selected" Error**
   - Ensure position data CSV files exist in `example_position_data/`
   - Run `extract_player_positions.py` first

2. **Map Not Displaying**
   - Check if map image exists in `map_images/`
   - Run `download_map_images.py` to create placeholders
   - Verify map name matches in position data

3. **Players Not Visible**
   - Check coordinate bounds for the map
   - Verify position data contains valid X, Y coordinates
   - Adjust map bounds if necessary

4. **Slow Performance**
   - Reduce trail length
   - Increase update interval
   - Filter position data to smaller time ranges

### Debug Mode
Run with debug enabled:
```bash
python3 interactive_map_visualizer.py
```
Check console output for detailed error messages.

## Data Format

### Input Requirements
The visualizer expects CSV files with these columns:
- `timestamp`: Time in seconds
- `player_name`: Player identifier
- `X`, `Y`: Game coordinates
- `team_name`: Team affiliation (T/CT)
- `health`: Player health (optional)
- `speed`: Movement speed (optional)
- `map_name`: Map identifier

### Example Data Structure
```csv
timestamp,player_name,X,Y,Z,team_name,health,speed,map_name
0.0,player1,-1776.0,-1800.0,-263.97,T,100,0.0,de_mirage
1.0,player1,-1776.0,-1800.0,-263.97,T,100,0.0,de_mirage
```

## Advanced Features

### Tactical Analysis
- **Formation Analysis**: Observe team positioning and coordination
- **Movement Patterns**: Identify common routes and strategies
- **Timing Analysis**: Correlate movements with round events
- **Heat Map Potential**: Foundation for density analysis

### Export Capabilities
- Position data readily available for further analysis
- Timestamp-based filtering for specific round analysis
- Player trajectory data for machine learning applications

### Integration Potential
- **Round Events**: Can be synchronized with bomb plants, kills, etc.
- **Communication Data**: Overlay with voice/text chat timing
- **Performance Metrics**: Combine with kill/death/assist data

## Future Enhancements

### Planned Features
- **Heat Map Visualization**: Player density analysis
- **Round Event Markers**: Bomb plants, defuses, kills
- **Multiple Timeline**: Compare different rounds
- **3D Visualization**: Height-based analysis
- **Statistical Overlays**: Win rate by position, etc.

### Technical Improvements
- **WebGL Rendering**: For better performance with large datasets
- **Real-time Streaming**: Live demo analysis
- **Mobile Optimization**: Touch-friendly controls
- **Export Functions**: Save visualizations as images/videos

This interactive visualizer provides a powerful foundation for tactical analysis of CS2/CS:GO gameplay, offering both real-time visualization and the flexibility for advanced analytical applications.
