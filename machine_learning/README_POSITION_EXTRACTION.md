# Player Position Extraction for CS2/CS:GO Demo Files

## Overview

This project provides a comprehensive solution for extracting player positions from CS2/CS:GO demo files and converting them into time series data suitable for machine learning analysis. The system handles both CS:GO (`.dem` files with `HL2DEMO` header) and CS2 (`.dem` files with `PBDEMS2` header) formats.

## Files Created

### Core Files
- **`extract_player_positions.py`** - Main extraction script with full functionality
- **`example_usage.py`** - Example script demonstrating how to use the extractor
- **`requirements_position_extraction.txt`** - Python dependencies
- **`TIMESERIES_DATA_DESIGN.md`** - Comprehensive documentation of data structures and analysis strategies

### Existing Files (Analyzed)
- **`parse_demo.py`** - Original demo parser using awpy (limited CS2 support)
- **`parse_demo_demoparser2.py`** - Enhanced parser using demoparser2 (better CS2 support)
- **`CS2_DEMO_PARSING_ISSUE.md`** - Analysis of parsing challenges with CS2 demos

## Key Features

### 🎮 Demo File Support
- **CS:GO demos**: Full support via awpy library
- **CS2 demos**: Enhanced support via demoparser2 library
- **Auto-detection**: Automatically detects demo format and uses appropriate parser

### 📊 Data Extraction
- **3D Positions**: X, Y, Z coordinates for all players
- **Movement Data**: Velocity vectors, speed, acceleration
- **Player State**: Health, armor, team, weapon, alive status
- **View Direction**: Pitch and yaw angles
- **Temporal Data**: Tick numbers, timestamps, game time

### ⏱️ Time Series Conversion
- **Resampling**: Convert tick-based data to regular time intervals
- **Multiple Frequencies**: Support for various sampling rates (100ms, 1s, 5s, etc.)
- **Feature Engineering**: Automatic calculation of movement features
- **Data Quality**: Handles missing data and provides validation

### 📈 Advanced Features
- **Movement Analysis**: Speed, acceleration, direction changes
- **Team Coordination**: Formation analysis, team centroids
- **Spatial Analysis**: Map area classification, relative positions
- **Pattern Detection**: Fourier analysis, trajectory clustering
- **Visualization**: Movement paths, speed plots, heat maps

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_position_extraction.txt
   ```

2. **Key Libraries**:
   - `demoparser2` - Primary parser for CS2 demos
   - `awpy` - Alternative parser for CS:GO demos
   - `pandas`, `numpy` - Data manipulation
   - `matplotlib`, `seaborn` - Visualization (optional)

## Usage

### Basic Usage

```python
from extract_player_positions import PlayerPositionExtractor

# Initialize extractor
extractor = PlayerPositionExtractor(output_dir="position_data")

# Process a single demo
result = extractor.process_demo_file(
    demo_path="path/to/demo.dem",
    resample_interval="1S",
    calculate_features=True,
    save_formats=["csv", "json"]
)
```

### Batch Processing

```python
# Process all demos in a directory
results = extractor.batch_process(
    demos_dir="demos_extracted/",
    resample_interval="500ms",
    calculate_features=True,
    save_formats=["csv"],
    create_visualizations=True
)
```

### Command Line Usage

```bash
# Process single file
python extract_player_positions.py path/to/demo.dem --output-dir results

# Process directory
python extract_player_positions.py demos_extracted/ --resample 500ms --formats csv json

# Full options
python extract_player_positions.py demos/ --output-dir results --resample 1S --no-features --no-plots
```

## Data Structure

### Core Time Series Schema
```python
{
    'timestamp': float,           # Time in seconds
    'player_name': str,          # Player identifier
    'X': float, 'Y': float, 'Z': float,  # 3D position
    'velocity_X': float, 'velocity_Y': float, 'velocity_Z': float,
    'health': int,               # Health points
    'team_name': str,            # Team (CT/T)
    'active_weapon_name': str,   # Current weapon
    'demo_file': str,            # Source demo
    'map_name': str              # Map name
}
```

### Derived Features
```python
{
    'speed': float,              # Movement speed
    'acceleration': float,       # Speed change rate
    'direction_change': float,   # Angular movement change
    'distance_moved': float,     # Distance since last tick
    # ... many more features available
}
```

## Research Findings

### Demo File Formats

1. **CS:GO Demos** (`HL2DEMO` header)
   - Well-supported by awpy library
   - Standard Source Engine format
   - Reliable parsing for rounds, kills, positions

2. **CS2 Demos** (`PBDEMS2` header)
   - New format with enhanced data
   - Better supported by demoparser2
   - Includes subtick precision data

### Time Series Transformation Strategies

1. **Temporal Transformations**
   - Resampling to regular intervals
   - Moving averages and exponential smoothing
   - Window-based feature calculation

2. **Spatial Transformations**
   - Coordinate normalization
   - Zone-based area classification
   - Polar coordinate conversion

3. **Feature Engineering**
   - Lag features for trajectory analysis
   - Rolling statistics for behavioral patterns
   - Team interaction metrics

4. **Advanced Analysis**
   - Fourier transforms for pattern detection
   - Trajectory clustering for play style analysis
   - State transition modeling

## Applications

### 🎯 Player Analysis
- Movement pattern classification
- Play style identification
- Performance prediction
- Skill assessment

### 🤝 Team Analysis
- Coordination measurement
- Formation analysis
- Strategic pattern recognition
- Communication inference

### 🗺️ Map Analysis
- Territory control over time
- Chokepoint utilization
- Rotation pattern analysis
- Optimal positioning identification

### 🔮 Predictive Modeling
- Round outcome prediction
- Player performance forecasting
- Risk assessment
- Strategic recommendation

## Example Analysis Workflows

### 1. Basic Position Analysis
```python
# Load extracted data
df = pd.read_csv("position_data/demo_positions_timeseries.csv")

# Analyze player movement patterns
player_stats = df.groupby('player_name').agg({
    'speed': ['mean', 'max', 'std'],
    'X': ['mean', 'std'],
    'Y': ['mean', 'std']
})
```

### 2. Team Coordination Analysis
```python
# Calculate team centroids
team_centers = df.groupby(['timestamp', 'team_name'])[['X', 'Y']].mean()

# Measure team spread
team_spread = df.groupby(['timestamp', 'team_name']).apply(
    lambda x: x[['X', 'Y']].std().mean()
)
```

### 3. Movement Pattern Clustering
```python
from sklearn.cluster import DBSCAN

# Extract movement features per player
features = df.groupby('player_name').agg({
    'speed': ['mean', 'std'],
    'direction_change': 'sum',
    'X': 'std', 'Y': 'std'
})

# Cluster similar movement patterns
clusterer = DBSCAN(eps=0.5, min_samples=2)
clusters = clusterer.fit_predict(features)
```

## Performance Considerations

- **Memory Usage**: Large demos can generate millions of position records
- **Processing Time**: Batch processing recommended for multiple demos
- **Storage**: Parquet format recommended for large datasets
- **Visualization**: Limit player count for readable plots

## Troubleshooting

### Common Issues

1. **"demoparser2 not available"**
   - Install: `pip install demoparser2`

2. **"awpy not available"**
   - Install: `pip install awpy`

3. **Empty CSV files**
   - Check demo format compatibility
   - Verify demo file integrity
   - Check parsing library versions

4. **Memory errors**
   - Process demos individually
   - Reduce resampling frequency
   - Use chunking for large datasets

### Demo Compatibility

- **CS2 demos**: Use demoparser2 (recommended)
- **CS:GO demos**: Use awpy or demoparser2
- **Mixed formats**: Auto-detection handles both

## Future Enhancements

- Real-time demo parsing
- Advanced ML model integration
- Interactive visualization dashboard
- Multi-demo comparative analysis
- Statistical significance testing

## Contributing

The codebase is designed to be extensible. Key areas for enhancement:

1. Additional parsers for other game formats
2. More sophisticated feature engineering
3. Advanced visualization capabilities
4. Performance optimizations
5. Integration with machine learning pipelines

This system provides a solid foundation for analyzing player behavior and team dynamics in tactical FPS games through time series analysis of positional data.
