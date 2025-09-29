#!/usr/bin/env python3
"""
Player Position Extractor for CS2/CS:GO Demo Files
==================================================

This script extracts player positions and movements from demo files and converts them
into time series format suitable for analysis. It supports both CS:GO and CS2 demo files.

Features:
- Extracts player positions (X, Y, Z coordinates) over time
- Includes player metadata (health, armor, team, weapon)
- Converts data to time series format with proper timestamps
- Supports multiple output formats (CSV, Parquet, JSON)
- Handles both individual demo files and batch processing
- Provides data quality checks and visualization options

Dependencies:
    pip install demoparser2 pandas numpy matplotlib seaborn

Author: Generated for CS 5612 Project
Date: 2024
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import json

# Visualization imports (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting features disabled.")

# Demo parsing imports
try:
    from demoparser2 import DemoParser
    DEMOPARSER2_AVAILABLE = True
except ImportError:
    DEMOPARSER2_AVAILABLE = False
    warnings.warn("demoparser2 not available. Install with: pip install demoparser2")

try:
    from awpy import Demo as AwpyDemo
    AWPY_AVAILABLE = True
except ImportError:
    AWPY_AVAILABLE = False
    warnings.warn("awpy not available. Install with: pip install awpy")


class PlayerPositionExtractor:
    """
    Extracts and processes player position data from CS2/CS:GO demo files.
    """
    
    def __init__(self, output_dir: str = "position_data", tick_rate: int = 128):
        """
        Initialize the position extractor.
        
        Args:
            output_dir: Directory to save extracted data
            tick_rate: Game tick rate (default 128 for competitive matches)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.tick_rate = tick_rate
        self.demo_metadata = {}
        
    def detect_demo_type(self, demo_path: Path) -> str:
        """
        Detect if demo is CS:GO or CS2 format.
        
        Args:
            demo_path: Path to demo file
            
        Returns:
            Demo type: 'cs2', 'csgo', or 'unknown'
        """
        try:
            with open(demo_path, "rb") as f:
                header = f.read(16)
            
            if header.startswith(b"PBDEMS2") or header.startswith(b"DEMS2"):
                return "cs2"
            elif header.startswith(b"HL2DEMO"):
                return "csgo"
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def extract_positions_demoparser2(self, demo_path: Path) -> pd.DataFrame:
        """
        Extract player positions using demoparser2 (preferred for CS2).
        
        Args:
            demo_path: Path to demo file
            
        Returns:
            DataFrame with player position data
        """
        if not DEMOPARSER2_AVAILABLE:
            raise ImportError("demoparser2 is required but not installed")
        
        print(f"Parsing {demo_path.name} with demoparser2...")
        
        try:
            parser = DemoParser(str(demo_path))
            
            # Get header information
            header = parser.parse_header()
            self.demo_metadata[demo_path.stem] = {
                'map_name': header.get('map_name', 'unknown'),
                'demo_type': self.detect_demo_type(demo_path),
                'tick_rate': header.get('tick_rate', self.tick_rate)
            }
            
            # Parse player position data
            # We want to get player positions at regular intervals
            ticks_df = parser.parse_ticks([
                "X", "Y", "Z",  # Position coordinates
                "health", "armor_value",  # Player stats
                "team_name", "team_num",  # Team information
                "active_weapon_name",  # Current weapon
                "velocity_X", "velocity_Y", "velocity_Z",  # Movement velocity
                "view_angle_X", "view_angle_Y",  # View direction
                "is_alive", "is_crouching", "is_walking"  # Player state
            ])
            
            if ticks_df.empty:
                print(f"Warning: No tick data found in {demo_path.name}")
                return pd.DataFrame()
            
            # Add metadata columns
            ticks_df['demo_file'] = demo_path.stem
            ticks_df['map_name'] = self.demo_metadata[demo_path.stem]['map_name']
            
            # Convert tick to timestamp
            actual_tick_rate = self.demo_metadata[demo_path.stem]['tick_rate']
            ticks_df['timestamp'] = ticks_df['tick'] / actual_tick_rate
            ticks_df['game_time'] = pd.to_timedelta(ticks_df['timestamp'], unit='s')
            
            print(f"Extracted {len(ticks_df)} position records from {demo_path.name}")
            return ticks_df
            
        except Exception as e:
            print(f"Error parsing {demo_path.name} with demoparser2: {e}")
            return pd.DataFrame()
    
    def extract_positions_awpy(self, demo_path: Path) -> pd.DataFrame:
        """
        Extract player positions using awpy (better for CS:GO).
        
        Args:
            demo_path: Path to demo file
            
        Returns:
            DataFrame with player position data
        """
        if not AWPY_AVAILABLE:
            raise ImportError("awpy is required but not installed")
        
        print(f"Parsing {demo_path.name} with awpy...")
        
        try:
            demo = AwpyDemo(str(demo_path))
            match = demo.parse()
            
            if match is None:
                print(f"Warning: awpy returned None for {demo_path.name} (likely CS2 demo)")
                return pd.DataFrame()
            
            # Extract tick data
            ticks_df = demo.ticks
            
            if ticks_df is None or ticks_df.empty:
                print(f"Warning: No tick data found in {demo_path.name}")
                return pd.DataFrame()
            
            # Add metadata
            ticks_df['demo_file'] = demo_path.stem
            ticks_df['map_name'] = getattr(match, 'map_name', 'unknown')
            
            # Convert tick to timestamp
            ticks_df['timestamp'] = ticks_df['tick'] / self.tick_rate
            ticks_df['game_time'] = pd.to_timedelta(ticks_df['timestamp'], unit='s')
            
            print(f"Extracted {len(ticks_df)} position records from {demo_path.name}")
            return ticks_df
            
        except Exception as e:
            print(f"Error parsing {demo_path.name} with awpy: {e}")
            return pd.DataFrame()
    
    def extract_positions(self, demo_path: Path, parser_preference: str = "auto") -> pd.DataFrame:
        """
        Extract player positions using the best available parser.
        
        Args:
            demo_path: Path to demo file
            parser_preference: "auto", "demoparser2", or "awpy"
            
        Returns:
            DataFrame with player position data
        """
        demo_type = self.detect_demo_type(demo_path)
        
        if parser_preference == "auto":
            # Use demoparser2 for CS2, awpy for CS:GO
            if demo_type == "cs2" and DEMOPARSER2_AVAILABLE:
                return self.extract_positions_demoparser2(demo_path)
            elif demo_type == "csgo" and AWPY_AVAILABLE:
                return self.extract_positions_awpy(demo_path)
            elif DEMOPARSER2_AVAILABLE:
                return self.extract_positions_demoparser2(demo_path)
            elif AWPY_AVAILABLE:
                return self.extract_positions_awpy(demo_path)
            else:
                raise ImportError("No demo parsing library available. Install demoparser2 or awpy.")
        
        elif parser_preference == "demoparser2":
            return self.extract_positions_demoparser2(demo_path)
        elif parser_preference == "awpy":
            return self.extract_positions_awpy(demo_path)
        else:
            raise ValueError(f"Unknown parser preference: {parser_preference}")
    
    def create_time_series(self, positions_df: pd.DataFrame, 
                          resample_interval: str = "1S") -> pd.DataFrame:
        """
        Convert position data to proper time series format.
        
        Args:
            positions_df: Raw position data
            resample_interval: Resampling interval (e.g., "1S" for 1 second, "100ms" for 100ms)
            
        Returns:
            Time series DataFrame with resampled data
        """
        if positions_df.empty:
            return pd.DataFrame()
        
        print(f"Converting to time series format (resampling: {resample_interval})...")
        
        # Create a copy and convert timestamp to datetime
        ts_df = positions_df.copy()
        
        # Convert timestamp to datetime for proper resampling
        # Use a reference date and add seconds as timedelta
        reference_date = pd.Timestamp('2000-01-01')
        ts_df['datetime'] = reference_date + pd.to_timedelta(ts_df['timestamp'], unit='s')
        ts_df = ts_df.set_index('datetime')
        
        # Group by player and resample
        time_series_data = []
        
        # Check for the correct player name column
        player_col = None
        for col in ['name', 'player_name', 'steam_id']:
            if col in ts_df.columns:
                player_col = col
                break
        
        if player_col is None:
            print("Warning: No player identifier column found")
            return pd.DataFrame()
        
        for player_name in ts_df[player_col].unique():
            if pd.isna(player_name):
                continue
                
            player_data = ts_df[ts_df[player_col] == player_name].copy()
            
            # Resample numeric columns
            numeric_cols = ['X', 'Y', 'Z', 'health', 'armor_value', 
                           'velocity_X', 'velocity_Y', 'velocity_Z',
                           'view_angle_X', 'view_angle_Y', 'timestamp']
            
            # Only include columns that exist
            available_numeric_cols = [col for col in numeric_cols if col in player_data.columns]
            
            if available_numeric_cols:
                resampled = player_data[available_numeric_cols].resample(resample_interval).mean()
                
                # Forward fill missing values
                resampled = resampled.ffill().bfill()
                
                # Add player metadata
                resampled['player_name'] = player_name
                resampled['demo_file'] = player_data['demo_file'].iloc[0] if 'demo_file' in player_data.columns else 'unknown'
                resampled['map_name'] = player_data['map_name'].iloc[0] if 'map_name' in player_data.columns else 'unknown'
                
                # Add team info if available
                if 'team_name' in player_data.columns:
                    resampled['team_name'] = player_data['team_name'].mode().iloc[0] if not player_data['team_name'].mode().empty else 'unknown'
                elif 'team_num' in player_data.columns:
                    resampled['team_num'] = player_data['team_num'].mode().iloc[0] if not player_data['team_num'].mode().empty else 0
                
                time_series_data.append(resampled)
        
        if not time_series_data:
            print("Warning: No valid player data found for time series conversion")
            return pd.DataFrame()
        
        # Combine all players
        final_ts = pd.concat(time_series_data, ignore_index=False)
        final_ts = final_ts.reset_index()
        
        # Convert datetime back to timestamp for consistency
        final_ts['timestamp'] = (final_ts['datetime'] - reference_date).dt.total_seconds()
        final_ts = final_ts.drop('datetime', axis=1)
        
        print(f"Created time series with {len(final_ts)} records")
        return final_ts
    
    def calculate_movement_features(self, ts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate movement-based features from position time series.
        
        Args:
            ts_df: Time series DataFrame
            
        Returns:
            DataFrame with additional movement features
        """
        if ts_df.empty:
            return ts_df
        
        print("Calculating movement features...")
        
        enhanced_df = ts_df.copy()
        
        # Group by player to calculate features
        for player_name in enhanced_df['player_name'].unique():
            mask = enhanced_df['player_name'] == player_name
            player_data = enhanced_df[mask].copy()
            
            if len(player_data) < 2:
                continue
            
            # Calculate speed (distance traveled per time unit)
            player_data['distance_moved'] = np.sqrt(
                (player_data['X'].diff()) ** 2 + 
                (player_data['Y'].diff()) ** 2 + 
                (player_data['Z'].diff()) ** 2
            )
            
            player_data['speed'] = player_data['distance_moved'] / player_data['timestamp'].diff()
            
            # Calculate acceleration
            player_data['acceleration'] = player_data['speed'].diff() / player_data['timestamp'].diff()
            
            # Calculate direction changes (angle between consecutive movement vectors)
            dx = player_data['X'].diff()
            dy = player_data['Y'].diff()
            prev_dx = dx.shift(1)
            prev_dy = dy.shift(1)
            
            # Angle between vectors
            dot_product = dx * prev_dx + dy * prev_dy
            mag1 = np.sqrt(dx**2 + dy**2)
            mag2 = np.sqrt(prev_dx**2 + prev_dy**2)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
                player_data['direction_change'] = np.arccos(cos_angle)
            
            # Update the main DataFrame
            enhanced_df.loc[mask, ['distance_moved', 'speed', 'acceleration', 'direction_change']] = \
                player_data[['distance_moved', 'speed', 'acceleration', 'direction_change']].values
        
        # Fill NaN values with 0 for the first records
        feature_cols = ['distance_moved', 'speed', 'acceleration', 'direction_change']
        enhanced_df[feature_cols] = enhanced_df[feature_cols].fillna(0)
        
        print(f"Added movement features: {feature_cols}")
        return enhanced_df
    
    def save_data(self, df: pd.DataFrame, filename: str, formats: List[str] = ["csv"]) -> Dict[str, Path]:
        """
        Save data in multiple formats.
        
        Args:
            df: DataFrame to save
            filename: Base filename (without extension)
            formats: List of formats to save ("csv", "parquet", "json")
            
        Returns:
            Dictionary mapping format to saved file path
        """
        saved_files = {}
        
        for fmt in formats:
            if fmt == "csv":
                filepath = self.output_dir / f"{filename}.csv"
                df.to_csv(filepath, index=False)
                saved_files["csv"] = filepath
                
            elif fmt == "parquet":
                try:
                    filepath = self.output_dir / f"{filename}.parquet"
                    df.to_parquet(filepath, index=False)
                    saved_files["parquet"] = filepath
                except ImportError:
                    print("Warning: pyarrow not available for Parquet format")
                    
            elif fmt == "json":
                filepath = self.output_dir / f"{filename}.json"
                df.to_json(filepath, orient="records", date_format="iso")
                saved_files["json"] = filepath
        
        return saved_files
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a summary report of the extracted data.
        
        Args:
            df: Position data DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {"status": "no_data", "message": "No data available for analysis"}
        
        report = {
            "data_summary": {
                "total_records": len(df),
                "unique_players": df['player_name'].nunique() if 'player_name' in df.columns else 0,
                "unique_demos": df['demo_file'].nunique() if 'demo_file' in df.columns else 0,
                "time_range_seconds": df['timestamp'].max() - df['timestamp'].min() if 'timestamp' in df.columns else 0,
                "maps": df['map_name'].unique().tolist() if 'map_name' in df.columns else []
            },
            "position_stats": {},
            "movement_stats": {}
        }
        
        # Position statistics
        if all(col in df.columns for col in ['X', 'Y', 'Z']):
            report["position_stats"] = {
                "x_range": [df['X'].min(), df['X'].max()],
                "y_range": [df['Y'].min(), df['Y'].max()],
                "z_range": [df['Z'].min(), df['Z'].max()],
                "position_variance": {
                    "x_var": df['X'].var(),
                    "y_var": df['Y'].var(),
                    "z_var": df['Z'].var()
                }
            }
        
        # Movement statistics
        if 'speed' in df.columns:
            report["movement_stats"] = {
                "avg_speed": df['speed'].mean(),
                "max_speed": df['speed'].max(),
                "speed_std": df['speed'].std()
            }
        
        return report
    
    def visualize_player_paths(self, df: pd.DataFrame, demo_name: str = None, 
                             max_players: int = 10) -> Optional[plt.Figure]:
        """
        Create visualization of player movement paths.
        
        Args:
            df: Position data DataFrame
            demo_name: Name of demo for title
            max_players: Maximum number of players to plot
            
        Returns:
            Matplotlib figure or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Install matplotlib and seaborn.")
            return None
        
        if df.empty or not all(col in df.columns for col in ['X', 'Y', 'player_name']):
            print("Insufficient data for plotting")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Player paths (top-down view)
        players = df['player_name'].unique()[:max_players]
        colors = plt.cm.tab10(np.linspace(0, 1, len(players)))
        
        for i, player in enumerate(players):
            if pd.isna(player):
                continue
            player_data = df[df['player_name'] == player]
            ax1.plot(player_data['X'], player_data['Y'], 
                    color=colors[i], alpha=0.7, linewidth=1, label=player)
            
            # Mark start and end points
            if len(player_data) > 0:
                ax1.scatter(player_data['X'].iloc[0], player_data['Y'].iloc[0], 
                           color=colors[i], marker='o', s=50, edgecolor='black')
                ax1.scatter(player_data['X'].iloc[-1], player_data['Y'].iloc[-1], 
                           color=colors[i], marker='s', s=50, edgecolor='black')
        
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title(f'Player Movement Paths - {demo_name or "Demo"}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speed over time
        if 'speed' in df.columns and 'timestamp' in df.columns:
            for i, player in enumerate(players[:5]):  # Limit to 5 players for readability
                if pd.isna(player):
                    continue
                player_data = df[df['player_name'] == player]
                ax2.plot(player_data['timestamp'], player_data['speed'], 
                        color=colors[i], alpha=0.7, label=player)
            
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Speed (units/second)')
            ax2.set_title('Player Speed Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Speed data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Speed Analysis (No Data)')
        
        plt.tight_layout()
        return fig
    
    def process_demo_file(self, demo_path: Path, 
                         resample_interval: str = "1S",
                         calculate_features: bool = True,
                         save_formats: List[str] = ["csv"],
                         create_visualizations: bool = True) -> Dict:
        """
        Process a single demo file and extract position data.
        
        Args:
            demo_path: Path to demo file
            resample_interval: Time series resampling interval
            calculate_features: Whether to calculate movement features
            save_formats: Formats to save data in
            create_visualizations: Whether to create plots
            
        Returns:
            Dictionary with processing results
        """
        print(f"\n=== Processing {demo_path.name} ===")
        
        # Extract raw positions
        positions_df = self.extract_positions(demo_path)
        
        if positions_df.empty:
            return {
                "status": "failed",
                "message": "No position data extracted",
                "demo_file": demo_path.name
            }
        
        # Convert to time series
        ts_df = self.create_time_series(positions_df, resample_interval)
        
        if ts_df.empty:
            return {
                "status": "failed", 
                "message": "Time series conversion failed",
                "demo_file": demo_path.name
            }
        
        # Calculate movement features
        if calculate_features:
            ts_df = self.calculate_movement_features(ts_df)
        
        # Save data
        base_filename = f"{demo_path.stem}_positions_timeseries"
        saved_files = self.save_data(ts_df, base_filename, save_formats)
        
        # Generate report
        report = self.generate_summary_report(ts_df)
        
        # Save report
        report_path = self.output_dir / f"{demo_path.stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create visualizations
        if create_visualizations and PLOTTING_AVAILABLE:
            fig = self.visualize_player_paths(ts_df, demo_path.stem)
            if fig:
                plot_path = self.output_dir / f"{demo_path.stem}_movement_plot.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files["plot"] = plot_path
        
        return {
            "status": "success",
            "demo_file": demo_path.name,
            "records_extracted": len(ts_df),
            "saved_files": saved_files,
            "report": report
        }
    
    def batch_process(self, demos_dir: Path, 
                     resample_interval: str = "1S",
                     calculate_features: bool = True,
                     save_formats: List[str] = ["csv"],
                     create_visualizations: bool = True) -> List[Dict]:
        """
        Process multiple demo files in a directory.
        
        Args:
            demos_dir: Directory containing demo files
            resample_interval: Time series resampling interval
            calculate_features: Whether to calculate movement features
            save_formats: Formats to save data in
            create_visualizations: Whether to create plots
            
        Returns:
            List of processing results for each demo
        """
        print(f"\n=== Batch Processing Demos from {demos_dir} ===")
        
        # Find all demo files
        demo_files = []
        for root, dirs, files in os.walk(demos_dir):
            for file in files:
                if file.lower().endswith('.dem'):
                    demo_files.append(Path(root) / file)
        
        if not demo_files:
            print("No demo files found!")
            return []
        
        print(f"Found {len(demo_files)} demo files")
        
        results = []
        for demo_file in demo_files:
            try:
                result = self.process_demo_file(
                    demo_file, 
                    resample_interval=resample_interval,
                    calculate_features=calculate_features,
                    save_formats=save_formats,
                    create_visualizations=create_visualizations
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {demo_file.name}: {e}")
                results.append({
                    "status": "error",
                    "demo_file": demo_file.name,
                    "error": str(e)
                })
        
        # Save batch summary
        batch_summary = {
            "processed_at": datetime.now().isoformat(),
            "total_demos": len(demo_files),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] != "success"]),
            "results": results
        }
        
        summary_path = self.output_dir / "batch_processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        print(f"\n=== Batch Processing Complete ===")
        print(f"Processed: {batch_summary['successful']}/{batch_summary['total_demos']} demos successfully")
        print(f"Results saved to: {self.output_dir}")
        
        return results


def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract player positions from CS2/CS:GO demo files")
    parser.add_argument("demo_path", help="Path to demo file or directory containing demos")
    parser.add_argument("--output-dir", default="position_data", help="Output directory")
    parser.add_argument("--resample", default="1S", help="Time series resampling interval")
    parser.add_argument("--no-features", action="store_true", help="Skip movement feature calculation")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization creation")
    parser.add_argument("--formats", nargs="+", default=["csv"], 
                       choices=["csv", "parquet", "json"], help="Output formats")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = PlayerPositionExtractor(output_dir=args.output_dir)
    
    demo_path = Path(args.demo_path)
    
    if demo_path.is_file():
        # Process single file
        result = extractor.process_demo_file(
            demo_path,
            resample_interval=args.resample,
            calculate_features=not args.no_features,
            save_formats=args.formats,
            create_visualizations=not args.no_plots
        )
        print(f"\nResult: {result}")
        
    elif demo_path.is_dir():
        # Batch process directory
        results = extractor.batch_process(
            demo_path,
            resample_interval=args.resample,
            calculate_features=not args.no_features,
            save_formats=args.formats,
            create_visualizations=not args.no_plots
        )
        
    else:
        print(f"Error: {demo_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
