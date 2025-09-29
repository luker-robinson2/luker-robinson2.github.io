#!/usr/bin/env python3
"""
Improved Round Parser for CS2 Position Data
==========================================

Uses actual round data from rounds_cleaned.csv and event timing to create 
proper round boundaries instead of artificial time divisions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def get_round_boundaries_from_events(match_id, tick_rate=128):
    """
    Extract actual round boundaries using bomb events and deaths.
    
    Args:
        match_id: Match identifier (e.g., 'aether-vs-full-send-m1-mirage')
        tick_rate: Server tick rate (default 128 for competitive)
    
    Returns:
        List of round dictionaries with start/end times
    """
    
    # Load event data
    try:
        bomb_df = pd.read_csv('bomb_events_cleaned.csv')
        deaths_df = pd.read_csv('deaths_cleaned.csv')
        rounds_df = pd.read_csv('rounds_cleaned.csv')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return []
    
    # Filter for specific match
    match_bombs = bomb_df[bomb_df['match_id'] == match_id].copy()
    match_deaths = deaths_df[deaths_df['match_id'] == match_id].copy()
    match_rounds = rounds_df[rounds_df['match_id'] == match_id].copy()
    
    if match_rounds.empty:
        print(f"No round data found for match: {match_id}")
        return []
    
    # Convert ticks to timestamps
    match_bombs['timestamp'] = match_bombs['tick'] / tick_rate
    match_deaths['timestamp'] = match_deaths['tick'] / tick_rate
    
    # Get all events sorted by time
    all_events = []
    
    # Add bomb events
    for _, row in match_bombs.iterrows():
        all_events.append({
            'timestamp': row['timestamp'],
            'type': f"bomb_{row['event_type']}",
            'tick': row['tick']
        })
    
    # Add death events (sample every 10th to avoid too many)
    for _, row in match_deaths.iloc[::10].iterrows():
        all_events.append({
            'timestamp': row['timestamp'],
            'type': 'death',
            'tick': row['tick']
        })
    
    # Sort events by timestamp
    all_events.sort(key=lambda x: x['timestamp'])
    
    # Detect round boundaries using event clustering
    rounds = []
    
    if not all_events:
        print("No events found for round boundary detection")
        return []
    
    # Method 1: Use bomb plants as round markers
    bomb_plants = [e for e in all_events if e['type'] == 'bomb_planted']
    
    if len(bomb_plants) >= 2:
        # Use bomb plants to infer round boundaries
        for i, plant_event in enumerate(bomb_plants):
            round_start = plant_event['timestamp'] - 60  # Assume round started ~60s before plant
            
            if i < len(bomb_plants) - 1:
                next_plant = bomb_plants[i + 1]
                round_end = next_plant['timestamp'] - 30  # End ~30s before next round's plant
            else:
                # Last round - estimate end
                round_end = plant_event['timestamp'] + 30
            
            # Ensure positive duration
            if round_end > round_start:
                rounds.append({
                    'round_num': i + 1,
                    'start_time': max(0, round_start),
                    'end_time': round_end,
                    'duration': round_end - round_start,
                    'key_events': [plant_event],
                    'bomb_planted': True
                })
    
    # Method 2: If no bomb plants, use death event clustering
    if not rounds and all_events:
        # Group events into clusters (rounds) based on time gaps
        time_gaps = []
        for i in range(1, len(all_events)):
            gap = all_events[i]['timestamp'] - all_events[i-1]['timestamp']
            time_gaps.append(gap)
        
        # Find large gaps (likely between rounds)
        if time_gaps:
            gap_threshold = np.percentile(time_gaps, 90)  # Top 10% of gaps
            
            round_starts = [all_events[0]['timestamp']]
            for i, gap in enumerate(time_gaps):
                if gap > gap_threshold:
                    round_starts.append(all_events[i+1]['timestamp'])
            
            # Create rounds from detected starts
            for i, start_time in enumerate(round_starts):
                if i < len(round_starts) - 1:
                    end_time = round_starts[i + 1] - 10  # End 10s before next round
                else:
                    end_time = all_events[-1]['timestamp']
                
                rounds.append({
                    'round_num': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'key_events': [],
                    'bomb_planted': False
                })
    
    # Add round metadata from rounds_cleaned.csv
    for round_data in rounds:
        round_num = round_data['round_num']
        round_info = match_rounds[match_rounds['round_num'] == round_num]
        
        if not round_info.empty:
            round_data.update({
                'winning_team': round_info.iloc[0]['winning_team'],
                'round_end_reason': round_info.iloc[0]['round_end_reason'],
                'bomb_planted': round_info.iloc[0]['bomb_planted']
            })
    
    print(f"Detected {len(rounds)} actual rounds for {match_id}")
    return rounds

def load_position_data_with_rounds(demo_name):
    """
    Load position data and extract proper round boundaries.
    
    Args:
        demo_name: Name of the demo (e.g., 'aether-vs-full-send-m1-mirage')
    
    Returns:
        tuple: (position_dataframe, rounds_list)
    """
    
    # Load position data
    data_dir = Path("example_position_data")
    csv_file = data_dir / f"{demo_name}_positions_timeseries.csv"
    
    if not csv_file.exists():
        print(f"Position data not found: {csv_file}")
        return None, []
    
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['X', 'Y', 'player_name'])
    
    print(f"Loaded {len(df)} position records for {demo_name}")
    print(f"Timestamp range: {df['timestamp'].min():.1f} - {df['timestamp'].max():.1f} seconds")
    
    # Get actual round boundaries
    rounds = get_round_boundaries_from_events(demo_name)
    
    # If no rounds detected, fall back to time-based estimation
    if not rounds:
        print("Falling back to time-based round estimation...")
        total_time = df['timestamp'].max() - df['timestamp'].min()
        estimated_rounds = max(1, int(total_time / 120))  # 2-minute rounds
        
        start_time = df['timestamp'].min()
        for i in range(estimated_rounds):
            round_start = start_time + (i * 120)
            round_end = min(start_time + ((i + 1) * 120), df['timestamp'].max())
            
            rounds.append({
                'round_num': i + 1,
                'start_time': round_start,
                'end_time': round_end,
                'duration': round_end - round_start,
                'winning_team': 'Unknown',
                'round_end_reason': 'estimated',
                'bomb_planted': False
            })
    
    return df, rounds

# Test the improved parser
if __name__ == "__main__":
    demo_name = "aether-vs-full-send-m1-mirage"
    df, rounds = load_position_data_with_rounds(demo_name)
    
    if df is not None and rounds:
        print(f"\n📊 Analysis Results for {demo_name}")
        print(f"Position records: {len(df):,}")
        print(f"Detected rounds: {len(rounds)}")
        print(f"Total match time: {df['timestamp'].max():.1f} seconds")
        
        print("\n🎯 Round Summary:")
        for round_data in rounds[:5]:  # Show first 5 rounds
            print(f"Round {round_data['round_num']}: "
                  f"{round_data['start_time']:.1f}s - {round_data['end_time']:.1f}s "
                  f"({round_data['duration']:.1f}s) "
                  f"Winner: {round_data.get('winning_team', 'Unknown')}")
