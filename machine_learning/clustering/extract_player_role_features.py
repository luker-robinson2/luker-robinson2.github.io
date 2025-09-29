#!/usr/bin/env python3
"""
Comprehensive Player Role Feature Extraction from CS2 Demo Files

This script extracts detailed player role features directly from demo files
for clustering analysis. Features include combat, positional, economic, 
and team coordination metrics.

Usage:
    python extract_player_role_features.py --input_dir ../demos_extracted --output_dir .
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import pandas as pd
import numpy as np
from demoparser2 import DemoParser

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class PlayerRoleFeatureExtractor:
    """Extract comprehensive player role features from CS2 demo files."""
    
    def __init__(self, demo_path: str):
        self.demo_path = Path(demo_path)
        self.parser = DemoParser(str(demo_path))
        self.demo_name = self.demo_path.stem
        self.features = {}
        
    def extract_all_features(self) -> Dict:
        """Extract all available features from the demo."""
        print(f"Extracting features from {self.demo_name}...")
        
        try:
            # Basic match information
            header = self.parser.parse_header()
            self.features['match_info'] = {
                'demo_file': self.demo_name,
                'map_name': header.get('map_name', 'unknown'),
                'match_id': header.get('match_id', 'unknown'),
                'server_name': header.get('server_name', 'unknown')
            }
            
            # Parse different event types
            self._extract_round_events()
            self._extract_kill_events()
            self._extract_bomb_events()
            self._extract_damage_events()
            self._extract_player_positions()
            self._extract_economy_events()
            
            # Calculate derived features
            self._calculate_player_features()
            
            return self.features
            
        except Exception as e:
            print(f"Error extracting features from {self.demo_name}: {e}")
            return {}
    
    def _extract_round_events(self):
        """Extract round start/end events."""
        try:
            rounds_df = self.parser.parse_event("round_end")
            if isinstance(rounds_df, list):
                rounds_df = pd.DataFrame(rounds_df)
            
            # Also get round start events for timing
            round_start_df = self.parser.parse_event("round_start")
            if isinstance(round_start_df, list):
                round_start_df = pd.DataFrame(round_start_df)
            
            self.features['rounds'] = rounds_df
            self.features['round_starts'] = round_start_df
            
            print(f"  Found {len(rounds_df)} rounds")
            
        except Exception as e:
            print(f"  Could not extract round events: {e}")
            self.features['rounds'] = pd.DataFrame()
            self.features['round_starts'] = pd.DataFrame()
    
    def _extract_kill_events(self):
        """Extract player death/kill events."""
        try:
            deaths_df = self.parser.parse_event("player_death")
            if isinstance(deaths_df, list):
                deaths_df = pd.DataFrame(deaths_df)
            
            self.features['deaths'] = deaths_df
            print(f"  Found {len(deaths_df)} death events")
            
        except Exception as e:
            print(f"  Could not extract death events: {e}")
            self.features['deaths'] = pd.DataFrame()
    
    def _extract_bomb_events(self):
        """Extract bomb-related events."""
        bomb_events = {}
        
        for event_type in ['bomb_planted', 'bomb_defused', 'bomb_exploded']:
            try:
                events_df = self.parser.parse_event(event_type)
                if isinstance(events_df, list):
                    events_df = pd.DataFrame(events_df)
                bomb_events[event_type] = events_df
                print(f"  Found {len(events_df)} {event_type} events")
                
            except Exception as e:
                print(f"  Could not extract {event_type}: {e}")
                bomb_events[event_type] = pd.DataFrame()
        
        self.features['bomb_events'] = bomb_events
    
    def _extract_damage_events(self):
        """Extract player damage events."""
        try:
            damage_df = self.parser.parse_event("player_hurt")
            if isinstance(damage_df, list):
                damage_df = pd.DataFrame(damage_df)
            
            self.features['damage'] = damage_df
            print(f"  Found {len(damage_df)} damage events")
            
        except Exception as e:
            print(f"  Could not extract damage events: {e}")
            self.features['damage'] = pd.DataFrame()
    
    def _extract_player_positions(self):
        """Extract player position data."""
        try:
            # Get player position data (this might be large)
            positions_df = self.parser.parse_event("player_positions")
            if isinstance(positions_df, list):
                positions_df = pd.DataFrame(positions_df)
            
            # If positions are too large, sample them
            if len(positions_df) > 50000:
                positions_df = positions_df.sample(n=50000, random_state=42)
                print(f"  Sampled {len(positions_df)} position records (was too large)")
            
            self.features['positions'] = positions_df
            print(f"  Found {len(positions_df)} position records")
            
        except Exception as e:
            print(f"  Could not extract positions: {e}")
            self.features['positions'] = pd.DataFrame()
    
    def _extract_economy_events(self):
        """Extract economy-related events."""
        economy_events = {}
        
        for event_type in ['item_purchase', 'item_pickup', 'item_drop']:
            try:
                events_df = self.parser.parse_event(event_type)
                if isinstance(events_df, list):
                    events_df = pd.DataFrame(events_df)
                economy_events[event_type] = events_df
                
            except Exception as e:
                economy_events[event_type] = pd.DataFrame()
        
        self.features['economy'] = economy_events
        print(f"  Extracted economy events")
    
    def _calculate_player_features(self):
        """Calculate derived player features from raw events."""
        if self.features['deaths'].empty:
            print("  No death events to analyze")
            return
        
        deaths_df = self.features['deaths']
        damage_df = self.features['damage']
        bomb_events = self.features['bomb_events']
        
        # Get unique players
        all_players = set()
        if 'attacker_name' in deaths_df.columns:
            all_players.update(deaths_df['attacker_name'].dropna().unique())
        if 'user_name' in deaths_df.columns:
            all_players.update(deaths_df['user_name'].dropna().unique())
        if 'assister_name' in deaths_df.columns:
            all_players.update(deaths_df['assister_name'].dropna().unique())
        
        player_features = []
        
        for player in all_players:
            if pd.isna(player):
                continue
                
            features = self._calculate_single_player_features(
                player, deaths_df, damage_df, bomb_events
            )
            if features:
                player_features.append(features)
        
        self.features['player_features'] = pd.DataFrame(player_features)
        print(f"  Calculated features for {len(player_features)} players")
    
    def _calculate_single_player_features(self, player, deaths_df, damage_df, bomb_events) -> Dict:
        """Calculate features for a single player."""
        features = {
            'player_name': player,
            'demo_file': self.demo_name,
            'map_name': self.features['match_info']['map_name']
        }
        
        try:
            # Combat features
            kills = deaths_df[deaths_df['attacker_name'] == player]
            deaths = deaths_df[deaths_df['user_name'] == player]
            
            features['kills'] = len(kills)
            features['deaths'] = len(deaths)
            features['kd_ratio'] = features['kills'] / max(features['deaths'], 1)
            
            # Headshot percentage
            headshot_kills = len(kills[kills.get('headshot', False) == True])
            features['headshot_percentage'] = (headshot_kills / max(features['kills'], 1)) * 100
            
            # Enhanced timing and tactical features
            timing_features = self._calculate_timing_features(player, kills, deaths_df, bomb_events)
            features.update(timing_features)
            
            # Multi-kill analysis with round grouping
            multikill_features = self._calculate_multikill_features(player, kills, deaths_df)
            features.update(multikill_features)
            
            # Damage features
            if not damage_df.empty and 'attacker_name' in damage_df.columns:
                player_damage = damage_df[damage_df['attacker_name'] == player]
                features['total_damage'] = player_damage['dmg_health'].sum() if 'dmg_health' in player_damage.columns else 0
                features['damage_per_round'] = features['total_damage'] / max(len(self.features['rounds']), 1)
            
            # Bomb interaction features
            bomb_planted = len(bomb_events['bomb_planted'][bomb_events['bomb_planted'].get('user_name') == player])
            bomb_defused = len(bomb_events['bomb_defused'][bomb_events['bomb_defused'].get('user_name') == player])
            
            features['bombs_planted'] = bomb_planted
            features['bombs_defused'] = bomb_defused
            
            # Position-based features (simplified)
            if not self.features['positions'].empty and 'user_name' in self.features['positions'].columns:
                player_positions = self.features['positions'][self.features['positions']['user_name'] == player]
                if len(player_positions) > 0:
                    # Calculate movement metrics
                    if 'x' in player_positions.columns and 'y' in player_positions.columns:
                        # Calculate total distance moved (simplified)
                        x_diff = player_positions['x'].diff().abs().sum()
                        y_diff = player_positions['y'].diff().abs().sum()
                        features['total_distance_moved'] = np.sqrt(x_diff**2 + y_diff**2)
                        
                        # Average position (map area preference)
                        features['avg_x_position'] = player_positions['x'].mean()
                        features['avg_y_position'] = player_positions['y'].mean()
            
            # Economic features
            economy_features = self._calculate_economy_features(player)
            features.update(economy_features)
            
            return features
            
        except Exception as e:
            print(f"    Error calculating features for player {player}: {e}")
            return None
    
    def _calculate_timing_features(self, player, kills, deaths_df, bomb_events) -> Dict:
        """Calculate timing-related features for a player."""
        timing_features = {}
        
        try:
            # Average time to kill per round (simplified - using tick as time proxy)
            if not kills.empty and 'tick' in kills.columns:
                # Get round starts to calculate time in rounds
                round_starts = self.features['round_starts']
                if not round_starts.empty and 'tick' in round_starts.columns:
                    # Calculate average time to kill (simplified)
                    avg_tick_to_kill = kills['tick'].mean()
                    timing_features['avg_time_to_kill_ticks'] = avg_tick_to_kill
                else:
                    timing_features['avg_time_to_kill_ticks'] = kills['tick'].mean()
            
            # Opening kills (first kill of each round)
            opening_kills = 0
            if not kills.empty and 'tick' in kills.columns and not self.features['round_starts'].empty:
                round_starts = self.features['round_starts']
                for _, round_start in round_starts.iterrows():
                    round_start_tick = round_start.get('tick', 0)
                    # Find first kill in this round (simplified approach)
                    round_kills = kills[kills['tick'] >= round_start_tick]
                    if not round_kills.empty:
                        first_kill_tick = round_kills['tick'].min()
                        # Check if this player got the first kill
                        if first_kill_tick in round_kills['tick'].values:
                            opening_kills += 1
            
            timing_features['opening_kills'] = opening_kills
            
            # Kills before and after bomb planting
            kills_before_bomb = 0
            kills_after_bomb = 0
            
            if not kills.empty and 'tick' in kills.columns:
                for _, bomb_plant in bomb_events['bomb_planted'].iterrows():
                    bomb_tick = bomb_plant.get('tick', 0)
                    round_kills_before = len(kills[kills['tick'] < bomb_tick])
                    round_kills_after = len(kills[kills['tick'] > bomb_tick])
                    kills_before_bomb += round_kills_before
                    kills_after_bomb += round_kills_after
            
            timing_features['kills_before_bomb'] = kills_before_bomb
            timing_features['kills_after_bomb'] = kills_after_bomb
            
            # Average time alive (simplified - using death ticks)
            player_deaths = deaths_df[deaths_df['user_name'] == player]
            if not player_deaths.empty and 'tick' in player_deaths.columns:
                if not self.features['round_starts'].empty and 'tick' in self.features['round_starts'].columns:
                    round_starts = self.features['round_starts']
                    total_time_alive = 0
                    rounds_alive = 0
                    
                    for _, round_start in round_starts.iterrows():
                        round_start_tick = round_start.get('tick', 0)
                        # Find death in this round
                        round_death = player_deaths[player_deaths['tick'] >= round_start_tick]
                        if not round_death.empty:
                            death_tick = round_death['tick'].min()
                            time_alive = death_tick - round_start_tick
                            total_time_alive += time_alive
                            rounds_alive += 1
                        else:
                            # Player survived the round (estimate round length)
                            total_time_alive += 115000  # Approximate round length in ticks
                            rounds_alive += 1
                    
                    timing_features['avg_time_alive_ticks'] = total_time_alive / max(rounds_alive, 1)
                else:
                    timing_features['avg_time_alive_ticks'] = player_deaths['tick'].mean()
            
            return timing_features
            
        except Exception as e:
            print(f"    Error calculating timing features for player {player}: {e}")
            return {}
    
    def _calculate_multikill_features(self, player, kills, deaths_df) -> Dict:
        """Calculate multi-kill features for a player."""
        multikill_features = {}
        
        try:
            # Multi-kill analysis (simplified round-based grouping)
            multikill_2k = 0
            multikill_3k = 0
            multikill_4k = 0
            multikill_5k = 0
            
            if not kills.empty and 'tick' in kills.columns and not self.features['round_starts'].empty:
                round_starts = self.features['round_starts']
                
                for _, round_start in round_starts.iterrows():
                    round_start_tick = round_start.get('tick', 0)
                    # Find next round start or end
                    next_round_start = None
                    for _, next_round in round_starts.iterrows():
                        if next_round.get('tick', 0) > round_start_tick:
                            next_round_start = next_round.get('tick', 0)
                            break
                    
                    if next_round_start:
                        round_kills = kills[(kills['tick'] >= round_start_tick) & (kills['tick'] < next_round_start)]
                    else:
                        round_kills = kills[kills['tick'] >= round_start_tick]
                    
                    kill_count = len(round_kills)
                    if kill_count >= 2:
                        multikill_2k += 1
                    if kill_count >= 3:
                        multikill_3k += 1
                    if kill_count >= 4:
                        multikill_4k += 1
                    if kill_count >= 5:
                        multikill_5k += 1
            
            multikill_features['multi_kill_2k'] = multikill_2k
            multikill_features['multi_kill_3k'] = multikill_3k
            multikill_features['multi_kill_4k'] = multikill_4k
            multikill_features['multi_kill_5k'] = multikill_5k
            
            return multikill_features
            
        except Exception as e:
            print(f"    Error calculating multikill features for player {player}: {e}")
            return {}

    def _calculate_economy_features(self, player) -> Dict:
        """Calculate economy-related features for a player."""
        economy_features = {}
        
        try:
            # Weapon purchases
            purchases = self.features['economy']['item_purchase']
            if not purchases.empty and 'user_name' in purchases.columns:
                player_purchases = purchases[purchases['user_name'] == player]
                
                # Count weapon types
                weapons = player_purchases.get('weapon', pd.Series())
                economy_features['rifle_purchases'] = len(weapons[weapons.str.contains('rifle|ak47|m4a1', case=False, na=False)])
                economy_features['awp_purchases'] = len(weapons[weapons.str.contains('awp', case=False, na=False)])
                economy_features['pistol_purchases'] = len(weapons[weapons.str.contains('pistol|glock|usp', case=False, na=False)])
                economy_features['utility_purchases'] = len(weapons[weapons.str.contains('flash|smoke|grenade|molotov', case=False, na=False)])
                
                economy_features['total_purchases'] = len(player_purchases)
            
            return economy_features
            
        except Exception as e:
            print(f"    Error calculating economy features for player {player}: {e}")
            return {}


def extract_features_from_demo(demo_path: str) -> Dict:
    """Extract features from a single demo file."""
    extractor = PlayerRoleFeatureExtractor(demo_path)
    return extractor.extract_all_features()


def process_all_demos(input_dir: str, output_dir: str) -> None:
    """Process all demo files in the input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all demo files
    demo_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith('.dem'):
                demo_files.append(Path(root) / file)
    
    if not demo_files:
        print(f"No .dem files found in {input_dir}")
        return
    
    print(f"Found {len(demo_files)} demo files to process")
    
    all_player_features = []
    match_summaries = []
    
    for demo_file in demo_files:
        print(f"\n=== Processing {demo_file.name} ===")
        
        try:
            features = extract_features_from_demo(demo_file)
            
            if features and 'player_features' in features:
                # Add player features to combined dataset
                player_features = features['player_features']
                if not player_features.empty:
                    all_player_features.append(player_features)
                
                # Create match summary
                match_info = features['match_info']
                match_summaries.append({
                    'demo_file': match_info['demo_file'],
                    'map_name': match_info['map_name'],
                    'match_id': match_info['match_id'],
                    'num_players': len(player_features),
                    'total_kills': player_features['kills'].sum() if 'kills' in player_features.columns else 0,
                    'total_deaths': player_features['deaths'].sum() if 'deaths' in player_features.columns else 0,
                    'processing_status': 'success'
                })
                
                # Save individual demo features
                demo_output_file = output_path / f"{demo_file.stem}_features.csv"
                player_features.to_csv(demo_output_file, index=False)
                print(f"  Saved {len(player_features)} player features to {demo_output_file}")
            
        except Exception as e:
            print(f"  Failed to process {demo_file.name}: {e}")
            match_summaries.append({
                'demo_file': demo_file.stem,
                'map_name': 'unknown',
                'match_id': 'unknown',
                'num_players': 0,
                'total_kills': 0,
                'total_deaths': 0,
                'processing_status': f'error: {str(e)[:100]}'
            })
    
    # Combine all player features
    if all_player_features:
        combined_features = pd.concat(all_player_features, ignore_index=True)
        combined_output = output_path / "player_role_features_combined.csv"
        combined_features.to_csv(combined_output, index=False)
        print(f"\n=== Combined Results ===")
        print(f"Saved {len(combined_features)} total player features to {combined_output}")
        
        # Save feature summary
        feature_summary = {
            'total_players': len(combined_features),
            'unique_maps': combined_features['map_name'].nunique() if 'map_name' in combined_features.columns else 0,
            'avg_kills_per_player': combined_features['kills'].mean() if 'kills' in combined_features.columns else 0,
            'avg_deaths_per_player': combined_features['deaths'].mean() if 'deaths' in combined_features.columns else 0,
            'avg_kd_ratio': combined_features['kd_ratio'].mean() if 'kd_ratio' in combined_features.columns else 0,
        }
        
        summary_df = pd.DataFrame([feature_summary])
        summary_output = output_path / "feature_summary.csv"
        summary_df.to_csv(summary_output, index=False)
        print(f"Saved feature summary to {summary_output}")
    
    # Save match summaries
    match_summary_df = pd.DataFrame(match_summaries)
    match_summary_output = output_path / "match_summaries.csv"
    match_summary_df.to_csv(match_summary_output, index=False)
    print(f"Saved {len(match_summaries)} match summaries to {match_summary_output}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract player role features from CS2 demo files")
    parser.add_argument("--input_dir", default="../demos_extracted", 
                       help="Input directory containing demo files")
    parser.add_argument("--output_dir", default=".", 
                       help="Output directory for generated CSV files")
    
    args = parser.parse_args()
    
    print("=== CS2 Player Role Feature Extraction ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    process_all_demos(args.input_dir, args.output_dir)
    
    print("\n=== Feature Extraction Complete ===")


if __name__ == "__main__":
    main()
