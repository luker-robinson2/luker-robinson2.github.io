#!/usr/bin/env python3
"""
Enhanced Round Feature Extractor V2
===================================
Extracts state-based features at multiple time points per round to improve
prediction accuracy. Focuses on economy, bomb state, time, and player counts.

Key Innovation: Time-series sampling - multiple samples per round instead of
single end-of-round sample with outcome features.
"""

from demoparser2 import DemoParser
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class EnhancedRoundFeatureExtractor:
    """Extract predictive state-based features from CS2 demo files."""
    
    def __init__(self, tick_rate: int = 64):
        """
        Initialize the extractor.
        
        Args:
            tick_rate: Server tick rate (default 64 for competitive CS2)
        """
        self.tick_rate = tick_rate
        
    def extract_from_demo(self, demo_path: Path, 
                         time_samples: List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) -> pd.DataFrame:
        """
        Extract enhanced features from a demo file.
        
        Args:
            demo_path: Path to .dem file
            time_samples: Seconds into each round to sample (default: 10 samples every 10s)
            
        Returns:
            DataFrame with features at each time sample for each round
        """
        print(f"Sampling at: {time_samples} seconds")
        
        try:
            parser = DemoParser(str(demo_path))
            
            # Parse header for metadata
            header = parser.parse_header()
            map_name = header.get('map_name', 'unknown')
            
            # Parse round end events (these mark the end of each round)
            round_ends = parser.parse_event("round_end")
            if isinstance(round_ends, list):
                round_ends = pd.DataFrame(round_ends)
            
            if round_ends.empty:
                print(f"  Warning: No round data found")
                return pd.DataFrame()
            
            # Sort by tick to get chronological order
            round_ends = round_ends.sort_values('tick').reset_index(drop=True)
            
            # Parse bomb events
            bomb_events = self._parse_bomb_events(parser)
            
            # Extract features for each round at each time sample
            all_features = []
            
            # Use consecutive round_end events to define rounds
            # Round i spans from end of round i-1 to end of round i
            for i in range(1, len(round_ends)):
                round_start_tick = round_ends.iloc[i-1]['tick']
                round_end_tick = round_ends.iloc[i]['tick']
                round_num = round_ends.iloc[i].get('round', i)
                winning_team = round_ends.iloc[i].get('winner', None)
                
                # Convert to team labels
                # demoparser2 returns 'T' or 'CT' as strings, not integers
                if winning_team in ['T', 'TERRORIST', 2]:
                    winner_label = 'T'
                elif winning_team in ['CT', 'CT', 3]:
                    winner_label = 'CT'
                else:
                    # Skip rounds with no winner (warmup, etc.)
                    continue
                
                round_duration_ticks = round_end_tick - round_start_tick
                round_duration_sec = round_duration_ticks / self.tick_rate
                
                # Skip very short rounds (< 10s, likely errors)
                if round_duration_sec < 10:
                    continue
                
                if i == 1:  # Debug first round
                    print(f"  Round {round_num}: ticks {round_start_tick}-{round_end_tick}, duration={round_duration_sec:.1f}s, winner={winner_label}")
                
                # Extract features at each time sample
                for time_sec in time_samples:
                    # Skip if this time is beyond the round duration
                    if time_sec >= round_duration_sec:
                        if i == 1:
                            print(f"    Skipping time {time_sec}s (round ended at {round_duration_sec:.1f}s)")
                        continue
                    
                    sample_tick = round_start_tick + int(time_sec * self.tick_rate)
                    
                    if i == 1:
                        print(f"    Extracting features at {time_sec}s (tick {sample_tick})...")
                    
                    # Extract game state at this tick
                    features = self._extract_tick_features(
                        parser, sample_tick, round_num, time_sec, 
                        round_start_tick, bomb_events, map_name
                    )
                    
                    if features:
                        features['winning_team'] = winner_label
                        features['round_duration'] = round_duration_sec
                        features['match_id'] = demo_path.stem
                        all_features.append(features)
                        if i == 1:
                            print(f"      ✓ Features extracted successfully")
                    else:
                        if i == 1:
                            print(f"      ✗ No features returned")
            
            if not all_features:
                print(f"  Warning: No features extracted")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_features)
            print(f"  Extracted {len(df)} samples from {len(round_ends)-1} rounds")
            return df
            
        except Exception as e:
            print(f"  Error processing {demo_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _parse_bomb_events(self, parser: DemoParser) -> Dict[str, pd.DataFrame]:
        """Parse all bomb-related events."""
        bomb_events = {}
        
        for event_type in ['bomb_planted', 'bomb_defused', 'bomb_exploded', 'bomb_begindefuse']:
            try:
                events = parser.parse_event(event_type)
                if isinstance(events, list):
                    events = pd.DataFrame(events)
                bomb_events[event_type] = events if not events.empty else pd.DataFrame()
            except:
                bomb_events[event_type] = pd.DataFrame()
        
        return bomb_events
    
    def _extract_tick_features(self, parser: DemoParser, tick: int, round_num: int,
                               time_elapsed: float, round_start_tick: int,
                               bomb_events: Dict, map_name: str) -> Optional[Dict]:
        """Extract features at a specific tick."""
        try:
            # Parse game state at this tick
            tick_data = parser.parse_ticks([
                "team_name", "team_num", "is_alive", "health", "armor_value",
                "active_weapon_name", "current_equip_value",
                "X", "Y", "Z"
            ], ticks=[tick])
            
            if isinstance(tick_data, list):
                tick_data = pd.DataFrame(tick_data)
            
            if tick_data.empty:
                return None
            
            # Separate by team
            t_players = tick_data[tick_data['team_name'] == 'TERRORIST']
            ct_players = tick_data[tick_data['team_name'] == 'CT']
            
            # Calculate features
            features = {
                'round_num': round_num,
                'time_elapsed': time_elapsed,
                'tick': tick,
                'map_name': map_name,
            }
            
            # Time features
            features.update(self._calculate_time_features(time_elapsed))
            
            # Player count features
            features.update(self._calculate_player_features(t_players, ct_players))
            
            # Economy features (CRITICAL)
            features.update(self._calculate_economy_features(t_players, ct_players))
            
            # Bomb state features
            features.update(self._calculate_bomb_features(
                tick, round_start_tick, bomb_events, t_players, ct_players
            ))
            
            # Weapon features
            features.update(self._calculate_weapon_features(t_players, ct_players))
            
            return features
            
        except Exception as e:
            # Silently skip individual tick errors
            return None
    
    def _calculate_time_features(self, time_elapsed: float) -> Dict:
        """Calculate time-based features."""
        return {
            'round_time_elapsed': time_elapsed,
            'round_time_remaining': max(0, 115 - time_elapsed),  # 115s default round time
            'time_phase': 0 if time_elapsed < 30 else (1 if time_elapsed < 70 else 2),
        }
    
    def _calculate_player_features(self, t_players: pd.DataFrame, 
                                   ct_players: pd.DataFrame) -> Dict:
        """Calculate player count and health features."""
        t_alive = t_players[t_players['is_alive'] == True]
        ct_alive = ct_players[ct_players['is_alive'] == True]
        
        return {
            't_players_alive': len(t_alive),
            'ct_players_alive': len(ct_alive),
            'player_count_advantage': len(t_alive) - len(ct_alive),
            't_total_hp': t_alive['health'].sum() if len(t_alive) > 0 else 0,
            'ct_total_hp': ct_alive['health'].sum() if len(ct_alive) > 0 else 0,
            't_avg_hp': t_alive['health'].mean() if len(t_alive) > 0 else 0,
            'ct_avg_hp': ct_alive['health'].mean() if len(ct_alive) > 0 else 0,
            't_total_armor': t_alive['armor_value'].sum() if len(t_alive) > 0 else 0,
            'ct_total_armor': ct_alive['armor_value'].sum() if len(ct_alive) > 0 else 0,
        }
    
    def _calculate_economy_features(self, t_players: pd.DataFrame,
                                    ct_players: pd.DataFrame) -> Dict:
        """Calculate economy and equipment features - CRITICAL for prediction."""
        t_alive = t_players[t_players['is_alive'] == True]
        ct_alive = ct_players[ct_players['is_alive'] == True]
        
        # Equipment value (if available in tick data)
        t_equip_value = 0
        ct_equip_value = 0
        
        if 'current_equip_value' in t_alive.columns:
            # Convert to int to avoid uint64 overflow issues with negative differences
            t_equip_value = int(t_alive['current_equip_value'].sum())
            ct_equip_value = int(ct_alive['current_equip_value'].sum())
        
        # Calculate average equipment per alive player
        t_avg_equip = t_equip_value / len(t_alive) if len(t_alive) > 0 else 0
        ct_avg_equip = ct_equip_value / len(ct_alive) if len(ct_alive) > 0 else 0
        
        return {
            't_equipment_value': int(t_equip_value),
            'ct_equipment_value': int(ct_equip_value),
            't_avg_equipment_value': float(t_avg_equip),
            'ct_avg_equipment_value': float(ct_avg_equip),
            'equipment_advantage': int(t_equip_value) - int(ct_equip_value),  # Explicit signed int
            'equipment_advantage_per_player': float(t_avg_equip - ct_avg_equip),
        }
    
    def _calculate_bomb_features(self, tick: int, round_start_tick: int,
                                 bomb_events: Dict, t_players: pd.DataFrame,
                                 ct_players: pd.DataFrame) -> Dict:
        """Calculate bomb state features."""
        features = {
            'bomb_planted': 0,
            'time_since_plant': 0,
            'time_until_explosion': 0,
            'bomb_being_defused': 0,
        }
        
        # Check if bomb is planted
        planted_events = bomb_events.get('bomb_planted', pd.DataFrame())
        if not planted_events.empty:
            # Find plants that happened before this tick in this round
            relevant_plants = planted_events[
                (planted_events['tick'] <= tick) & 
                (planted_events['tick'] >= round_start_tick)
            ]
            
            if not relevant_plants.empty:
                plant_tick = relevant_plants.iloc[-1]['tick']  # Most recent plant
                features['bomb_planted'] = 1
                time_since_plant = (tick - plant_tick) / self.tick_rate
                features['time_since_plant'] = time_since_plant
                features['time_until_explosion'] = max(0, 40 - time_since_plant)  # 40s timer
        
        # Check if bomb is being defused
        defuse_events = bomb_events.get('bomb_begindefuse', pd.DataFrame())
        if not defuse_events.empty and features['bomb_planted'] == 1:
            relevant_defuses = defuse_events[
                (defuse_events['tick'] <= tick) &
                (defuse_events['tick'] >= round_start_tick)
            ]
            if not relevant_defuses.empty:
                # Check if defuse is still in progress
                last_defuse_tick = relevant_defuses.iloc[-1]['tick']
                # Defuse takes ~10s with kit, ~5s without (simplified)
                if (tick - last_defuse_tick) < (10 * self.tick_rate):
                    features['bomb_being_defused'] = 1
        
        return features
    
    def _calculate_weapon_features(self, t_players: pd.DataFrame,
                                   ct_players: pd.DataFrame) -> Dict:
        """Calculate weapon-related features."""
        t_alive = t_players[t_players['is_alive'] == True]
        ct_alive = ct_players[ct_players['is_alive'] == True]
        
        # Count AWPs (most impactful weapon)
        t_awps = 0
        ct_awps = 0
        
        if 'active_weapon_name' in t_alive.columns:
            t_awps = (t_alive['active_weapon_name'].str.contains('awp', case=False, na=False)).sum()
            ct_awps = (ct_alive['active_weapon_name'].str.contains('awp', case=False, na=False)).sum()
        
        # Count rifles
        t_rifles = 0
        ct_rifles = 0
        
        if 'active_weapon_name' in t_alive.columns:
            rifle_pattern = 'ak47|m4a1|aug|sg556|famas|galilar'
            t_rifles = (t_alive['active_weapon_name'].str.contains(rifle_pattern, case=False, na=False)).sum()
            ct_rifles = (ct_alive['active_weapon_name'].str.contains(rifle_pattern, case=False, na=False)).sum()
        
        return {
            't_awp_count': t_awps,
            'ct_awp_count': ct_awps,
            'awp_advantage': t_awps - ct_awps,
            't_rifle_count': t_rifles,
            'ct_rifle_count': ct_rifles,
            'rifle_advantage': t_rifles - ct_rifles,
        }


def process_single_demo(demo_path: Path, output_dir: Path,
                       time_samples: List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) -> pd.DataFrame:
    """
    Process a single demo file and save features.
    
    Args:
        demo_path: Path to demo file
        output_dir: Directory to save output
        time_samples: Time points to sample
        
        Returns:
        DataFrame with extracted features
    """
    extractor = EnhancedRoundFeatureExtractor()
    features_df = extractor.extract_from_demo(demo_path, time_samples)
    
    if not features_df.empty:
        # Save individual match features
        output_path = output_dir / f"{demo_path.stem}_enhanced_features.csv"
        features_df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path.name}")
    
    return features_df


if __name__ == "__main__":
    # Use both sets of demo files
    DEMOS_DIR_1 = Path(__file__).parent.parent / "demos_extracted"
    DEMOS_DIR_2 = Path("/Users/lukerobinson/Dropbox/school/csci_5502/CS_ANALYTICS_DM/machine_learning/demos")
    OUTPUT_DIR = Path(__file__).parent / "enhanced_features"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("ENHANCED ROUND FEATURE EXTRACTION V2")
    print("="*80)
    
    # Find all demo files from both locations
    demo_files = []
    
    print(f"\nLooking for demos in location 1: {DEMOS_DIR_1}")
    if DEMOS_DIR_1.exists():
        demos_1 = list(DEMOS_DIR_1.rglob("*.dem"))
        print(f"  Found {len(demos_1)} demo files")
        demo_files.extend(demos_1)
    else:
        print(f"  Directory not found")
    
    print(f"\nLooking for demos in location 2: {DEMOS_DIR_2}")
    if DEMOS_DIR_2.exists():
        demos_2 = list(DEMOS_DIR_2.glob("*.dem"))
        print(f"  Found {len(demos_2)} demo files")
        demo_files.extend(demos_2)
    else:
        print(f"  Directory not found")
    
    demo_files = sorted(demo_files)
    
    if not demo_files:
        print("\nNo demo files found in either location!")
    else:
        print(f"\n{'='*80}")
        print(f"Total: {len(demo_files)} valid demo files across both locations")
        print(f"{'='*80}")
        
        # Process first 3 for testing
        print(f"\nProcessing first 3 demos for validation...")
        for demo_path in demo_files[:3]:
            try:
                process_single_demo(demo_path, OUTPUT_DIR)
            except Exception as e:
                print(f"  Failed to process {demo_path.name}: {e}")
        
        print(f"\n{'='*80}")
        print(f"Processing complete! Features saved to: {OUTPUT_DIR}")
        print(f"{'='*80}")

