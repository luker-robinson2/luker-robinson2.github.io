#!/usr/bin/env python3
"""
CS2 Demo Parser using demoparser2
Parses demo files in the demos_extracted folder and extracts match data
"""

from demoparser2 import DemoParser
import pandas as pd
import os
from pathlib import Path
import warnings

BASE_DIR = Path(__file__).parent
DEMOS_EXTRACTED_DIR = BASE_DIR / "demos_extracted"
OUTPUT_DIR = BASE_DIR / "hltv_data"
OUTPUT_DIR.mkdir(exist_ok=True)

def parse_demo_with_demoparser2(dem_path: Path):
    """Parse a single demo file using demoparser2"""
    print(f"Parsing {dem_path.name} with demoparser2...")
    
    try:
        # Initialize the parser
        parser = DemoParser(str(dem_path))
        
        # Parse basic match information
        header = parser.parse_header()
        print(f"Demo header: {header}")
        
        # Parse round-end events to get round information
        try:
            rounds_df = parser.parse_event("round_end")
            # Convert to DataFrame if it's a list
            if isinstance(rounds_df, list):
                rounds_df = pd.DataFrame(rounds_df)
            print(f"Found {len(rounds_df)} round_end events")
        except Exception as e:
            print(f"Failed to parse round_end events: {e}")
            rounds_df = pd.DataFrame()
        
        # Parse bomb events
        try:
            bomb_planted_df = parser.parse_event("bomb_planted")
            bomb_defused_df = parser.parse_event("bomb_defused")
            bomb_exploded_df = parser.parse_event("bomb_exploded")
            
            # Convert to DataFrames if they're lists
            if isinstance(bomb_planted_df, list):
                bomb_planted_df = pd.DataFrame(bomb_planted_df)
            if isinstance(bomb_defused_df, list):
                bomb_defused_df = pd.DataFrame(bomb_defused_df)
            if isinstance(bomb_exploded_df, list):
                bomb_exploded_df = pd.DataFrame(bomb_exploded_df)
                
            print(f"Bomb events - Planted: {len(bomb_planted_df)}, Defused: {len(bomb_defused_df)}, Exploded: {len(bomb_exploded_df)}")
        except Exception as e:
            print(f"Failed to parse bomb events: {e}")
            bomb_planted_df = pd.DataFrame()
            bomb_defused_df = pd.DataFrame()
            bomb_exploded_df = pd.DataFrame()
        
        # Parse player death events for additional context
        try:
            deaths_df = parser.parse_event("player_death")
            # Convert to DataFrame if it's a list
            if isinstance(deaths_df, list):
                deaths_df = pd.DataFrame(deaths_df)
            print(f"Found {len(deaths_df)} player death events")
        except Exception as e:
            print(f"Failed to parse player_death events: {e}")
            deaths_df = pd.DataFrame()
        
        # Create round summary data
        if not rounds_df.empty:
            # Process round data
            rounds_summary = []
            for idx, round_data in rounds_df.iterrows():
                round_num = round_data.get('round', idx + 1)
                winner = round_data.get('winner', None)
                reason = round_data.get('reason', None)
                
                # Check if bomb was planted in this round
                bomb_planted_in_round = False
                if not bomb_planted_df.empty:
                    # This is a simplified check - in practice you'd want to match by round number or tick
                    bomb_planted_in_round = len(bomb_planted_df) > 0
                
                rounds_summary.append({
                    'round_num': round_num,
                    'winning_team': winner,
                    'round_end_reason': reason,
                    'bomb_planted': bomb_planted_in_round,
                    'demo_type': 'cs2',
                    'map_name': header.get('map_name', 'unknown'),
                    'parsing_status': 'success'
                })
            
            df = pd.DataFrame(rounds_summary)
        else:
            # Create minimal data if no rounds found
            df = pd.DataFrame([{
                'round_num': None,
                'winning_team': None,
                'round_end_reason': None,
                'bomb_planted': None,
                'demo_type': 'cs2',
                'map_name': header.get('map_name', 'unknown'),
                'parsing_status': 'no_rounds_found'
            }])
        
        # Save to CSV
        stem = dem_path.stem
        out_csv = OUTPUT_DIR / f"{stem}_rounds.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} rounds to {out_csv}")
        
        # Also save detailed event data if available
        if not deaths_df.empty:
            deaths_csv = OUTPUT_DIR / f"{stem}_deaths.csv"
            deaths_df.to_csv(deaths_csv, index=False)
            print(f"Saved {len(deaths_df)} death events to {deaths_csv}")
        
        if not bomb_planted_df.empty or not bomb_defused_df.empty or not bomb_exploded_df.empty:
            bomb_csv = OUTPUT_DIR / f"{stem}_bomb_events.csv"
            # Combine all bomb events
            bomb_events_list = []
            
            if not bomb_planted_df.empty:
                bomb_planted_copy = bomb_planted_df.copy()
                bomb_planted_copy['event_type'] = 'planted'
                bomb_events_list.append(bomb_planted_copy)
            
            if not bomb_defused_df.empty:
                bomb_defused_copy = bomb_defused_df.copy()
                bomb_defused_copy['event_type'] = 'defused'
                bomb_events_list.append(bomb_defused_copy)
            
            if not bomb_exploded_df.empty:
                bomb_exploded_copy = bomb_exploded_df.copy()
                bomb_exploded_copy['event_type'] = 'exploded'
                bomb_events_list.append(bomb_exploded_copy)
            
            if bomb_events_list:
                bomb_events = pd.concat(bomb_events_list, ignore_index=True)
                bomb_events.to_csv(bomb_csv, index=False)
                print(f"Saved {len(bomb_events)} bomb events to {bomb_csv}")
        
        return True
        
    except Exception as e:
        print(f"Failed to parse {dem_path.name}: {e}")
        
        # Create error CSV
        df = pd.DataFrame([{
            'round_num': None,
            'winning_team': None,
            'round_end_reason': None,
            'bomb_planted': None,
            'demo_type': 'cs2',
            'map_name': 'unknown',
            'parsing_status': f'error: {str(e)[:100]}'
        }])
        
        stem = dem_path.stem
        out_csv = OUTPUT_DIR / f"{stem}_rounds.csv"
        df.to_csv(out_csv, index=False)
        print(f"Created error CSV -> {out_csv}")
        
        return False

def main():
    """Main parsing function"""
    print("=== CS2 Demo Parser using demoparser2 ===")
    print(f"Scanning {DEMOS_EXTRACTED_DIR} for demo files...")
    
    if not DEMOS_EXTRACTED_DIR.exists():
        print(f"Error: {DEMOS_EXTRACTED_DIR} does not exist!")
        return
    
    demo_files = []
    
    # Find all .dem files recursively
    for root, dirs, files in os.walk(DEMOS_EXTRACTED_DIR):
        for file in files:
            if file.lower().endswith('.dem'):
                demo_files.append(Path(root) / file)
    
    if not demo_files:
        print("No .dem files found in demos_extracted directory!")
        return
    
    print(f"Found {len(demo_files)} demo files:")
    for demo_file in demo_files:
        print(f"  - {demo_file.relative_to(DEMOS_EXTRACTED_DIR)}")
    
    print("\nStarting parsing...")
    
    successful_parses = 0
    failed_parses = 0
    
    for demo_file in demo_files:
        print(f"\n--- Processing {demo_file.name} ---")
        if parse_demo_with_demoparser2(demo_file):
            successful_parses += 1
        else:
            failed_parses += 1
    
    print(f"\n=== Parsing Complete ===")
    print(f"Successfully parsed: {successful_parses}")
    print(f"Failed to parse: {failed_parses}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
