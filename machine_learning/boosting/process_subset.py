#!/usr/bin/env python3
"""
Process a subset of demos (10 total) for faster iteration.
"""

from pathlib import Path
import pandas as pd
from extract_enhanced_round_features import EnhancedRoundFeatureExtractor

# Directories
DEMOS_DIR_1 = Path(__file__).parent.parent / "demos_extracted"
DEMOS_DIR_2 = Path("/Users/lukerobinson/Dropbox/school/csci_5502/CS_ANALYTICS_DM/machine_learning/demos")
OUTPUT_DIR = Path(__file__).parent / "enhanced_features"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("PROCESSING ALL AVAILABLE DEMOS")
print("="*80)

# Find all demo files
demo_files = []
if DEMOS_DIR_1.exists():
    demo_files.extend(sorted(list(DEMOS_DIR_1.rglob("*.dem"))))
if DEMOS_DIR_2.exists():
    demo_files.extend(sorted(list(DEMOS_DIR_2.glob("*.dem"))))

print(f"\nFound {len(demo_files)} total demos")
print(f"Will process all demos with 10 time samples per round (every 10 seconds)")

# Check which ones are already done
existing_files = set([f.stem.replace('_enhanced_features', '') for f in OUTPUT_DIR.glob("*_enhanced_features.csv")])
print(f"Already processed: {len(existing_files)} demos")

# Filter to demos that need processing - PROCESS ALL, NOT JUST 5
demos_to_process = [d for d in demo_files if d.stem not in existing_files]

if not demos_to_process:
    print("\nAll demos already processed! Combining results...")
else:
    print(f"\nNeed to process: {len(demos_to_process)} more demos")
    print("="*80)
    
    # Process demos
    extractor = EnhancedRoundFeatureExtractor()
    
    for i, demo_path in enumerate(demos_to_process, 1):
        print(f"\n[{i}/{len(demos_to_process)}] {demo_path.name}")
        try:
            features_df = extractor.extract_from_demo(demo_path)
            
            if not features_df.empty:
                output_path = OUTPUT_DIR / f"{demo_path.stem}_enhanced_features.csv"
                features_df.to_csv(output_path, index=False)
                print(f"✓ Extracted {len(features_df)} samples")
            else:
                print(f"✗ No features extracted")
        except Exception as e:
            print(f"✗ Error: {e}")

# Combine all processed features
print(f"\n{'='*80}")
print("COMBINING FEATURES")
print("="*80)

all_feature_files = list(OUTPUT_DIR.glob("*_enhanced_features.csv"))
print(f"\nFound {len(all_feature_files)} feature files")

if all_feature_files:
    all_dfs = []
    for f in all_feature_files:
        df = pd.read_csv(f)
        all_dfs.append(df)
        print(f"  {f.stem}: {len(df)} samples")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save combined
    combined_path = OUTPUT_DIR / "combined_enhanced_features.csv"
    combined_df.to_csv(combined_path, index=False)
    
    print(f"\n{'='*80}")
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Total samples: {len(combined_df)}")
    print(f"Total matches: {combined_df['match_id'].nunique()}")
    print(f"\nWinning team distribution:")
    print(combined_df['winning_team'].value_counts())
    print(f"\nTime samples per round: {combined_df.groupby(['match_id', 'round_num']).size().mean():.1f}")
    print(f"\nSaved to: {combined_path}")
    print("="*80)
else:
    print("No feature files found!")

