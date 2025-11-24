#!/usr/bin/env python3
"""
Batch process all demo files to create the complete enhanced features dataset.
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
print("BATCH PROCESSING ALL DEMO FILES")
print("="*80)

# Find all demo files
demo_files = []

print(f"\nLocation 1: {DEMOS_DIR_1}")
if DEMOS_DIR_1.exists():
    demos_1 = sorted(list(DEMOS_DIR_1.rglob("*.dem")))
    print(f"  Found {len(demos_1)} demo files")
    demo_files.extend(demos_1)

print(f"\nLocation 2: {DEMOS_DIR_2}")
if DEMOS_DIR_2.exists():
    demos_2 = sorted(list(DEMOS_DIR_2.glob("*.dem")))
    print(f"  Found {len(demos_2)} demo files")
    demo_files.extend(demos_2)

print(f"\nTotal: {len(demo_files)} demo files to process")
print("="*80)

# Process all demos
extractor = EnhancedRoundFeatureExtractor()
all_features = []
successful = 0
failed = 0

for i, demo_path in enumerate(demo_files, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(demo_files)}] {demo_path.name}")
    print(f"{'='*60}")
    try:
        features_df = extractor.extract_from_demo(demo_path)
        
        if not features_df.empty:
            # Save individual match features
            output_path = OUTPUT_DIR / f"{demo_path.stem}_enhanced_features.csv"
            features_df.to_csv(output_path, index=False)
            print(f"✓ SUCCESS: Extracted {len(features_df)} samples")
            
            all_features.append(features_df)
            successful += 1
        else:
            print(f"✗ FAILED: No features extracted")
            failed += 1
    except Exception as e:
        print(f"✗ ERROR: {e}")
        failed += 1
    
    # Show running totals
    print(f"\nProgress: {successful} successful, {failed} failed")
    if all_features:
        total_samples = sum(len(df) for df in all_features)
        print(f"Total samples so far: {total_samples}")

# Combine all features
if all_features:
    print(f"\n{'='*80}")
    print("COMBINING ALL FEATURES")
    print("="*80)
    
    combined_df = pd.concat(all_features, ignore_index=True)
    
    # Save combined dataset
    combined_path = OUTPUT_DIR / "all_matches_enhanced_features.csv"
    combined_df.to_csv(combined_path, index=False)
    
    print(f"\nSuccessfully processed: {successful}/{len(demo_files)} demos")
    print(f"Failed: {failed}/{len(demo_files)} demos")
    print(f"\nTotal samples: {len(combined_df)}")
    print(f"Total rounds: {combined_df['round_num'].nunique()}")
    print(f"Total matches: {combined_df['match_id'].nunique()}")
    print(f"\nFeatures extracted:")
    for col in sorted(combined_df.columns):
        print(f"  - {col}")
    
    print(f"\nWinning team distribution:")
    print(combined_df['winning_team'].value_counts())
    
    print(f"\n{'='*80}")
    print(f"Combined dataset saved to: {combined_path}")
    print(f"{'='*80}")
else:
    print("\n✗ No features extracted from any demo!")
