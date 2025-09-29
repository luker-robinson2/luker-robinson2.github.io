#!/usr/bin/env python3
"""
Test Script for Interactive Map Visualizer
==========================================

This script tests the interactive visualizer components to ensure
everything is working correctly before launching the web interface.

Usage:
    python test_visualizer.py
"""

import sys
from pathlib import Path
import pandas as pd

def test_dependencies():
    """Test if all required dependencies are available"""
    print("Testing dependencies...")
    
    try:
        import dash
        print("✓ Dash available")
    except ImportError:
        print("✗ Dash not available - run: pip install dash")
        return False
    
    try:
        import plotly
        print("✓ Plotly available")
    except ImportError:
        print("✗ Plotly not available - run: pip install plotly")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow available")
    except ImportError:
        print("✗ PIL/Pillow not available - run: pip install pillow")
        return False
    
    return True

def test_position_data():
    """Test if position data files exist and are valid"""
    print("\nTesting position data...")
    
    data_dir = Path("example_position_data")
    if not data_dir.exists():
        print("✗ Position data directory not found")
        print("  Run: python extract_player_positions.py")
        return False
    
    csv_files = list(data_dir.glob("*_positions_timeseries.csv"))
    if not csv_files:
        print("✗ No position CSV files found")
        print("  Run: python extract_player_positions.py")
        return False
    
    print(f"✓ Found {len(csv_files)} position data files")
    
    # Test loading one file
    try:
        sample_file = csv_files[0]
        df = pd.read_csv(sample_file)
        
        required_columns = ['timestamp', 'player_name', 'X', 'Y']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"✗ Missing columns in {sample_file.name}: {missing_columns}")
            return False
        
        print(f"✓ Sample file {sample_file.name} has {len(df)} records")
        print(f"  Players: {df['player_name'].nunique()}")
        print(f"  Time range: {df['timestamp'].min():.1f}s - {df['timestamp'].max():.1f}s")
        
        if 'map_name' in df.columns:
            print(f"  Map: {df['map_name'].iloc[0]}")
        
    except Exception as e:
        print(f"✗ Error loading sample file: {e}")
        return False
    
    return True

def test_map_images():
    """Test if map images exist"""
    print("\nTesting map images...")
    
    map_images_dir = Path("map_images")
    if not map_images_dir.exists():
        print("✗ Map images directory not found")
        print("  Run: python download_map_images.py")
        return False
    
    png_files = list(map_images_dir.glob("*.png"))
    if not png_files:
        print("✗ No map image files found")
        print("  Run: python download_map_images.py")
        return False
    
    print(f"✓ Found {len(png_files)} map images")
    for img_file in png_files:
        print(f"  - {img_file.name}")
    
    return True

def test_visualizer_import():
    """Test if the visualizer can be imported and initialized"""
    print("\nTesting visualizer import...")
    
    try:
        from interactive_map_visualizer import MapVisualizer
        print("✓ MapVisualizer imported successfully")
        
        # Try to initialize
        visualizer = MapVisualizer()
        print("✓ MapVisualizer initialized")
        
        if not visualizer.position_data:
            print("✗ No position data loaded")
            return False
        
        print(f"✓ Loaded data for {len(visualizer.position_data)} demos")
        
        # Test demo options
        options = visualizer.get_demo_options()
        print(f"✓ Generated {len(options)} demo options")
        
        return True
        
    except Exception as e:
        print(f"✗ Error importing/initializing visualizer: {e}")
        return False

def test_coordinate_mapping():
    """Test coordinate mapping functionality"""
    print("\nTesting coordinate mapping...")
    
    try:
        from interactive_map_visualizer import MapVisualizer
        visualizer = MapVisualizer()
        
        if not visualizer.position_data:
            print("✗ No position data for coordinate testing")
            return False
        
        # Get sample data
        demo_name = list(visualizer.position_data.keys())[0]
        df = visualizer.position_data[demo_name].head()
        
        # Test normalization
        map_name = visualizer.get_map_name(demo_name)
        normalized_df = visualizer.normalize_coordinates(df, map_name)
        
        if 'norm_x' not in normalized_df.columns or 'norm_y' not in normalized_df.columns:
            print("✗ Coordinate normalization failed")
            return False
        
        print("✓ Coordinate normalization working")
        print(f"  Sample X: {df['X'].iloc[0]:.1f} → {normalized_df['norm_x'].iloc[0]:.1f}")
        print(f"  Sample Y: {df['Y'].iloc[0]:.1f} → {normalized_df['norm_y'].iloc[0]:.1f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing coordinate mapping: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Interactive Map Visualizer Test Suite ===\n")
    
    tests = [
        test_dependencies,
        test_position_data,
        test_map_images,
        test_visualizer_import,
        test_coordinate_mapping
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The visualizer is ready to run.")
        print("  Start it with: python interactive_map_visualizer.py")
        print("  Then open: http://localhost:8050")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
