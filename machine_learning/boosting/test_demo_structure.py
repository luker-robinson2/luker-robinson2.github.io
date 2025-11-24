#!/usr/bin/env python3
"""Quick test to see what data is available in demo files"""

from demoparser2 import DemoParser
from pathlib import Path
import pandas as pd

demo_path = Path("/Users/lukerobinson/Dropbox/school/5612/Project/luker-robinson2.github.io/machine_learning/demos_extracted/2385025_99903/semperfi-vs-underground-m1-nuke.dem")

print(f"Testing: {demo_path.name}\n")

parser = DemoParser(str(demo_path))

# Check header
header = parser.parse_header()
print("HEADER:")
for key, value in header.items():
    print(f"  {key}: {value}")

# Check round events
print("\n" + "="*80)
print("ROUND START EVENTS:")
round_starts = parser.parse_event("round_start")
if isinstance(round_starts, list):
    round_starts = pd.DataFrame(round_starts)
print(f"Columns: {round_starts.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(round_starts.head())

print("\n" + "="*80)
print("ROUND END EVENTS:")
round_ends = parser.parse_event("round_end")
if isinstance(round_ends, list):
    round_ends = pd.DataFrame(round_ends)
print(f"Columns: {round_ends.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(round_ends.head())

print("\n" + "="*80)
print("TESTING TICK PARSING:")
print("Trying to parse a specific tick range...")
try:
    # Try parsing ticks from 1000 to 10000
    tick_data = parser.parse_ticks(["team_name", "is_alive"], ticks=list(range(1000, 10000, 1000)))
    if isinstance(tick_data, list):
        tick_data = pd.DataFrame(tick_data)
    print(f"Successfully parsed {len(tick_data)} rows")
    print(f"Columns: {tick_data.columns.tolist()}")
    if not tick_data.empty:
        print(f"\nSample data:")
        print(tick_data.head(10))
except Exception as e:
    print(f"Error: {e}")

