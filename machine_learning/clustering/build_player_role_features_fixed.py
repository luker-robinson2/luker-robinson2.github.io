#!/usr/bin/env python3
"""
Build per-player role features from Counter-Strike position timeseries files.

Outputs: player_role_features.csv with columns:
  - player_name, team_name, demo_file, map_name
  - avg_time_to_contact
  - avg_dist_at_contact
  - avg_early_speed_5s
  - avg_early_distance_10s

Assumptions (no kill/round logs available):
  - "Life start" is approximated when a player's health returns to 100 after being < 100,
    or the first record for that player in a file.
  - "Close contact" occurs when nearest opponent distance <= 300 units.

Usage:
  python build_player_role_features_fixed.py \
      --input_dir example_position_data \
      --output_csv player_role_features.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import glob
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def compute_nearest_opponent_distance(frame: pd.DataFrame) -> np.ndarray:
    """Compute nearest opponent distance for each row at a single timestamp."""
    distances = np.full(len(frame), np.nan)
    t_side = frame[frame["team_name"] == "TERRORIST"]
    ct_side = frame[frame["team_name"] != "TERRORIST"]

    if len(t_side) and len(ct_side):
        ct_tree = cKDTree(ct_side[["X", "Y", "Z"]].values)
        dists, _ = ct_tree.query(t_side[["X", "Y", "Z"]].values, k=1)
        # Use relative positions within the frame, not original DataFrame indices
        t_relative_indices = t_side.index.get_indexer(t_side.index)
        distances[t_relative_indices] = dists

    if len(ct_side) and len(t_side):
        t_tree = cKDTree(t_side[["X", "Y", "Z"]].values)
        dists, _ = t_tree.query(ct_side[["X", "Y", "Z"]].values, k=1)
        # Use relative positions within the frame, not original DataFrame indices
        ct_relative_indices = ct_side.index.get_indexer(ct_side.index)
        distances[ct_relative_indices] = dists

    return distances


def add_nearest_opponent_distances(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["timestamp", "team_name", "player_name"]).reset_index(drop=True)
    nearest = np.full(len(df), np.nan)
    for ts, grp in df.groupby("timestamp", sort=False):
        nearest[grp.index] = compute_nearest_opponent_distance(grp)
    df["nearest_opponent_dist"] = nearest
    return df


def mark_life_starts(df: pd.DataFrame) -> pd.DataFrame:
    flags = np.zeros(len(df), dtype=bool)
    for player, g in df.groupby("player_name", sort=False):
        idx = g.index
        health = g["health"].values
        prev_health = np.r_[np.nan, health[:-1]]
        starts = np.isnan(prev_health) | ((prev_health < 100) & (health == 100))
        flags[idx] = starts
    df["life_start"] = flags
    return df


def compute_role_features_for_file(csv_path: str, contact_threshold: float = 300.0) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows from {os.path.basename(csv_path)}")
        
        required = {
            "X", "Y", "Z", "health", "timestamp", "player_name", "team_name",
            "distance_moved", "speed"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        # annotate
        df = add_nearest_opponent_distances(df)
        df = mark_life_starts(df)

        records: List[Dict] = []
        meta_cols = [c for c in ["demo_file", "map_name"] if c in df.columns]

        for (player, team), g in df.groupby(["player_name", "team_name"], sort=False):
            g = g.sort_values("timestamp").reset_index(drop=True)
            
            # Find life starts
            starts = g.index[g["life_start"]].tolist()
            if not starts:
                print(f"    No life starts found for {player} ({team})")
                continue

            # Create life segments
            segments: List[pd.DataFrame] = []
            for i, start_idx in enumerate(starts):
                end_idx = starts[i + 1] if i + 1 < len(starts) else len(g)
                seg_df = g.iloc[start_idx:end_idx].copy()
                if len(seg_df) > 1:  # Need at least 2 points for meaningful analysis
                    segments.append(seg_df)

            if not segments:
                print(f"    No valid segments for {player} ({team})")
                continue

            times_to_contact: List[float] = []
            dist_at_contact: List[float] = []
            early_speed_vals: List[float] = []
            early_dist_vals: List[float] = []

            for seg_df in segments:
                if len(seg_df) == 0:
                    continue
                    
                t0 = seg_df["timestamp"].iloc[0]

                # Early movement features (first 5 and 10 seconds)
                w5 = seg_df[seg_df["timestamp"] <= t0 + 5]
                w10 = seg_df[seg_df["timestamp"] <= t0 + 10]
                
                early_speed_vals.append(w5["speed"].mean() if len(w5) > 0 else np.nan)
                early_dist_vals.append(
                    (w10["distance_moved"].iloc[-1] - w10["distance_moved"].iloc[0]) if len(w10) > 1 else 0.0
                )

                # Contact features
                hit = seg_df[seg_df["nearest_opponent_dist"] <= contact_threshold]
                if len(hit) > 0:
                    t_hit = hit["timestamp"].iloc[0]
                    times_to_contact.append(float(t_hit - t0))
                    dist_at_contact.append(float(hit["nearest_opponent_dist"].iloc[0]))
                else:
                    times_to_contact.append(np.nan)
                    dist_at_contact.append(np.nan)

            rec = {
                "player_name": player,
                "team_name": team,
                "avg_time_to_contact": float(np.nanmean(times_to_contact)) if len(times_to_contact) else np.nan,
                "avg_dist_at_contact": float(np.nanmean(dist_at_contact)) if len(dist_at_contact) else np.nan,
                "avg_early_speed_5s": float(np.nanmean(early_speed_vals)) if len(early_speed_vals) else np.nan,
                "avg_early_distance_10s": float(np.nanmean(early_dist_vals)) if len(early_dist_vals) else np.nan,
            }

            for m in meta_cols:
                rec[m] = g[m].iloc[0] if len(g) > 0 else None

            records.append(rec)
            print(f"    Processed {player} ({team}): {len(segments)} segments")

        out = pd.DataFrame(records)
        out["source_file"] = os.path.basename(csv_path)
        print(f"  Generated {len(out)} player records")
        return out
        
    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def build_dataset(input_dir: str, output_csv: str) -> None:
    paths = sorted(glob.glob(os.path.join(input_dir, "*_positions_timeseries.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_positions_timeseries.csv found under {input_dir}")

    print(f"Found {len(paths)} position timeseries files")
    
    frames = []
    for p in paths:
        print(f"Processing {os.path.basename(p)} ...")
        feat = compute_role_features_for_file(p)
        if not feat.empty:
            frames.append(feat)

    if not frames:
        raise RuntimeError("No features produced from any input file.")

    all_feats = pd.concat(frames, ignore_index=True)
    all_feats.to_csv(output_csv, index=False)
    print(f"Wrote {len(all_feats)} rows to {output_csv}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build player role features from position timeseries")
    parser.add_argument("--input_dir", default="example_position_data", help="Input directory with *_positions_timeseries.csv")
    parser.add_argument("--output_csv", default="player_role_features.csv", help="Output CSV path")
    args = parser.parse_args(argv)

    build_dataset(args.input_dir, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
