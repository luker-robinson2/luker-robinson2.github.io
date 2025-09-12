from awpy import Demo
import pandas as pd
import os
import zipfile
from pathlib import Path
import shutil
import subprocess
import warnings

# Optional RAR support (fallback)
try:
    import rarfile  # requires unrar/unar backend installed on the system
    RAR_AVAILABLE = True
except Exception:
    rarfile = None
    RAR_AVAILABLE = False

BASE_DIR = Path(__file__).parent
DEMOS_DIR = BASE_DIR / "demos"
EXTRACT_DIR = BASE_DIR / "demos_extracted"
OUTPUT_DIR = BASE_DIR / "hltv_data"
EXTRACT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def detect_demo_type(dem_path: Path) -> str:
    try:
        size = dem_path.stat().st_size
        if size < 1024:  # clearly invalid
            return "invalid"
        with open(dem_path, "rb") as f:
            head = f.read(16)
        if head.startswith(b"HL2DEMO"):
            return "csgo"
        # Common CS2 identifiers seen in the wild
        if head.startswith(b"PBDEMS2") or head.startswith(b"DEMS2"):
            return "cs2"
        return "unknown"
    except Exception:
        return "invalid"


def parse_and_save(dem_path: Path):
    dtype = detect_demo_type(dem_path)
    if dtype == "invalid":
        print(f"Skipping invalid/empty demo: {dem_path.name}")
        return
    
    print(f"Parsing {dem_path.name} (type={dtype})")
    
    if dtype == "cs2":
        print(f"WARNING: {dem_path.name} is a CS2 demo. awpy has limited CS2 support.")
        print("The parsing may not extract complete round data.")
    
    try:
        demo = Demo(str(dem_path))
        
        # Try to get basic demo info first
        try:
            header = demo.parse_header()
            print(f"Demo header parsed successfully: {header.get('map_name', 'unknown map')}")
        except Exception as e:
            print(f"Failed to parse header: {e}")
            header = {}
        
        # Attempt to parse the demo
        match = demo.parse()
        
        if match is None:
            print(f"WARNING: demo.parse() returned None for {dem_path.name}")
            print("This is a known issue with CS2 demos in awpy 2.0.2")
            
            # Create a minimal CSV with just header info
            rows = [{
                "round_num": None,
                "winning_team": None,
                "bomb_planted": None,
                "demo_type": dtype,
                "map_name": header.get('map_name', 'unknown'),
                "parsing_status": "failed_cs2_unsupported"
            }]
            df = pd.DataFrame(rows)
            stem = dem_path.stem
            out_csv = OUTPUT_DIR / f"{stem}_rounds.csv"
            df.to_csv(out_csv, index=False)
            print(f"Created placeholder CSV (parsing failed) -> {out_csv}")
            return
        
        # If we get here, parsing worked
        rounds = getattr(match, "rounds", [])
        print(f"Successfully extracted {len(rounds)} rounds")
        
        if not rounds:
            print("No rounds found in demo")
            # Create empty but valid CSV
            rows = [{
                "round_num": None,
                "winning_team": None,
                "bomb_planted": None,
                "demo_type": dtype,
                "map_name": header.get('map_name', 'unknown'),
                "parsing_status": "no_rounds_found"
            }]
        else:
            rows = []
            for i, r in enumerate(rounds):
                row = {
                    "round_num": getattr(r, "number", i + 1),
                    "winning_team": getattr(r, "winner", None),
                    "bomb_planted": getattr(r, "bomb_planted", None),
                    "demo_type": dtype,
                    "map_name": header.get('map_name', 'unknown'),
                    "parsing_status": "success"
                }
                rows.append(row)
                
        df = pd.DataFrame(rows)
        stem = dem_path.stem
        out_csv = OUTPUT_DIR / f"{stem}_rounds.csv"
        df.to_csv(out_csv, index=False)
        print(f"Parsed {len(rows)} rounds -> {out_csv}")
        
    except Exception as e:
        print(f"Failed to parse {dem_path.name}: {e}")
        # Create error CSV
        rows = [{
            "round_num": None,
            "winning_team": None,
            "bomb_planted": None,
            "demo_type": dtype,
            "map_name": "unknown",
            "parsing_status": f"error: {str(e)[:100]}"
        }]
        df = pd.DataFrame(rows)
        stem = dem_path.stem
        out_csv = OUTPUT_DIR / f"{stem}_rounds.csv"
        df.to_csv(out_csv, index=False)
        print(f"Created error CSV -> {out_csv}")


def extract_rar_cli(rar_path: Path, dest_dir: Path) -> bool:
    dest_dir.mkdir(exist_ok=True)
    unar = shutil.which("unar")
    if unar:
        try:
            result = subprocess.run([unar, "-force-overwrite", "-quiet", "-output-directory", str(dest_dir), str(rar_path)], capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to extract with unar: {e.stderr.strip() or e.stdout.strip()}")
    unrar = shutil.which("unrar")
    if unrar:
        try:
            result = subprocess.run([unrar, "x", "-o+", str(rar_path), str(dest_dir)], capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to extract with unrar: {e.stderr.strip() or e.stdout.strip()}")
    return False


print("=== CS Demo Parser ===")
print("Note: This script has limited support for CS2 demos due to awpy library limitations.")
print("CS2 demos may produce empty or incomplete results.\n")

for entry in DEMOS_DIR.iterdir():
    if entry.name.endswith(".part"):
        continue
    # ZIP
    is_zip = entry.suffix.lower() == ".zip" or zipfile.is_zipfile(entry)
    if is_zip:
        try:
            with zipfile.ZipFile(entry) as z:
                for member in z.namelist():
                    if member.lower().endswith(".dem"):
                        target_path = EXTRACT_DIR / Path(member).name
                        if not target_path.exists():
                            z.extract(member, path=EXTRACT_DIR)
                            extracted = EXTRACT_DIR / member
                            if extracted.exists() and extracted.is_file():
                                extracted.rename(target_path)
                        parse_and_save(target_path)
        except Exception as e:
            print(f"Failed to extract {entry.name}: {e}")
        continue
    # RAR -> prefer CLI tools (unar/unrar); fallback to rarfile if installed
    is_rar = entry.suffix.lower() == ".rar"
    if is_rar:
        ok = extract_rar_cli(entry, EXTRACT_DIR)
        if not ok and RAR_AVAILABLE:
            try:
                with rarfile.RarFile(entry) as rf:
                    for member in rf.infolist():
                        name = getattr(member, 'filename', getattr(member, 'filename', ''))
                        if name.lower().endswith(".dem"):
                            target_path = EXTRACT_DIR / Path(name).name
                            if not target_path.exists():
                                rf.extract(member, path=EXTRACT_DIR)
                                extracted = EXTRACT_DIR / name
                                if extracted.exists() and extracted.is_file():
                                    extracted.rename(target_path)
                            parse_and_save(target_path)
            except Exception as e:
                print(f"Failed to extract RAR {entry.name}: {e}")
        # After extraction, parse any new .dem files
        for dem in EXTRACT_DIR.glob("*.dem"):
            parse_and_save(dem)
        continue
    # Direct DEM
    if entry.suffix.lower() == ".dem":
        parse_and_save(entry)
    else:
        print(f"Skipping unsupported file: {entry.name}")

print("\n=== Parsing Complete ===")
print("Check the hltv_data folder for results.")
print("Note: CS2 demos may have limited data due to library constraints.")