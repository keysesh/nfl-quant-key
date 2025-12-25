#!/usr/bin/env python3
"""
Backfill ALL weeks of 2024 season historical player props.
This will fetch props + outcomes from The Odds API for the entire 2024 season.
"""

import subprocess
import sys

print("=" * 80)
print("BACKFILLING 2024 NFL SEASON PLAYER PROPS")
print("=" * 80)
print()
print("This will fetch historical player prop lines and outcomes")
print("from The Odds API for weeks 1-18 of the 2024 season.")
print()
print("⚠️  NOTE: Historical API calls use your Odds API quota")
print("   Estimated: ~100-200 requests for full season")
print()

response = input("Continue? (y/n): ")
if response.lower() != 'y':
    print("Aborted.")
    sys.exit(0)

print()
print("Starting backfill...")
print()

# Backfill all 18 weeks of 2024 regular season
weeks = list(range(1, 19))  # Weeks 1-18

for week in weeks:
    print(f"\n{'='*80}")
    print(f"WEEK {week} / 18")
    print(f"{'='*80}\n")

    try:
        # Run backfill script
        cmd = [
            "python",
            "backfill_historical_player_props.py",
            "--season", "2024",
            "--weeks", str(week),
        ]

        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=300,  # 5 minute timeout per week
        )

        if result.returncode == 0:
            print(f"✅ Week {week} complete")
        else:
            print(f"⚠️  Week {week} failed with code {result.returncode}")

    except subprocess.TimeoutExpired:
        print(f"⚠️  Week {week} timed out")
    except Exception as e:
        print(f"❌ Week {week} error: {e}")

    print()

print()
print("=" * 80)
print("BACKFILL COMPLETE")
print("=" * 80)
print()
print("📊 Next steps:")
print("   1. Check data/historical/live_archive/player_props_archive.csv")
print("   2. Run build_prop_training_dataset.py to process into training format")
print("   3. Retrain calibrator with combined 2024+2025 data")
print()
