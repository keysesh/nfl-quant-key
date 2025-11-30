#!/usr/bin/env python3
"""
Auto-open dashboard if CSV files are newer than dashboard.
Checks if data has been updated and opens dashboard accordingly.
"""

import sys
from pathlib import Path
import webbrowser

def check_and_open_dashboard():
    """Check if dashboard needs regeneration and open it."""
    dashboard_path = Path("reports/simplified_dashboard.html")

    # Check which CSV files exist
    csv_paths = [
        Path("reports/unified_betting_recommendations.csv"),
        Path("reports/unified_betting_recommendations_v2.csv"),
        Path("reports/unified_betting_recommendations_v2_ranked.csv")
    ]

    # Find the most recent CSV file
    most_recent_csv = None
    most_recent_time = 0

    for csv_path in csv_paths:
        if csv_path.exists():
            mtime = csv_path.stat().st_mtime
            if mtime > most_recent_time:
                most_recent_time = mtime
                most_recent_csv = csv_path

    if not most_recent_csv:
        print("❌ No CSV recommendation files found!")
        print("   Run pipeline first: python scripts/predict/generate_current_week_recommendations.py")
        return False

    # Check if dashboard exists and is newer than CSV
    if dashboard_path.exists():
        dashboard_mtime = dashboard_path.stat().st_mtime
        if dashboard_mtime >= most_recent_time:
            print(f"✅ Dashboard is up-to-date (generated after {most_recent_csv.name})")
            print(f"🌐 Opening dashboard...")
            dashboard_url = dashboard_path.absolute().as_uri()
            webbrowser.open(dashboard_url)
            return True
        else:
            print(f"⚠️  Dashboard is outdated (CSV files are newer)")
            print(f"   Regenerating dashboard...")
    else:
        print(f"📊 Dashboard not found - generating...")

    # Regenerate dashboard
    import subprocess
    risk_mode = sys.argv[1] if len(sys.argv) > 1 else 'balanced'
    result = subprocess.run(
        [sys.executable, 'scripts/generate_table_dashboard.py', risk_mode, '--open'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print("❌ Failed to generate dashboard:")
        print(result.stderr)
        return False

if __name__ == "__main__":
    check_and_open_dashboard()





























