#!/usr/bin/env python3
"""
Live Pipeline Monitor
Monitors the NFL QUANT pipeline progress in real-time
"""

import time
import sys
from pathlib import Path

def monitor_pipeline(log_file="/tmp/pipeline_week10.log", pid_file="/tmp/pipeline_pid.txt"):
    """Monitor pipeline progress live."""

    print("="*80)
    print("🔍 NFL QUANT PIPELINE MONITOR")
    print("="*80)
    print(f"Log file: {log_file}")
    print(f"PID file: {pid_file}")
    print("="*80)
    print("\nPress Ctrl+C to stop monitoring\n")

    log_path = Path(log_file)
    pid_path = Path(pid_file)

    # Check if process is running
    if pid_path.exists():
        pid = int(pid_path.read_text().strip())
        try:
            import os
            os.kill(pid, 0)  # Check if process exists
            print(f"✅ Pipeline running (PID: {pid})")
        except OSError:
            print(f"⚠️  Pipeline process not found (PID: {pid})")
    else:
        print("⚠️  PID file not found - pipeline may not be running")

    # Monitor log file
    if not log_path.exists():
        print(f"⏳ Waiting for log file to be created...")
        for i in range(10):
            time.sleep(1)
            if log_path.exists():
                break
        if not log_path.exists():
            print("❌ Log file not found. Pipeline may not have started.")
            return

    # Track last position
    last_pos = 0

    try:
        while True:
            if log_path.exists():
                with open(log_path, 'r') as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    if new_lines:
                        for line in new_lines:
                            # Filter for important messages
                            if any(keyword in line for keyword in [
                                'STEP', 'Complete', '✅', '❌', 'ERROR',
                                'Integrating', 'factors', 'Saved', 'Found',
                                'Processing', 'Generated', 'Creating'
                            ]):
                                print(line.rstrip())
                        last_pos = f.tell()
            else:
                print("⏳ Waiting for log file...")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print(f"Full log available at: {log_file}")

if __name__ == "__main__":
    monitor_pipeline()
