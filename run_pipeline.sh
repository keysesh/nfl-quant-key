#!/bin/bash
export ODDS_API_KEY=1fa38c2a5b8df1b50ad9be8887386f04
export PYTHONPATH="/Users/keyonnesession/Desktop/NFL QUANT:$PYTHONPATH"

cd "/Users/keyonnesession/Desktop/NFL QUANT"

echo "Starting Week 9 pipeline with comprehensive odds data..."
.venv/bin/python scripts/run_week9_complete.py 9 balanced
