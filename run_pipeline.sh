#!/bin/bash
# Load API key from .env file (NEVER hardcode API keys)
if [ -f "/Users/keyonnesession/Desktop/NFL QUANT/.env" ]; then
    export $(grep -v '^#' "/Users/keyonnesession/Desktop/NFL QUANT/.env" | xargs)
fi

if [ -z "$ODDS_API_KEY" ]; then
    echo "ERROR: ODDS_API_KEY not set. Add it to .env file."
    exit 1
fi

export PYTHONPATH="/Users/keyonnesession/Desktop/NFL QUANT:$PYTHONPATH"

cd "/Users/keyonnesession/Desktop/NFL QUANT"

echo "Starting Week 9 pipeline with comprehensive odds data..."
.venv/bin/python scripts/run_week9_complete.py 9 balanced
