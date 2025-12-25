#!/bin/bash
# Load API key from .env file (NEVER hardcode API keys)
if [ -f "/Users/keyonnesession/Desktop/NFL QUANT/.env" ]; then
    export $(grep -v '^#' "/Users/keyonnesession/Desktop/NFL QUANT/.env" | xargs)
fi

if [ -z "$ODDS_API_KEY" ]; then
    echo "ERROR: ODDS_API_KEY not set. Add it to .env file."
    exit 1
fi

cd "/Users/keyonnesession/Desktop/NFL QUANT"

# Run with automatic yes
echo "y" | .venv/bin/python scripts/fetch/fetch_comprehensive_odds.py 9
