#!/bin/bash
export ODDS_API_KEY=1fa38c2a5b8df1b50ad9be8887386f04
cd "/Users/keyonnesession/Desktop/NFL QUANT"

# Run with automatic yes
echo "y" | .venv/bin/python scripts/fetch/fetch_comprehensive_odds.py 9
