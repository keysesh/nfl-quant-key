#!/bin/bash
# GET_BETS.sh - One command to get your Week 9 betting recommendations
# Usage: ./GET_BETS.sh

cd "/Users/keyonnesession/Desktop/NFL QUANT"

echo "=================================="
echo "NFL QUANT - WEEK 9 BETS"
echo "=================================="
echo ""

# Check if recommendations exist
if [ ! -f "reports/unified_betting_recommendations_v2.csv" ]; then
    echo "❌ No recommendations found. Run: python scripts/run_week9_complete.py 9 balanced"
    exit 1
fi

# Count bets
TOTAL_BETS=$(wc -l < reports/unified_betting_recommendations_v2.csv)
TOTAL_BETS=$((TOTAL_BETS - 1))  # Subtract header

echo "📊 Total Recommendations: $TOTAL_BETS bets"
echo ""

# Show top 10 bets
echo "🎯 TOP 10 BETS (by edge):"
echo "-----------------------------------"
head -11 reports/unified_betting_recommendations_v2.csv | tail -10 | awk -F',' '{
    printf "%2d. %-30s | Edge: %5.1f%% | Bet: $%-4s\n",
    NR,
    $4,
    $8 * 100,
    $13
}'
echo ""

# Calculate total stake
TOTAL_STAKE=$(awk -F',' 'NR>1 {sum+=$13} END {print sum}' reports/unified_betting_recommendations_v2.csv)
echo "💰 Total Stake: \$$TOTAL_STAKE"
echo ""

# Open dashboard
echo "📈 Opening dashboard..."
open reports/enhanced_dashboard.html

echo ""
echo "✅ DONE"
echo "   • Full list: reports/unified_betting_recommendations_v2.csv"
echo "   • Dashboard: reports/enhanced_dashboard.html (opened)"
echo ""
echo "=================================="
