#!/usr/bin/env python3
"""
Bet Decision Support Tool

Explains the logic behind each betting recommendation:
- Why the bet was recommended
- Model projection vs market line
- Edge calculation breakdown
- Risk factors
- Confidence level

Usage:
    python scripts/analyze/bet_decision_support.py --week 10
    python scripts/analyze/bet_decision_support.py --bet "PHI @ GB Under 45.5"
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def explain_bet_logic(bet: pd.Series) -> str:
    """Generate detailed explanation of bet logic."""
    explanation = []
    
    player = bet.get('player', '')
    game = bet.get('game', '')
    market = bet.get('market', '')
    pick = bet.get('pick', '')
    line = bet.get('line', np.nan)
    odds = bet.get('odds', np.nan)
    model_prob = bet.get('model_prob', np.nan)
    model_projection = bet.get('model_projection', np.nan)
    edge_pct = bet.get('edge_pct', np.nan)
    roi_pct = bet.get('roi_pct', np.nan)
    
    # Header
    bet_name = f"{player} {pick} {line}" if player else f"{game} {pick} {line}"
    explanation.append("="*80)
    explanation.append(f"BET EXPLANATION: {bet_name}")
    explanation.append("="*80)
    explanation.append("")
    
    # Basic info
    explanation.append("📋 BET DETAILS:")
    explanation.append("-"*80)
    if player:
        explanation.append(f"  Player: {player}")
    if game:
        explanation.append(f"  Game: {game}")
    explanation.append(f"  Market: {market}")
    explanation.append(f"  Pick: {pick}")
    explanation.append(f"  Line: {line}")
    explanation.append(f"  Odds: {odds}")
    explanation.append("")
    
    # Model projection
    if pd.notna(model_projection):
        explanation.append("🎯 MODEL PROJECTION:")
        explanation.append("-"*80)
        explanation.append(f"  Model predicts: {model_projection:.1f}")
        explanation.append(f"  Market line: {line}")
        
        if 'under' in str(pick).lower():
            gap = model_projection - line
            explanation.append(f"  Gap: {gap:.1f} points/yards UNDER the line")
            if gap < -2:
                explanation.append(f"  ✅ Strong signal: Model predicts {abs(gap):.1f} points/yards below line")
            elif gap < 0:
                explanation.append(f"  ⚠️  Close call: Model predicts {abs(gap):.1f} points/yards below line")
            else:
                explanation.append(f"  ❌ Warning: Model predicts ABOVE the line!")
        elif 'over' in str(pick).lower():
            gap = model_projection - line
            explanation.append(f"  Gap: {gap:.1f} points/yards OVER the line")
            if gap > 2:
                explanation.append(f"  ✅ Strong signal: Model predicts {gap:.1f} points/yards above line")
            elif gap > 0:
                explanation.append(f"  ⚠️  Close call: Model predicts {gap:.1f} points/yards above line")
            else:
                explanation.append(f"  ❌ Warning: Model predicts BELOW the line!")
        explanation.append("")
    
    # Probability breakdown
    if pd.notna(model_prob):
        explanation.append("📊 PROBABILITY BREAKDOWN:")
        explanation.append("-"*80)
        
        # Market implied probability
        if pd.notna(odds):
            if odds > 0:
                market_prob = 100 / (odds + 100)
            else:
                market_prob = abs(odds) / (abs(odds) + 100)
            
            explanation.append(f"  Model Probability: {model_prob:.1%}")
            explanation.append(f"  Market Probability: {market_prob:.1%}")
            
            if pd.notna(edge_pct):
                explanation.append(f"  Edge: {edge_pct:.1f}%")
                explanation.append("")
                
                # Edge interpretation
                if edge_pct >= 20:
                    explanation.append(f"  ✅ EXCELLENT EDGE: {edge_pct:.1f}% edge indicates strong value")
                elif edge_pct >= 10:
                    explanation.append(f"  ✅ GOOD EDGE: {edge_pct:.1f}% edge indicates solid value")
                elif edge_pct >= 5:
                    explanation.append(f"  ⚠️  MODERATE EDGE: {edge_pct:.1f}% edge - acceptable but not strong")
                else:
                    explanation.append(f"  ❌ LOW EDGE: {edge_pct:.1f}% edge - consider skipping")
        explanation.append("")
    
    # ROI explanation
    if pd.notna(roi_pct):
        explanation.append("💰 EXPECTED RETURN:")
        explanation.append("-"*80)
        explanation.append(f"  Expected ROI: {roi_pct:.1f}%")
        
        if roi_pct >= 30:
            explanation.append(f"  ✅ Excellent expected return - high confidence bet")
        elif roi_pct >= 15:
            explanation.append(f"  ✅ Good expected return - solid bet")
        elif roi_pct >= 5:
            explanation.append(f"  ⚠️  Moderate expected return - acceptable bet")
        else:
            explanation.append(f"  ❌ Low expected return - consider skipping")
        explanation.append("")
    
    # Risk factors
    explanation.append("⚠️  RISK FACTORS:")
    explanation.append("-"*80)
    
    risk_factors = []
    
    # Close line
    if pd.notna(model_projection) and pd.notna(line):
        if 'under' in str(pick).lower():
            if model_projection > line - 2:
                risk_factors.append("Close to line - small margin for error")
        elif 'over' in str(pick).lower():
            if model_projection < line + 2:
                risk_factors.append("Close to line - small margin for error")
    
    # Low edge
    if pd.notna(edge_pct) and edge_pct < 5:
        risk_factors.append("Low edge - market may be efficient")
    
    # Low confidence
    if pd.notna(model_prob):
        if model_prob < 0.55:
            risk_factors.append("Low model confidence (<55%)")
        elif model_prob > 0.85:
            risk_factors.append("Very high confidence - check for overconfidence")
    
    if risk_factors:
        for factor in risk_factors:
            explanation.append(f"  • {factor}")
    else:
        explanation.append("  ✅ No major risk factors identified")
    
    explanation.append("")
    
    # Confidence level
    explanation.append("🎯 CONFIDENCE ASSESSMENT:")
    explanation.append("-"*80)
    
    confidence_score = 0
    
    # Edge contribution
    if pd.notna(edge_pct):
        if edge_pct >= 20:
            confidence_score += 3
        elif edge_pct >= 10:
            confidence_score += 2
        elif edge_pct >= 5:
            confidence_score += 1
    
    # Model projection gap
    if pd.notna(model_projection) and pd.notna(line):
        gap = abs(model_projection - line)
        if gap >= 5:
            confidence_score += 2
        elif gap >= 2:
            confidence_score += 1
    
    # Model probability
    if pd.notna(model_prob):
        if model_prob >= 0.70:
            confidence_score += 2
        elif model_prob >= 0.60:
            confidence_score += 1
    
    if confidence_score >= 6:
        explanation.append("  ✅ HIGH CONFIDENCE")
        explanation.append("     Strong edge, good projection gap, high model probability")
    elif confidence_score >= 4:
        explanation.append("  ⚠️  MEDIUM CONFIDENCE")
        explanation.append("     Moderate edge and projection gap")
    else:
        explanation.append("  ❌ LOW CONFIDENCE")
        explanation.append("     Weak edge or close projection - consider skipping")
    
    explanation.append("")
    
    # Summary
    explanation.append("📝 SUMMARY:")
    explanation.append("-"*80)
    
    summary_parts = []
    
    if pd.notna(model_projection) and pd.notna(line):
        if 'under' in str(pick).lower():
            summary_parts.append(f"Model predicts {model_projection:.1f} (under {line:.1f})")
        else:
            summary_parts.append(f"Model predicts {model_projection:.1f} (over {line:.1f})")
    
    if pd.notna(edge_pct):
        summary_parts.append(f"{edge_pct:.1f}% edge")
    
    if pd.notna(roi_pct):
        summary_parts.append(f"{roi_pct:.1f}% expected ROI")
    
    explanation.append("  " + " | ".join(summary_parts))
    explanation.append("")
    
    # Recommendation
    if confidence_score >= 6:
        explanation.append("  ✅ RECOMMENDATION: BET")
    elif confidence_score >= 4:
        explanation.append("  ⚠️  RECOMMENDATION: CONSIDER BETTING")
    else:
        explanation.append("  ❌ RECOMMENDATION: SKIP")
    
    explanation.append("")
    explanation.append("="*80)
    
    return "\n".join(explanation)


def explain_game_total_bet(bet: pd.Series) -> str:
    """Special explanation for game total bets."""
    explanation = []
    
    game = bet.get('game', '')
    pick = bet.get('pick', '')
    line = bet.get('line', np.nan)
    model_projection = bet.get('model_projection', np.nan)
    model_prob = bet.get('model_prob', np.nan)
    edge_pct = bet.get('edge_pct', np.nan)
    
    explanation.append("="*80)
    explanation.append(f"GAME TOTAL EXPLANATION: {game} {pick} {line}")
    explanation.append("="*80)
    explanation.append("")
    
    explanation.append("🎯 THE LOGIC:")
    explanation.append("-"*80)
    
    if pd.notna(model_projection) and pd.notna(line):
        if 'under' in str(pick).lower():
            gap = line - model_projection
            explanation.append(f"  Model projects: {model_projection:.1f} total points")
            explanation.append(f"  Market line: {line:.1f}")
            explanation.append(f"  Gap: {gap:.1f} points")
            explanation.append("")
            explanation.append(f"  ✅ The model thinks this game will score {model_projection:.1f} points,")
            explanation.append(f"     which is {gap:.1f} points BELOW the {line:.1f} line.")
            explanation.append("")
            explanation.append(f"  This means:")
            explanation.append(f"  • If the model is correct, the UNDER wins")
            explanation.append(f"  • The model gives this a {model_prob:.1%} chance of happening" if pd.notna(model_prob) else "")
            explanation.append(f"  • You have a {edge_pct:.1f}% edge over the market" if pd.notna(edge_pct) else "")
        else:
            gap = model_projection - line
            explanation.append(f"  Model projects: {model_projection:.1f} total points")
            explanation.append(f"  Market line: {line:.1f}")
            explanation.append(f"  Gap: {gap:.1f} points")
            explanation.append("")
            explanation.append(f"  ✅ The model thinks this game will score {model_projection:.1f} points,")
            explanation.append(f"     which is {gap:.1f} points ABOVE the {line:.1f} line.")
            explanation.append("")
            explanation.append(f"  This means:")
            explanation.append(f"  • If the model is correct, the OVER wins")
            explanation.append(f"  • The model gives this a {model_prob:.1%} chance of happening" if pd.notna(model_prob) else "")
            explanation.append(f"  • You have a {edge_pct:.1f}% edge over the market" if pd.notna(edge_pct) else "")
    
    explanation.append("")
    explanation.append("="*80)
    
    return "\n".join(explanation)


def main():
    parser = argparse.ArgumentParser(description='Explain bet logic')
    parser.add_argument('--week', type=int, help='Week number')
    parser.add_argument('--bet', type=str, help='Specific bet to explain (e.g., "PHI @ GB Under 45.5")')
    parser.add_argument('--all', action='store_true', help='Explain all recommendations')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Load recommendations
    if args.week:
        from scripts.validate.validate_framework_2025 import load_recommendations
        recommendations = load_recommendations(args.week)
    else:
        rec_file = Path('reports/CURRENT_WEEK_RECOMMENDATIONS.csv')
        if rec_file.exists():
            recommendations = pd.read_csv(rec_file)
        else:
            print("⚠️  No recommendations found")
            return
    
    if recommendations.empty:
        print("⚠️  No recommendations available")
        return
    
    explanations = []
    
    if args.bet:
        # Find specific bet - try multiple search strategies
        search_term = args.bet.lower()
        bet_match = recommendations[
            recommendations.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
        ]
        
        # Also try searching in game column
        if bet_match.empty and 'game' in recommendations.columns:
            bet_match = recommendations[
                recommendations['game'].astype(str).str.contains(search_term, case=False, na=False)
            ]
        
        # Try searching pick + line
        if bet_match.empty:
            parts = search_term.split()
            if len(parts) >= 2:
                pick_part = parts[-2] if len(parts) >= 2 else ''
                line_part = parts[-1] if len(parts) >= 1 else ''
                bet_match = recommendations[
                    (recommendations['pick'].astype(str).str.contains(pick_part, case=False, na=False)) &
                    (recommendations['line'].astype(str).str.contains(line_part, case=False, na=False))
                ]
        
        if bet_match.empty:
            print(f"⚠️  Bet '{args.bet}' not found")
            print(f"Available bets:")
            for idx, row in recommendations.iterrows():
                bet_desc = f"{row.get('player', '') or row.get('game', '')} {row.get('pick', '')} {row.get('line', '')}"
                print(f"  {bet_desc}")
            return
        
        bet = bet_match.iloc[0]
        
        # Check if game total
        if bet.get('market', '') == 'game_total':
            explanation = explain_game_total_bet(bet)
        else:
            explanation = explain_bet_logic(bet)
        
        print(explanation)
        explanations.append(explanation)
    
    elif args.all:
        # Explain all bets
        for idx, bet in recommendations.iterrows():
            if bet.get('market', '') == 'game_total':
                explanation = explain_game_total_bet(bet)
            else:
                explanation = explain_bet_logic(bet)
            
            explanations.append(explanation)
            print(explanation)
            print("\n" + "="*80 + "\n")
    
    else:
        # Explain top bets
        if 'edge_pct' in recommendations.columns:
            top_bets = recommendations.nlargest(5, 'edge_pct')
        else:
            top_bets = recommendations.head(5)
        
        for idx, bet in top_bets.iterrows():
            if bet.get('market', '') == 'game_total':
                explanation = explain_game_total_bet(bet)
            else:
                explanation = explain_bet_logic(bet)
            
            explanations.append(explanation)
            print(explanation)
            print("\n" + "="*80 + "\n")
    
    # Save if output specified
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n\n".join(explanations))
        print(f"\n✅ Explanations saved to: {output_file}")


if __name__ == '__main__':
    main()

