#!/usr/bin/env python3
"""
Add clear 'framework_prediction' column to recommendations.

This column shows what value our framework is actually predicting,
making it easy to compare against the line.

Example:
- Line: Under 209.5 pass yards
- Framework Prediction: 104.8 yards
- Interpretation: We predict 104.8, well under the line of 209.5
"""

import pandas as pd
import re
from pathlib import Path


def add_framework_prediction_column(input_csv: str, output_csv: str = None):
    """
    Add framework_prediction column to make model values clearer.

    Args:
        input_csv: Path to recommendations CSV
        output_csv: Optional output path (defaults to overwrite input)
    """

    print("="*80)
    print("ADDING FRAMEWORK PREDICTION COLUMN")
    print("="*80)
    print()

    # Load recommendations
    df = pd.read_csv(input_csv)
    print(f"✅ Loaded {len(df)} recommendations")

    # Create new column
    predictions = []

    for idx, row in df.iterrows():
        bet_type = row['bet_type']
        pick = str(row['pick'])
        model_value = str(row.get('model_value', ''))
        market = str(row.get('market', ''))

        if bet_type == 'Player Prop':
            # Extract predicted value from model_value
            # Format is like "16.9" or "104.8"
            try:
                predicted_val = float(model_value)

                # Determine the stat type from market
                if 'pass_yds' in market or 'passing' in market.lower():
                    stat = 'pass yards'
                elif 'rush_yds' in market or 'rushing' in market.lower():
                    stat = 'rush yards'
                elif 'reception_yds' in market or 'receiving' in market.lower():
                    stat = 'rec yards'
                elif 'receptions' in market:
                    stat = 'receptions'
                elif 'pass_tds' in market:
                    stat = 'pass TDs'
                elif 'rush_tds' in market:
                    stat = 'rush TDs'
                elif 'rec_tds' in market:
                    stat = 'rec TDs'
                else:
                    stat = market.replace('player_', '').replace('_', ' ')

                # Extract line from pick
                line_match = re.search(r'(\d+\.?\d*)', pick)
                if line_match:
                    line = float(line_match.group(1))
                    direction = "Over" if "Over" in pick else "Under"

                    diff = abs(predicted_val - line)

                    if direction == "Under" and predicted_val < line:
                        prediction = f"Predict {predicted_val:.1f} {stat} ({diff:.1f} under line)"
                    elif direction == "Over" and predicted_val > line:
                        prediction = f"Predict {predicted_val:.1f} {stat} ({diff:.1f} over line)"
                    else:
                        # Edge case: betting opposite of prediction?
                        prediction = f"Predict {predicted_val:.1f} {stat} (line: {line})"
                else:
                    prediction = f"Predict {predicted_val:.1f} {stat}"

            except (ValueError, TypeError):
                prediction = model_value

        elif bet_type == 'Game Line':
            # Extract from model_value format: "17-17 (Model Total: 34.0)"
            if 'total' in market.lower():
                total_match = re.search(r'Model Total:\s*(\d+\.?\d*)', model_value)
                if total_match:
                    predicted_total = float(total_match.group(1))

                    # Extract line
                    line_match = re.search(r'(\d+\.?\d*)', pick)
                    if line_match:
                        line = float(line_match.group(1))
                        direction = "Over" if "Over" in pick else "Under"
                        diff = abs(predicted_total - line)

                        if direction == "Under" and predicted_total < line:
                            prediction = f"Predict {predicted_total:.1f} total ({diff:.1f} under)"
                        elif direction == "Over" and predicted_total > line:
                            prediction = f"Predict {predicted_total:.1f} total ({diff:.1f} over)"
                        else:
                            prediction = f"Predict {predicted_total:.1f} total (line: {line})"
                    else:
                        prediction = f"Predict {predicted_total:.1f} total"
                else:
                    # Try extracting scores
                    score_match = re.search(r'(\d+)-(\d+)', model_value)
                    if score_match:
                        away = int(score_match.group(1))
                        home = int(score_match.group(2))
                        total = away + home
                        prediction = f"Predict {total} total ({away}-{home})"
                    else:
                        prediction = model_value

            elif 'spread' in market.lower():
                spread_match = re.search(r'Model Spread:\s*([\+\-]?\d+\.?\d*)', model_value)
                if spread_match:
                    predicted_spread = float(spread_match.group(1))

                    # Extract line
                    line_match = re.search(r'([\+\-]\d+\.?\d*)', pick)
                    if line_match:
                        line_spread = float(line_match.group(1))
                        prediction = f"Predict spread {predicted_spread:+.1f} (line: {line_spread:+.1f})"
                    else:
                        prediction = f"Predict spread {predicted_spread:+.1f}"
                else:
                    # Try extracting scores
                    score_match = re.search(r'(\d+)-(\d+)', model_value)
                    if score_match:
                        away = int(score_match.group(1))
                        home = int(score_match.group(2))
                        spread = home - away
                        prediction = f"Predict spread {spread:+.1f} ({away}-{home})"
                    else:
                        prediction = model_value
            else:
                prediction = model_value
        else:
            prediction = model_value

        predictions.append(prediction)

    # Add column after 'model_value'
    model_value_idx = df.columns.get_loc('model_value')
    df.insert(model_value_idx + 1, 'framework_prediction', predictions)

    print(f"\n✅ Added 'framework_prediction' column")
    print(f"\nSample predictions:")
    for i, pred in enumerate(predictions[:5]):
        print(f"  {i+1}. {pred}")

    # Save
    output_path = output_csv or input_csv
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")

    return df


if __name__ == "__main__":
    input_file = "reports/unified_betting_recommendations_v2_ranked.csv"

    if Path(input_file).exists():
        add_framework_prediction_column(input_file)
    else:
        print(f"❌ File not found: {input_file}")
