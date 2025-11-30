#!/usr/bin/env python3
"""Diagnostic script to analyze 92% uniform probability issue"""

import pandas as pd
import numpy as np
from scipy.stats import norm

# Examples from actual data
examples = [
    {'name': 'Amon-Ra St. Brown', 'model': 4.68946, 'line': 6.5, 'pick': 'Under'},
    {'name': 'De\'Von Achane', 'model': 17.98214942486024, 'line': 32.5, 'pick': 'Under'},
    {'name': 'Chuba Hubbard', 'model': 24.08242386468187, 'line': 7.5, 'pick': 'Over'},
    {'name': 'Bo Nix', 'model': 12.433185350809982, 'line': 20.5, 'pick': 'Under'},
    {'name': 'Evan Engram', 'model': 55.15923269186957, 'line': 27.5, 'pick': 'Over'},
]

print('='*80)
print('ROOT CAUSE ANALYSIS: 92% Uniform Probability Issue')
print('='*80)
print()
print('The problem: All probabilities are clamped between 52% and 92%')
print('When model projections are far from lines, raw probabilities exceed 92%')
print('These all get clipped to exactly 92%\n')
print('-'*80)

for ex in examples:
    # Same calculation as in generate_current_week_recommendations.py
    std = max(ex['model'] * 0.25, ex['model'] * 0.15)
    z_score = (ex['line'] - ex['model']) / std

    if ex['pick'] == 'Over':
        prob_raw = 1.0 - norm.cdf(z_score)
    else:
        prob_raw = norm.cdf(z_score)

    prob_clipped = np.clip(prob_raw, 0.52, 0.92)
    is_clipped = prob_raw != prob_clipped

    print(f"{ex['name']:30} {ex['pick']:5} {ex['line']:5.1f}")
    print(f"  Model Projection: {ex['model']:7.2f}")
    print(f"  Estimated Std:    {std:7.2f} ({std/ex['model']*100:.1f}% of mean)")
    print(f"  Z-Score:           {z_score:7.2f}")
    print(f"  Raw Probability:   {prob_raw:7.1%} {'→ CLIPPED TO' if is_clipped else ''}")
    if is_clipped:
        print(f"  Clipped Probability: {prob_clipped:7.1%} [LOSING INFORMATION]")
    print()

print('='*80)
print('SOLUTIONS:')
print('='*80)
print('1. Use calibration instead of hard clamping')
print('2. Increase std estimation (currently 15-25% may be too small)')
print('3. Remove or raise the 92% cap (e.g., to 95-97%)')
print('4. Use logit transformation for extreme probabilities')
print('5. Apply shrinkage toward market probability instead of fixed bounds')
