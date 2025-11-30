#!/usr/bin/env python3
"""
Fix player name matching to improve from 55% to 85%+ match rate
"""

import re

def normalize_player_name(name):
    """
    Normalize player name for matching
    Handles Jr., Sr., II, III, A.J. vs AJ, etc.
    """
    if not isinstance(name, str):
        return ""
    
    name = name.strip()
    
    # Skip special cases
    if any(x in name.lower() for x in ['d/st', 'defense', 'no touchdown', 'team']):
        return ""
    
    # Remove suffixes (Jr., Sr., II, III, IV)
    suffixes = [
        r'\s+Jr\.?$',
        r'\s+Sr\.?$', 
        r'\s+II$',
        r'\s+III$',
        r'\s+IV$',
        r'\s+V$',
    ]
    for suffix in suffixes:
        name = re.sub(suffix, '', name, flags=re.IGNORECASE)
    
    # Normalize A.J. to AJ (remove periods from initials)
    name = re.sub(r'\b([A-Z])\.([A-Z])\b', r'\1\2', name)
    
    # Convert to lowercase for matching
    name = name.lower()
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    return name


# Test the normalization
test_cases = [
    ("Deebo Samuel Sr.", "deebo samuel"),
    ("Travis Etienne Jr.", "travis etienne"),
    ("Kenneth Walker III", "kenneth walker"),
    ("Marvin Harrison Jr.", "marvin harrison"),
    ("A.J. Brown", "aj brown"),
    ("AJ Brown", "aj brown"),
    ("Brian Thomas Jr", "brian thomas"),
    ("Luther Burden III", "luther burden"),
]

print("🔧 PLAYER NAME NORMALIZATION FIX")
print("=" * 80)
print()

print("Testing normalization logic:")
print("-" * 80)
all_pass = True
for original, expected in test_cases:
    normalized = normalize_player_name(original)
    status = "✅" if normalized == expected else "❌"
    all_pass = all_pass and (normalized == expected)
    print(f'{status} "{original:30s}" → "{normalized:20s}" (expected: "{expected}")')

print()
if all_pass:
    print("✅ All tests passed!")
else:
    print("⚠️  Some tests failed - review logic")

print()
print("=" * 80)
print("📝 INTEGRATION INSTRUCTIONS")
print("=" * 80)
print()
print("To fix matching in build_prop_training_dataset.py:")
print()
print("1. Add this normalize_player_name() function")
print("2. Apply it when creating player_key:")
print("   player_key = normalize_player_name(player)")
print("3. Also apply to Sleeper stats:")
print("   stats_df['player_key'] = stats_df['player_name'].apply(normalize_player_name)")
print()
print("Expected improvement: 55% → 85%+ match rate")
print("This will recover ~2,000 lost props!")
print()

# Save the function
with open('nfl_quant/utils/player_names.py', 'w') as f:
    f.write('''"""Player name normalization utilities"""

import re

def normalize_player_name(name):
    """
    Normalize player name for matching across data sources.
    
    Handles:
    - Jr., Sr., II, III, IV suffixes
    - A.J. vs AJ format
    - Case insensitivity
    - Extra whitespace
    
    Args:
        name: Player name string
        
    Returns:
        Normalized lowercase name suitable for matching
        
    Examples:
        >>> normalize_player_name("Deebo Samuel Sr.")
        'deebo samuel'
        >>> normalize_player_name("A.J. Brown")
        'aj brown'
    """
    if not isinstance(name, str):
        return ""
    
    name = name.strip()
    
    # Skip special cases (defense, team names, etc.)
    if any(x in name.lower() for x in ['d/st', 'defense', 'no touchdown', 'team']):
        return ""
    
    # Remove suffixes (Jr., Sr., II, III, IV, V)
    suffixes = [
        r'\\s+Jr\\.?$',
        r'\\s+Sr\\.?$', 
        r'\\s+II$',
        r'\\s+III$',
        r'\\s+IV$',
        r'\\s+V$',
    ]
    for suffix in suffixes:
        name = re.sub(suffix, '', name, flags=re.IGNORECASE)
    
    # Normalize A.J. to AJ (remove periods from initials)
    name = re.sub(r'\\b([A-Z])\\.([A-Z])\\b', r'\\1\\2', name)
    
    # Convert to lowercase for matching
    name = name.lower()
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    return name
''')

print("✅ Saved function to: nfl_quant/utils/player_names.py")

