# Week 12 Missing Recommendations - Root Cause Analysis

**Date**: November 19, 2025 (continued)
**Analyst**: NFL QUANT System Investigation
**Status**: 🚨 **CRITICAL DATA ISSUE IDENTIFIED**

---

## Executive Summary

User identified TWO critical concerns:
1. **"all gamers should have values i feel like"** - Only 4 of 14 games have recommendations
2. **"lets do a test of if the atl falcons game is taking into account penix being injured"** - Injury integration validation

**FINDING**: Root cause identified - **INCOMPLETE Week 12 prediction data**

---

## Issue #1: Missing Recommendations for 10 Games

### Current State
- **Total recommendations**: 51 picks
- **Games with picks**: 4 out of 14 (29%)
- **Games without picks**: 10 out of 14 (71%)

| Game | Picks | Status |
|------|-------|--------|
| BUF @ HOU | 28 | ✅ Has picks |
| TB @ LA | 12 | ✅ Has picks |
| CAR @ SF | 7 | ✅ Has picks |
| IND @ KC | 4 | ✅ Has picks |
| **ATL @ NO** | **0** | ❌ **NO PICKS** |
| CLE @ LV | 0 | ❌ NO PICKS |
| JAX @ ARI | 0 | ❌ NO PICKS |
| MIN @ GB | 0 | ❌ NO PICKS |
| NE @ CIN | 0 | ❌ NO PICKS |
| NYG @ DET | 0 | ❌ NO PICKS |
| NYJ @ BAL | 0 | ❌ NO PICKS |
| PHI @ DAL | 0 | ❌ NO PICKS |
| PIT @ CHI | 0 | ❌ NO PICKS |
| SEA @ TEN | 0 | ❌ NO PICKS |

---

## Issue #2: Michael Penix Jr. Injury Not Accounted For

### User's Concern
> "lets do a test of if the atl falcons game is taking itno account penix being injured and how does it treat it?"

### Investigation Results

**❌ PENIX NOT IN PREDICTIONS AT ALL**

**ATL Falcons Predictions**:
- Total ATL players: **1 out of expected ~35-40**
- Players found: Bijan Robinson (RB only)
- **Missing**: ALL QBs, TEs, most WRs

**NO Saints Predictions**:
- Total NO players: **1 out of expected ~35-40**
- Players found: Chris Olave (WR only)
- **Missing**: ALL QBs, TEs, most WRs

---

## Root Cause Analysis

### 🚨 **CRITICAL FINDING: Incomplete Prediction Data**

```
Expected: ~500+ players
Actual: 171 players (66% MISSING)
```

**Missing Player Distribution**:
- ATL: 1 player (should be ~35-40)
- NO: 1 player (should be ~35-40)
- Other 12 teams: Likely similar deficiencies

### Why This Explains Both Issues

1. **No ATL @ NO recommendations** → Because ATL/NO players missing from predictions
2. **Penix injury not accounted for** → Because NO ATL QBs exist in predictions to apply injury logic to
3. **10 games without recommendations** → Because most players missing from those teams

---

## Validation: Week 12 Prediction File Analysis

**File**: `data/model_predictions_week12.csv`

**Findings**:
```
Total players: 171 (should be ~500+)

Sample teams with data deficiency:
- ATL: 1 player (Bijan Robinson RB, snap_share=0.779)
- NO: 1 player (Chris Olave WR, snap_share=0.835)

QBs in entire file:
- ❌ NO ATL QBs
- ❌ NO NO QBs
- ❌ Likely missing QBs from other teams as well
```

**Expected vs Actual**:
| Metric | Expected | Actual | Gap |
|--------|----------|--------|-----|
| Total players | ~500 | 171 | -66% |
| ATL players | ~35-40 | 1 | -97% |
| NO players | ~35-40 | 1 | -97% |
| ATL QBs | 1-2 | 0 | -100% |

---

## Why Recommendations Are Concentrated

The 55% concentration in BUF@HOU (28 picks) is **VALID** for that specific game, BUT it's artificially inflated as a percentage because:

1. **Denominator problem**: Total picks = 51 (should be ~300+)
2. **Missing numerators**: Other games have 0 picks because their players weren't predicted
3. **BUF@HOU appears concentrated** only because other games are missing data

**True distribution if all players were predicted**:
- BUF@HOU: 28 picks / ~300 total = ~9% (NORMAL)
- ATL@NO: Would likely have 15-25 picks if data existed
- Other games: Would collectively add ~200+ picks

---

## Impact Assessment

### ❌ **Production System NOT Ready**

**Immediate Impacts**:
1. **Cannot evaluate ATL game** - No QBs, TEs, most skill position players
2. **Cannot test Penix injury logic** - No ATL QB data exists to apply injury adjustments to
3. **66% of expected picks missing** - Severe underutilization of betting opportunities
4. **Misleading concentration analysis** - BUF@HOU looks concentrated due to missing data from other games

**Risk Exposure**:
- Missing ~250+ potential value bets across 10 games
- No ability to validate injury integration for ATL
- Dashboard showing incomplete game coverage

---

## Recommended Actions

### 🚨 **IMMEDIATE** (Before Week 12 Games)

1. **Regenerate Week 12 Predictions with Full Player Set**
   ```bash
   cd "/Users/keyonnesession/Desktop/NFL QUANT"
   rm -f data/model_predictions_week12.csv
   .venv/bin/python scripts/predict/generate_model_predictions.py 12
   ```
   - **Expected output**: ~500+ players (not 171)
   - **Validation**: Check that ALL 14 games have 30-40 players each

2. **Verify ATL QBs Included**
   ```python
   # After regeneration, verify:
   atl_qbs = preds[(preds['team'] == 'ATL') & (preds['position'] == 'QB')]
   # Should show: Kirk Cousins, potentially Michael Penix Jr.
   ```

3. **Regenerate Recommendations**
   ```bash
   .venv/bin/python scripts/predict/generate_unified_recommendations_v3.py --week 12
   ```
   - **Expected output**: ~300+ recommendations (not 51)
   - **Expected games with picks**: 12-14 out of 14 (not 4)

4. **Regenerate Dashboard**
   ```bash
   .venv/bin/python scripts/dashboard/generate_elite_picks_dashboard_v2.py
   ```

### 📊 **VALIDATION CHECKS**

After regeneration, verify:

1. **Player Count by Team**
   ```python
   preds.groupby('team').size().sort_values()
   # All teams should have 30-45 players
   ```

2. **ATL QB Check**
   ```python
   atl_qbs = preds[(preds['team'] == 'ATL') & (preds['position'] == 'QB')]
   print(atl_qbs[['player_name', 'passing_yards_mean', 'snap_share']])
   # Should show Kirk Cousins (primary QB)
   ```

3. **Injury Data Loaded**
   ```python
   # Check injuries file date
   ls -lh data/injuries/injuries_latest.csv
   # Should be dated November 16-19, 2025
   ```

4. **Game Coverage**
   ```python
   recs = pd.read_csv('reports/CURRENT_WEEK_RECOMMENDATIONS.csv')
   games_with_picks = recs['game'].nunique()
   print(f"Games with picks: {games_with_picks}/14")
   # Should show 12-14/14 (not 4/14)
   ```

---

## ✅ ROOT CAUSE IDENTIFIED

**CONFIRMED: Script Only Predicts Players With Betting Odds**

From prediction log (`/tmp/week12_predictions_v2.log`):
```
INFO:__main__:   ✅ Loaded 1778 players from NFLverse (2025 season)
4. Loading players from odds data...
INFO:__main__:   ✅ Found 184 unique players
INFO:__main__:   Processing 184 players (this may take a few minutes)...
```

**The Problem**:
1. Script loads **1,778 players from NFLverse** (all active players)
2. Then **overrides** this with **184 players from odds file**
3. Only generates predictions for those 184 players with betting markets
4. After processing/filtering, only 171 make it to final output

**Why This Design Is Wrong**:
- ❌ Sportsbooks don't offer odds on ALL players (no backup QBs, depth WRs/TEs)
- ❌ Kirk Cousins, Michael Penix Jr. missing because no Week 12 odds yet
- ❌ Most ATL/NO players missing (only 1 each found with odds)
- ❌ 10 games have no recommendations because key players lack odds
- ❌ Prediction layer is coupled to betting markets (should be independent)

**Correct Architecture**:
```
Step 1: Generate predictions for ALL ~500 active players
Step 2: Filter predictions to only players with odds during recommendation generation
```

**Current (Wrong) Architecture**:
```
Step 1: Load only ~184 players with odds
Step 2: Generate predictions only for those 184
Step 3: Recommendation generation finds nothing for players without odds
```

---

## Expected Outcomes After Fix

### Before Fix (Current State)
```
Total players: 171
Total recommendations: 51
Games with picks: 4/14 (29%)
ATL @ NO picks: 0
ATL QBs: 0
```

### After Fix (Expected State)
```
Total players: ~500+
Total recommendations: ~300+
Games with picks: 12-14/14 (86-100%)
ATL @ NO picks: 15-25
ATL QBs: 1-2 (Kirk Cousins primary, Penix backup)
```

### Penix Injury Test (After Fix)

**What should happen**:
1. Kirk Cousins should be primary QB (snap_share ~0.96)
2. Michael Penix Jr. should be backup or inactive (snap_share ~0.04 or 0.00)
3. Injury status should be reflected in:
   - `injury_qb_status` column
   - Adjusted projections for ATL pass catchers
   - Game script adjustments

**How to validate**:
```python
# Check ATL QB injury status
atl_qbs = preds[(preds['team'] == 'ATL') & (preds['position'] == 'QB')]
print(atl_qbs[['player_name', 'snap_share', 'injury_qb_status']])

# Check ATL pass catchers have injury adjustment
atl_wrs = preds[(preds['team'] == 'ATL') & (preds['position'].isin(['WR', 'TE']))]
print(atl_wrs[['player_name', 'receptions_mean', 'injury_qb_status']].head())
```

---

## Conclusion

**Primary Issue**: Incomplete Week 12 prediction data (171 players vs expected ~500+)

**Secondary Issue**: Cannot validate Penix injury handling because ATL QBs missing entirely

**Resolution**: Regenerate Week 12 predictions with full player set, then re-run recommendations and dashboard

**Status**: 🚨 **BLOCKING PRODUCTION DEPLOYMENT** until prediction data complete

---

## ✅ ARCHITECTURAL FIX IMPLEMENTED

**File**: `scripts/predict/generate_model_predictions.py`
**Lines Modified**: 1706-1716 (main call), 131-181 (new function)

**Change Summary**:
1. **Replaced**: `load_players_from_odds()` with `load_active_players_from_nflverse()`
2. **New Architecture**:
   - Step 1: Generate predictions for ALL ~500 active players from NFLverse
   - Step 2: Recommendation script filters to players with betting odds
3. **Expected Output**: ~500+ predictions (not 171)

**Code Changes**:

```python
# OLD (LINES 1706-1716) - REMOVED
logger.info("\n4. Loading players from odds data...")
odds_file = Path('data/nfl_player_props_draftkings.csv')
players_df = load_players_from_odds(odds_file, trailing_stats=trailing_stats)

# NEW (LINES 1706-1716) - IMPLEMENTED
logger.info("\n4. Loading active players from NFLverse roster data...")
players_df = load_active_players_from_nflverse(week, season, trailing_stats=trailing_stats)
```

**New Function** (lines 131-181):
```python
def load_active_players_from_nflverse(week: int, season: int, trailing_stats: Dict = None) -> pd.DataFrame:
    """
    Load ALL active players from NFLverse data (not just those with betting odds).

    This is the CORRECT architecture: Generate predictions for ALL active players,
    then filter to players with odds during recommendation generation.
    """
    # Extract all unique players from trailing stats
    # Filter to skill positions (QB, RB, WR, TE)
    # Only include players with weeks_played > 0 (active this season)
    ...
```

---

**Next Steps**:
1. ✅ Architectural fix implemented
2. 🔄 Regenerate Week 12 predictions with full player set
3. ⏳ Validate ATL QB injury handling after regeneration
4. ⏳ Confirm all 14 games have recommendations

