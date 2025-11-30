# Week 12 Root Cause Analysis - ODDS DATA MISMATCH

**Date**: November 19, 2025
**Status**: 🚨 **CRITICAL ISSUE IDENTIFIED**

---

## Executive Summary

**ROOT CAUSE**: Odds data contains **WRONG GAME MATCHUPS** for Week 12, causing 10 of 14 games to have zero recommendations.

**Impact**: Only 4 games have betting recommendations because only 4 games in the odds file match actual Week 12 matchups.

---

## The Problem

### Current State
- **391 predictions generated** (all 28 teams with players)
- **51 recommendations** (only 4 games)
- **10 games with 0 picks** (71% of games)

### Root Cause
The odds file (`data/nfl_player_props_draftkings.csv`) contains **incorrect Week 12 matchups**.

---

## Evidence

### Actual Week 12 Matchups (From NFL Schedule)
1. CHI @ DET
2. BUF @ HOU ✅ **MATCHES**
3. MIA @ GB
4. DEN @ LV
5. IND @ KC ✅ **MATCHES** (reversed)
6. ATL @ NO ✅ **MATCHES** (reversed)
7. TB @ LA ✅ **MATCHES** (reversed)
8. NE @ DAL
9. SEA @ ARI
10. CAR @ SF
11. TEN @ MIN
12. LAC @ BAL
13. PHI @ WAS
14. CLE @ PIT

### What Odds File Contains (14 games with wrong matchups)
1. Buffalo Bills @ Houston Texans ✅
2. New York Jets @ Baltimore Ravens ❌ (should be LAC @ BAL)
3. Pittsburgh Steelers @ Chicago Bears ❌ (should be CHI @ DET)
4. New England Patriots @ Cincinnati Bengals ❌ (not Week 12)
5. New York Giants @ Detroit Lions ❌ (not Week 12)
6. Minnesota Vikings @ Green Bay Packers ❌ (should be MIA @ GB)
7. Indianapolis Colts @ Kansas City Chiefs ✅ (reversed but matches)
8. Seattle Seahawks @ Tennessee Titans ❌ (should be TEN @ MIN)
9. Jacksonville Jaguars @ Arizona Cardinals ❌ (not Week 12)
10. Cleveland Browns @ Las Vegas Raiders ❌ (should be DEN @ LV)
11. Atlanta Falcons @ New Orleans Saints ✅ (reversed but matches)
12. Philadelphia Eagles @ Dallas Cowboys ❌ (should be NE @ DAL)
13. Tampa Bay Buccaneers @ Los Angeles Rams ✅ (reversed but matches)
14. Carolina Panthers @ San Francisco 49ers ✅ (reversed but matches)

**Only 4-5 games match** (BUF@HOU, IND@KC, ATL@NO, TB@LA, CAR@SF)

---

## Why This Explains Everything

### Observation 1: Only 4 Games Have Recommendations
**Explanation**: Only games with matching odds can generate recommendations. The recommendation script filters predictions to players with odds.

Games with picks:
- ✅ BUF @ HOU: 28 picks (matches odds: "Buffalo Bills @ Houston Texans")
- ✅ TB @ LA: 12 picks (matches odds: "Tampa Bay Buccaneers @ Los Angeles Rams")
- ✅ CAR @ SF: 7 picks (matches odds: "Carolina Panthers @ San Francisco 49ers")
- ✅ IND @ KC: 4 picks (matches odds: "Indianapolis Colts @ Kansas City Chiefs")

### Observation 2: 10 Games Have Zero Picks
**Explanation**: These games have predictions generated but no matching odds in the file.

Games without picks (no matching odds):
- ❌ CHI @ DET (odds has: PIT @ CHI)
- ❌ MIA @ GB (odds has: MIN @ GB)
- ❌ DEN @ LV (odds has: CLE @ LV)
- ❌ NE @ DAL (odds has: PHI @ DAL)
- ❌ SEA @ ARI (odds has: JAX @ ARI)
- ❌ TEN @ MIN (odds has: SEA @ TEN)
- ❌ LAC @ BAL (odds has: NYJ @ BAL)
- ❌ PHI @ WAS (no match)
- ❌ CLE @ PIT (odds has: PIT @ CHI)
- ❌ ATL @ NO (might match but reversed)

### Observation 3: Concentration Looks Suspicious (55% in one game)
**Explanation**: Concentration is VALID for the 4 games that DO have odds, but appears extreme because we're missing 10 games.

- **Actual**: 28 picks / 51 total = 55% in BUF@HOU
- **If all games had odds**: 28 picks / ~200+ total = ~14% (normal)

---

## Technical Details

### Odds File Analysis
```
File: data/nfl_player_props_draftkings.csv
Total props: 706
Games: 14 (but wrong matchups)

Market distribution:
  player_anytime_td: 169
  player_1st_td: 146
  player_reception_longest: 108
  player_rush_longest: 58
  player_pass_longest_completion: 44
```

### Missing Core Markets
The odds file primarily contains **touchdown props** and **longest reception/rush** markets, but is missing key markets like:
- `player_receptions` (only 22 props)
- `player_reception_yds` (only 22 props)
- `player_pass_yds` (only 18 props)
- `player_pass_tds` (only 18 props)
- `player_rush_yds` (missing entirely)
- `player_rush_attempts` (missing entirely)

This explains why even for games WITH matching odds, we only get limited recommendations.

---

## Resolution

### Immediate Action Required

1. **Re-fetch Week 12 Odds** with correct matchups:
   ```bash
   .venv/bin/python scripts/fetch/fetch_nfl_player_props.py --week 12
   ```

2. **Verify game matchups** match actual NFL Week 12 schedule

3. **Ensure core markets are fetched**:
   - player_receptions
   - player_reception_yds
   - player_rush_yds
   - player_rush_attempts
   - player_pass_yds
   - player_pass_tds

4. **Regenerate recommendations** after odds are fixed:
   ```bash
   .venv/bin/python scripts/predict/generate_unified_recommendations_v3.py --week 12
   ```

### Expected Outcome After Fix

**Before** (Current):
- 391 predictions
- 51 recommendations across 4 games

**After** (Expected):
- 391 predictions (same)
- ~150-250 recommendations across 12-14 games
- More balanced distribution (no single game with 55%)

---

## Lessons Learned

1. **Always validate odds data** against official NFL schedule
2. **Check for home/away correctness** (can be reversed)
3. **Verify market coverage** (not just TD props)
4. **Test with a known-good week** before production

---

## Status

🚨 **BLOCKED**: Week 12 production recommendations blocked until odds data is corrected

**Next Step**: Re-fetch odds with correct Week 12 game matchups
