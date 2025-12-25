# NFLverse Data Dictionary

**Generated:** 2025-12-14 17:56:33
**Source:** [nflreadr documentation](https://nflreadr.nflverse.com/)

---

## Table of Contents

1. [Summary Statistics](#summary-statistics)
2. [Common Fields Across Datasets](#common-fields-across-datasets)
3. [Data Dictionaries](#data-dictionaries)
   - [Play By Play](#play-by-play)
   - [Player Stats](#player-stats)
   - [Players](#players)
   - [Rosters](#rosters)
   - [Schedules](#schedules)
   - [Nextgen Stats](#nextgen-stats)
   - [Pfr Passing](#pfr-passing)
   - [Snap Counts](#snap-counts)
   - [Team Stats](#team-stats)
   - [Combine](#combine)
   - [Contracts](#contracts)
   - [Depth Charts](#depth-charts)
   - [Draft Picks](#draft-picks)
   - [Espn Qbr](#espn-qbr)
   - [Injuries](#injuries)
   - [Participation](#participation)
   - [Ftn Charting](#ftn-charting)
   - [Ff Playerids](#ff-playerids)
   - [Ff Rankings](#ff-rankings)
   - [Trades](#trades)

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Fields (including duplicates) | 700 |
| Unique Field Names | 542 |
| Number of Datasets | 20 |
| Fields in Multiple Datasets | 86 |

### Data Type Distribution

| Data Type | Count |
|-----------|-------|
| numeric | 411 |
| character | 256 |
| logical | 18 |
| integer | 10 |
| date | 2 |
| double | 1 |
| POSIXct | 1 |
| Date | 1 |

---

## Common Fields Across Datasets

These fields appear in multiple datasets and can be used for joining:

| Field Name | Appears In |
|------------|------------|
| `season` | play_by_play, player_stats, rosters, schedules, nextgen_stats, pfr_passing, snap_counts, team_stats, combine, draft_picks, espn_qbr, injuries, ftn_charting, trades |
| `team` | player_stats, rosters, pfr_passing, snap_counts, team_stats, contracts, depth_charts, draft_picks, espn_qbr, injuries, ff_playerids, ff_rankings |
| `week` | play_by_play, player_stats, rosters, schedules, nextgen_stats, snap_counts, team_stats, injuries, ftn_charting |
| `position` | player_stats, players, rosters, snap_counts, contracts, draft_picks, injuries, ff_playerids |
| `gsis_id` | players, rosters, depth_charts, draft_picks, injuries, ff_playerids |
| `pfr_id` | players, rosters, pfr_passing, combine, ff_playerids, trades |
| `season_type` | play_by_play, player_stats, nextgen_stats, team_stats, espn_qbr, injuries |
| `espn_id` | players, rosters, depth_charts, ff_playerids |
| `player` | pfr_passing, snap_counts, contracts, ff_rankings |
| `receptions` | player_stats, nextgen_stats, team_stats, draft_picks |
| `attempts` | player_stats, nextgen_stats, team_stats |
| `college` | rosters, draft_picks, ff_playerids |
| `completions` | player_stats, nextgen_stats, team_stats |
| `draft_round` | players, combine, ff_playerids |
| `draft_year` | players, combine, ff_playerids |
| `first_name` | players, rosters, injuries |
| `game_id` | play_by_play, schedules, snap_counts |
| `game_type` | rosters, schedules, snap_counts |
| `height` | players, rosters, ff_playerids |
| `last_name` | players, rosters, injuries |
| `old_game_id` | play_by_play, schedules, participation |
| `passing_yards` | play_by_play, player_stats, team_stats |
| `pff_id` | players, rosters, ff_playerids |
| `player_name` | player_stats, combine, depth_charts |
| `receiving_yards` | play_by_play, player_stats, team_stats |
| `rushing_yards` | play_by_play, player_stats, team_stats |
| `targets` | player_stats, nextgen_stats, team_stats |
| `weight` | players, rosters, ff_playerids |
| `yahoo_id` | rosters, ff_playerids, ff_rankings |
| `age` | draft_picks, ff_playerids |
| `away_coach` | play_by_play, schedules |
| `away_team` | play_by_play, schedules |
| `birth_date` | players, rosters |
| `carries` | player_stats, team_stats |
| `cbs_id` | ff_playerids, ff_rankings |
| `def_sacks` | team_stats, draft_picks |
| `div_game` | play_by_play, schedules |
| `draft_ovr` | combine, ff_playerids |
| `draft_pick` | players, ff_playerids |
| `draft_team` | players, combine |
| `fantasy_data_id` | rosters, ff_playerids |
| `full_name` | rosters, injuries |
| `headshot_url` | player_stats, rosters |
| `home_coach` | play_by_play, schedules |
| `home_team` | play_by_play, schedules |
| `id` | play_by_play, ff_rankings |
| `jersey_number` | players, rosters |
| `name` | play_by_play, ff_playerids |
| `nfl_id` | players, ff_playerids |
| `nflverse_game_id` | participation, ftn_charting |
| `ngs_position` | players, rosters |
| `opponent_team` | player_stats, team_stats |
| `otc_id` | players, contracts |
| `pass` | play_by_play, espn_qbr |
| `pass_attempts` | pfr_passing, draft_picks |
| `pass_yards` | nextgen_stats, draft_picks |
| `passing_cpoe` | player_stats, team_stats |
| `passing_epa` | player_stats, team_stats |
| `passing_interceptions` | player_stats, team_stats |
| `passing_tds` | player_stats, team_stats |
| `penalty` | play_by_play, espn_qbr |
| `pfr_player_id` | snap_counts, draft_picks |
| `play_id` | play_by_play, participation |
| `player_display_name` | player_stats, nextgen_stats |
| `player_id` | player_stats, espn_qbr |
| `pos` | combine, ff_rankings |
| `position_group` | player_stats, players |
| `receiving_epa` | player_stats, team_stats |
| `receiving_tds` | player_stats, team_stats |
| `roof` | play_by_play, schedules |
| `rotowire_id` | rosters, ff_playerids |
| `rush_yards` | nextgen_stats, draft_picks |
| `rushing_epa` | player_stats, team_stats |
| `rushing_tds` | player_stats, team_stats |
| `sack` | play_by_play, espn_qbr |
| `sacks_suffered` | player_stats, team_stats |
| `sleeper_id` | rosters, ff_playerids |
| `special_teams_tds` | player_stats, team_stats |
| `sportradar_id` | rosters, ff_playerids |
| `spread_line` | play_by_play, schedules |
| `stadium` | play_by_play, schedules |
| `status` | players, rosters |
| `surface` | play_by_play, schedules |
| `temp` | play_by_play, schedules |
| `total_line` | play_by_play, schedules |
| `wind` | play_by_play, schedules |

---

## Data Dictionaries

### Play By Play

**Description:** Play-by-play data for NFL games with detailed play information, EPA, WPA, and player involvement

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_pbp.html](https://nflreadr.nflverse.com/articles/dictionary_pbp.html)

**Fields:** 188

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `play_id` | numeric | Numeric play id that when used with game_id and drive provides the unique identifier for a single play. |
| `game_id` | character | Ten digit identifier for NFL game. |
| `old_game_id` | character | Legacy NFL game ID. |
| `home_team` | character | String abbreviation for the home team. |
| `away_team` | character | String abbreviation for the away team. |
| `season_type` | character | 'REG' or 'POST' indicating if the game belongs to regular or post season. |
| `week` | numeric | Season week. |
| `posteam` | character | String abbreviation for the team with possession. |
| `posteam_type` | character | String indicating whether the posteam team is home or away. |
| `defteam` | character | String abbreviation for the team on defense. |
| `side_of_field` | character | String abbreviation for which team's side of the field the team with possession is currently on. |
| `yardline_100` | numeric | Numeric distance in yards from the opponent's endzone for the posteam. |
| `game_date` | character | Date of the game. |
| `quarter_seconds_remaining` | numeric | Numeric seconds remaining in the quarter. |
| `half_seconds_remaining` | numeric | Numeric seconds remaining in the half. |
| `game_seconds_remaining` | numeric | Numeric seconds remaining in the game. |
| `game_half` | character | String indicating which half the play is in, either Half1, Half2, or Overtime. |
| `quarter_end` | numeric | Binary indicator for whether or not the row of the data is marking the end of a quarter. |
| `drive` | numeric | Numeric drive number in the game. |
| `sp` | numeric | Binary indicator for whether or not a score occurred on the play. |
| `qtr` | numeric | Quarter of the game (5 is overtime). |
| `down` | numeric | The down for the given play. |
| `goal_to_go` | numeric | Binary indicator for whether or not the posteam is in a goal down situation. |
| `time` | character | Time at start of play provided in string format as minutes:seconds remaining in the quarter. |
| `yrdln` | character | String indicating the current field position for a given play. |
| `ydstogo` | numeric | Numeric yards in distance from either the first down marker or the endzone in goal down situations. |
| `ydsnet` | numeric | Numeric value for total yards gained on the given drive. |
| `desc` | character | Detailed string description for the given play. |
| `play_type` | character | String indicating the type of play: pass, run, punt, field_goal, kickoff, extra_point, qb_kneel, qb_spike, no_play. |
| `yards_gained` | numeric | Numeric yards gained (or lost) by the possessing team, excluding yards gained via fumble recoveries and laterals. |
| `shotgun` | numeric | Binary indicator for whether or not the play was in shotgun formation. |
| `no_huddle` | numeric | Binary indicator for whether or not the play was in no_huddle formation. |
| `qb_dropback` | numeric | Binary indicator for whether or not the QB dropped back on the play. |
| `qb_kneel` | numeric | Binary indicator for whether or not the QB took a knee. |
| `qb_spike` | numeric | Binary indicator for whether or not the QB spiked the ball. |
| `qb_scramble` | numeric | Binary indicator for whether or not the QB scrambled. |
| `pass_length` | character | String indicator for pass length: short or deep. |
| `pass_location` | character | String indicator for pass location: left, middle, or right. |
| `air_yards` | numeric | Numeric value for distance in yards perpendicular to the line of scrimmage at targeted receiver. |
| `yards_after_catch` | numeric | Numeric value for distance in yards where the receiver made reception to where play ended. |
| `run_location` | character | String indicator for location of run: left, middle, or right. |
| `run_gap` | character | String indicator for line gap of run: end, guard, or tackle. |
| `field_goal_result` | character | String indicator for result of field goal attempt: made, missed, or blocked. |
| `kick_distance` | numeric | Numeric distance in yards for kickoffs, field goals, and punts. |
| `extra_point_result` | character | String indicator for the result of the extra point attempt: good, failed, blocked, or aborted. |
| `two_point_conv_result` | character | String indicator for result of two point conversion attempt: success, failure, safety, or return. |
| `home_timeouts_remaining` | numeric | Numeric timeouts remaining in the half for the home team. |
| `away_timeouts_remaining` | numeric | Numeric timeouts remaining in the half for the away team. |
| `timeout` | numeric | Binary indicator for whether or not a timeout was called by either team. |
| `timeout_team` | character | String abbreviation for which team called the timeout. |
| `td_team` | character | String abbreviation for which team scored the touchdown. |
| `td_player_name` | character | String name of the player who scored a touchdown. |
| `td_player_id` | character | Unique identifier of the player who scored a touchdown. |
| `posteam_timeouts_remaining` | numeric | Number of timeouts remaining for the possession team. |
| `defteam_timeouts_remaining` | numeric | Number of timeouts remaining for the team on defense. |
| `total_home_score` | numeric | Score for the home team at the start of the play. |
| `total_away_score` | numeric | Score for the away team at the start of the play. |
| `posteam_score` | numeric | Score the posteam at the start of the play. |
| `defteam_score` | numeric | Score the defteam at the start of the play. |
| `score_differential` | numeric | Score differential between the posteam and defteam at the start of the play. |
| `posteam_score_post` | numeric | Score for the posteam at the end of the play. |
| `defteam_score_post` | numeric | Score for the defteam at the end of the play. |
| `score_differential_post` | numeric | Score differential between the posteam and defteam at the end of the play. |
| `no_score_prob` | numeric | Predicted probability of no score occurring for the rest of the half. |
| `opp_fg_prob` | numeric | Predicted probability of the defteam scoring a FG next. |
| `opp_safety_prob` | numeric | Predicted probability of the defteam scoring a safety next. |
| `opp_td_prob` | numeric | Predicted probability of the defteam scoring a TD next. |
| `fg_prob` | numeric | Predicted probability of the posteam scoring a FG next. |
| `safety_prob` | numeric | Predicted probability of the posteam scoring a safety next. |
| `td_prob` | numeric | Predicted probability of the posteam scoring a TD next. |
| `extra_point_prob` | numeric | Predicted probability of the posteam scoring an extra point. |
| `two_point_conversion_prob` | numeric | Predicted probability of the posteam scoring the two point conversion. |
| `ep` | numeric | Using the scoring event probabilities, the estimated expected points with respect to the possession team. |
| `epa` | numeric | Expected points added (EPA) by the posteam for the given play. |
| `total_home_epa` | numeric | Cumulative total EPA for the home team in the game so far. |
| `total_away_epa` | numeric | Cumulative total EPA for the away team in the game so far. |
| `total_home_rush_epa` | numeric | Cumulative total rushing EPA for the home team in the game so far. |
| `total_away_rush_epa` | numeric | Cumulative total rushing EPA for the away team in the game so far. |
| `total_home_pass_epa` | numeric | Cumulative total passing EPA for the home team in the game so far. |
| `total_away_pass_epa` | numeric | Cumulative total passing EPA for the away team in the game so far. |
| `air_epa` | numeric | EPA from the air yards alone. For completions this represents the actual value provided through the air. |
| `yac_epa` | numeric | EPA from the yards after catch alone. For completions this represents the actual value provided after catch. |
| `comp_air_epa` | numeric | EPA from the air yards alone only for completions. |
| `comp_yac_epa` | numeric | EPA from the yards after catch alone only for completions. |
| `wp` | numeric | Estimated win probability for the posteam given the current situation at the start of the play. |
| `def_wp` | numeric | Estimated win probability for the defteam. |
| `home_wp` | numeric | Estimated win probability for the home team. |
| `away_wp` | numeric | Estimated win probability for the away team. |
| `wpa` | numeric | Win probability added (WPA) for the posteam. |
| `vegas_wpa` | numeric | Win probability added (WPA) for the posteam: spread_adjusted model. |
| `vegas_home_wpa` | numeric | Win probability added (WPA) for the home team: spread_adjusted model. |
| `home_wp_post` | numeric | Estimated win probability for the home team at the end of the play. |
| `away_wp_post` | numeric | Estimated win probability for the away team at the end of the play. |
| `vegas_wp` | numeric | Estimated win probability for the posteam, incorporating pre-game Vegas line. |
| `vegas_home_wp` | numeric | Estimated win probability for the home team incorporating pre-game Vegas line. |
| `punt_blocked` | numeric | Binary indicator for if the punt was blocked. |
| `first_down_rush` | numeric | Binary indicator for if a running play converted the first down. |
| `first_down_pass` | numeric | Binary indicator for if a passing play converted the first down. |
| `first_down_penalty` | numeric | Binary indicator for if a penalty converted the first down. |
| `third_down_converted` | numeric | Binary indicator for if the first down was converted on third down. |
| `third_down_failed` | numeric | Binary indicator for if the posteam failed to convert first down on third down. |
| `fourth_down_converted` | numeric | Binary indicator for if the first down was converted on fourth down. |
| `fourth_down_failed` | numeric | Binary indicator for if the posteam failed to convert first down on fourth down. |
| `incomplete_pass` | numeric | Binary indicator for if the pass was incomplete. |
| `touchback` | numeric | Binary indicator for if a touchback occurred on the play. |
| `interception` | numeric | Binary indicator for if the pass was intercepted. |
| `fumble_forced` | numeric | Binary indicator for if the fumble was forced. |
| `fumble_not_forced` | numeric | Binary indicator for if the fumble was not forced. |
| `fumble_out_of_bounds` | numeric | Binary indicator for if the fumble went out of bounds. |
| `solo_tackle` | numeric | Binary indicator if the play had a solo tackle (could be multiple due to fumbles). |
| `safety` | numeric | Binary indicator for whether or not a safety occurred. |
| `penalty` | numeric | Binary indicator for whether or not a penalty occurred. |
| `tackled_for_loss` | numeric | Binary indicator for whether or not a tackle for loss on a run play occurred. |
| `fumble_lost` | numeric | Binary indicator for if the fumble was lost. |
| `qb_hit` | numeric | Binary indicator if the QB was hit on the play. |
| `rush_attempt` | numeric | Binary indicator for if the play was a run. |
| `pass_attempt` | numeric | Binary indicator for if the play was a pass attempt (includes sacks). |
| `sack` | numeric | Binary indicator for if the play ended in a sack. |
| `touchdown` | numeric | Binary indicator for if the play resulted in a TD. |
| `pass_touchdown` | numeric | Binary indicator for if the play resulted in a passing TD. |
| `rush_touchdown` | numeric | Binary indicator for if the play resulted in a rushing TD. |
| `return_touchdown` | numeric | Binary indicator for if the play resulted in a return TD. |
| `extra_point_attempt` | numeric | Binary indicator for extra point attempt. |
| `two_point_attempt` | numeric | Binary indicator for two point conversion attempt. |
| `field_goal_attempt` | numeric | Binary indicator for field goal attempt. |
| `kickoff_attempt` | numeric | Binary indicator for kickoff. |
| `punt_attempt` | numeric | Binary indicator for punts. |
| `fumble` | numeric | Binary indicator for if a fumble occurred. |
| `complete_pass` | numeric | Binary indicator for if the pass was completed. |
| `passer_player_id` | character | Unique identifier for the player that attempted the pass. |
| `passer_player_name` | character | String name for the player that attempted the pass. |
| `passing_yards` | numeric | Numeric yards by the passer_player_name, including yards gained in pass plays with laterals. |
| `receiver_player_id` | character | Unique identifier for the receiver that was targeted on the pass. |
| `receiver_player_name` | character | String name for the targeted receiver. |
| `receiving_yards` | numeric | Numeric yards by the receiver_player_name, excluding yards gained in pass plays with laterals. |
| `rusher_player_id` | character | Unique identifier for the player that attempted the run. |
| `rusher_player_name` | character | String name for the player that attempted the run. |
| `rushing_yards` | numeric | Numeric yards by the rusher_player_name, excluding yards gained in rush plays with laterals. |
| `interception_player_id` | character | Unique identifier for the player that intercepted the pass. |
| `interception_player_name` | character | String name for the player that intercepted the pass. |
| `sack_player_id` | character | Unique identifier of the player who recorded a solo sack. |
| `sack_player_name` | character | String name of the player who recorded a solo sack. |
| `penalty_team` | character | String abbreviation of the team with the penalty. |
| `penalty_player_id` | character | Unique identifier for the player with the penalty. |
| `penalty_player_name` | character | String name for the player with the penalty. |
| `penalty_yards` | numeric | Yards gained (or lost) by the posteam from the penalty. |
| `penalty_type` | character | String indicating the penalty type of the first penalty in the given play. |
| `season` | numeric | 4 digit number indicating to which season the game belongs to. |
| `cp` | numeric | Numeric value indicating the probability for a complete pass based on comparable situations. |
| `cpoe` | numeric | For a single pass play, 1 - cp when completed or 0 - cp when incomplete. |
| `series` | numeric | Starts at 1, each new first down increments, numbers shared across both teams. |
| `series_success` | numeric | 1: scored touchdown, gained enough yards for first down. |
| `series_result` | character | Possible values: First down, Touchdown, Opp touchdown, Field goal, Missed field goal, Safety, Turnover, Punt. |
| `stadium` | character | Game site name. |
| `weather` | character | String describing the weather including temperature, humidity and wind. |
| `spread_line` | numeric | The closing spread line for the game. |
| `total_line` | numeric | The closing total line for the game. |
| `div_game` | numeric | Binary indicator for if the given game was a division game. |
| `roof` | character | One of 'dome', 'outdoors', 'closed', 'open' indicating the roof status of the stadium. |
| `surface` | character | What type of ground the game was played on. |
| `temp` | numeric | The temperature at the stadium only for 'roof' = 'outdoors' or 'open'. |
| `wind` | numeric | The speed of the wind in miles/hour only for 'roof' = 'outdoors' or 'open'. |
| `home_coach` | character | First and last name of the home team coach. |
| `away_coach` | character | First and last name of the away team coach. |
| `success` | numeric | Binary indicator whether epa > 0 in the given play. |
| `passer` | character | Name of the dropback player (scrambles included) including plays with penalties. |
| `rusher` | character | Name of the rusher (no scrambles) including plays with penalties. |
| `receiver` | character | Name of the receiver including plays with penalties. |
| `pass` | numeric | Binary indicator if the play was a pass play (sacks and scrambles included). |
| `rush` | numeric | Binary indicator if the play was a rushing play. |
| `first_down` | numeric | Binary indicator if the play ended in a first down. |
| `special` | numeric | Binary indicator if play_type is one of extra_point, field_goal, kickoff, or punt. |
| `play` | numeric | Binary indicator: 1 if the play was a 'normal' play (including penalties), 0 otherwise. |
| `passer_id` | character | ID of the player in the 'passer' column. |
| `rusher_id` | character | ID of the player in the 'rusher' column. |
| `receiver_id` | character | ID of the player in the 'receiver' column. |
| `name` | character | Name of the 'passer' if not 'NA', or name of the 'rusher' otherwise. |
| `id` | character | ID of the player in the 'name' column. |
| `fantasy_player_name` | character | Name of the rusher on rush plays or receiver on pass plays (from official stats). |
| `fantasy_player_id` | character | ID of the rusher on rush plays or receiver on pass plays (from official stats). |
| `qb_epa` | numeric | Gives QB credit for EPA for up to the point where a receiver lost a fumble. |
| `xyac_epa` | numeric | Expected value of EPA gained after the catch. |
| `xyac_mean_yardage` | numeric | Average expected yards after the catch based on where the ball was caught. |
| `xyac_median_yardage` | numeric | Median expected yards after the catch based on where the ball was caught. |
| `xyac_success` | numeric | Probability play earns positive EPA relative to where play started. |
| `xyac_fd` | numeric | Probability play earns a first down based on where the ball was caught. |
| `xpass` | numeric | Probability of dropback scaled from 0 to 1. |
| `pass_oe` | numeric | Dropback percent over expected on a given play scaled from 0 to 100. |

[Back to Top](#table-of-contents)

---

### Player Stats

**Description:** Weekly player statistics aggregated from play-by-play data

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_player_stats.html](https://nflreadr.nflverse.com/articles/dictionary_player_stats.html)

**Fields:** 53

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `player_id` | character | Player's gsis_id. Use this to join to other sources. |
| `player_name` | character | Abbreviated name of player as provided by playstats api |
| `player_display_name` | character | Name of player as provided by load_players() |
| `position` | character | Position of player as listed by NFL |
| `position_group` | character | Position group of player as listed by NFL |
| `headshot_url` | character | Player's nfl.com headshot URL |
| `season` | numeric | Official NFL season |
| `week` | numeric | Game week number |
| `season_type` | character | REG for regular season, POST for postseason |
| `team` | character | Abbreviation of player's team |
| `opponent_team` | character | Abbreviation of opponent's team |
| `completions` | numeric | The number of completed passes. |
| `attempts` | numeric | The number of pass attempts as defined by the NFL. |
| `passing_yards` | numeric | Yards gained on pass plays. |
| `passing_tds` | numeric | The number of passing touchdowns. |
| `passing_interceptions` | numeric | Number of passing interceptions |
| `sacks_suffered` | numeric | Number of sacks taken as a QB |
| `sack_yards_lost` | numeric | Yards lost from sacks suffered by this player |
| `sack_fumbles` | numeric | The number of sacks suffered with a fumble. |
| `sack_fumbles_lost` | numeric | The number of sacks suffered with a lost fumble. |
| `passing_air_yards` | numeric | Passing air yards (includes incomplete passes). |
| `passing_yards_after_catch` | numeric | Yards after the catch gained on plays in which player was the passer |
| `passing_first_downs` | numeric | First downs on pass attempts. |
| `passing_epa` | numeric | Total expected points added on pass attempts and sacks |
| `passing_cpoe` | numeric | Completion percentage over expected for this player. |
| `passing_2pt_conversions` | numeric | Two-point conversion passes. |
| `pacr` | numeric | Passing Air Conversion Ratio - passing yards per air yards per game |
| `carries` | numeric | The number of official rush attempts (incl. scrambles and kneel downs). |
| `rushing_yards` | numeric | Yards gained when rushing with the ball (incl. scrambles and kneel downs). |
| `rushing_tds` | numeric | The number of rushing touchdowns (incl. scrambles). |
| `rushing_fumbles` | numeric | The number of rushes with a fumble. |
| `rushing_fumbles_lost` | numeric | The number of rushes with a lost fumble. |
| `rushing_first_downs` | numeric | First downs on rush attempts (incl. scrambles). |
| `rushing_epa` | numeric | Expected points added on rush attempts (incl. scrambles and kneel downs). |
| `rushing_2pt_conversions` | numeric | Two-point conversion rushes |
| `receptions` | numeric | The number of pass receptions. |
| `targets` | numeric | The number of pass plays where the player was the targeted receiver. |
| `receiving_yards` | numeric | Yards gained after a pass reception. |
| `receiving_tds` | numeric | The number of touchdowns following a pass reception. |
| `receiving_fumbles` | numeric | The number of fumbles after a pass reception. |
| `receiving_fumbles_lost` | numeric | The number of fumbles lost after a pass reception. |
| `receiving_air_yards` | numeric | Receiving air yards (incl. incomplete passes). |
| `receiving_yards_after_catch` | numeric | Yards after the catch gained on plays in which player was receiver |
| `receiving_first_downs` | numeric | Total number of first downs gained on receptions |
| `receiving_epa` | numeric | Total EPA on plays where this receiver was targeted |
| `receiving_2pt_conversions` | numeric | Two-point conversion receptions |
| `racr` | numeric | Receiving Air Conversion Ratio - receiving yards per air yards per game |
| `target_share` | numeric | Player's share of team receiving targets in this game |
| `air_yards_share` | numeric | Player's share of the team's air yards in this game |
| `wopr` | numeric | Weighted OPportunity Rating - weighted average contextualizing fantasy usage |
| `special_teams_tds` | numeric | Total number of kick/punt return touchdowns |
| `fantasy_points` | numeric | Standard fantasy points. |
| `fantasy_points_ppr` | numeric | PPR fantasy points. |

[Back to Top](#table-of-contents)

---

### Players

**Description:** NFL player biographical and identification information

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_players.html](https://nflreadr.nflverse.com/articles/dictionary_players.html)

**Fields:** 33

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `gsis_id` | character | GSIS ID for player. Primary key for all data. |
| `display_name` | character | Full name of player |
| `first_name` | character | First name of player |
| `last_name` | character | Last name of player |
| `short_name` | character | Player short name (i.e. F.Last) |
| `football_name` | character | Common player name |
| `suffix` | character | Name suffix of player |
| `esb_id` | character | ESB ID of player |
| `nfl_id` | character | NFL ID of player (used in Big Data Bowl) |
| `pfr_id` | character | Pro-Football-Reference ID for player |
| `pff_id` | character | PFF ID for player |
| `otc_id` | character | Over the Cap ID for player |
| `espn_id` | character | ESPN ID for player |
| `smart_id` | character | SMART ID for player |
| `birth_date` | character | Player birth date |
| `position_group` | character | Position group of player as listed by NFL |
| `position` | character | Position of player as listed by NFL |
| `ngs_position_group` | character | Position group of player as listed by Next Gen Stats |
| `ngs_position` | character | Position of player as listed by Next Gen Stats |
| `headshot` | character | NFL headshot url for player |
| `college_name` | character | Official college (usually the last one attended) |
| `college_conference` | character | Conference of college |
| `jersey_number` | character | Latest jersey number of player |
| `latest_team` | character | Latest team the player was listed in |
| `status` | character | Roster status |
| `height` | numeric | Height of player (inches) |
| `weight` | numeric | Weight of player (lbs) |
| `rookie_season` | numeric | Season the player was a rookie |
| `draft_year` | numeric | Year that player was drafted |
| `draft_round` | numeric | Round that player was drafted in |
| `draft_pick` | numeric | Pick number of player |
| `draft_team` | character | Team that drafted player |
| `years_of_experience` | numeric | Years played in league |

[Back to Top](#table-of-contents)

---

### Rosters

**Description:** Weekly NFL team rosters with player status and identifiers

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_rosters.html](https://nflreadr.nflverse.com/articles/dictionary_rosters.html)

**Fields:** 31

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `season` | numeric | NFL season. Defaults to current year after March. |
| `team` | character | NFL team. Uses official abbreviations as per NFL.com |
| `position` | character | Primary position as reported by NFL.com |
| `depth_chart_position` | character | Position assigned on depth chart. |
| `jersey_number` | numeric | Jersey number. |
| `status` | character | Roster status: Active, Inactive, Injured Reserve, Practice Squad etc |
| `full_name` | character | Full name as per NFL.com |
| `first_name` | character | First name as per NFL.com |
| `last_name` | character | Last name as per NFL.com |
| `birth_date` | date | Birthdate, as recorded by Sleeper API |
| `height` | character | Official height, in inches |
| `weight` | character | Official weight, in pounds |
| `college` | character | Official college (usually the last one attended) |
| `gsis_id` | character | Game Stats and Info Service ID: the primary ID for play-by-play data. |
| `espn_id` | numeric | Player ID for ESPN API |
| `sportradar_id` | character | Player ID for Sportradar API |
| `yahoo_id` | numeric | Player ID for Yahoo API |
| `rotowire_id` | numeric | Player ID for Rotowire |
| `pff_id` | numeric | Player ID for Pro Football Focus |
| `pfr_id` | character | Player ID for Pro Football Reference |
| `fantasy_data_id` | numeric | Player ID for FantasyData |
| `sleeper_id` | character | Player ID for Sleeper API |
| `years_exp` | numeric | Years played in league |
| `headshot_url` | character | URL string that points to player photos |
| `ngs_position` | character | Primary position as reported by the NextGen stats API. |
| `week` | numeric | The most recent week of that season that a player appeared on the roster. |
| `game_type` | character | The most recent game type of that season. |
| `entry_year` | numeric | The year a player first became eligible to play in the NFL. |
| `rookie_year` | numeric | The year a player lost their rookie eligibility. |
| `draft_club` | character | The team that originally drafted a player. |
| `draft_number` | numeric | The number pick that was used to select a given player. |

[Back to Top](#table-of-contents)

---

### Schedules

**Description:** NFL game schedules with scores, betting lines, and game information

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_schedules.html](https://nflreadr.nflverse.com/articles/dictionary_schedules.html)

**Fields:** 38

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `game_id` | numeric | A human-readable game ID. |
| `season` | numeric | The year of the NFL season. |
| `game_type` | character | What type of game? One of REG, WC, DIV, CON, SB |
| `week` | numeric | The week of the NFL season the game occurs in. |
| `gameday` | character | The date on which the game occurred. |
| `weekday` | character | The day of the week on which the game occurred. |
| `gametime` | character | The kickoff time of the game in 24-hour Eastern time. |
| `away_team` | character | The away team. |
| `away_score` | numeric | The number of points the away team scored. |
| `home_team` | character | The home team. |
| `home_score` | numeric | The number of points the home team scored. |
| `location` | character | Either Home or Neutral. |
| `result` | numeric | Home score minus away score. |
| `total` | numeric | Sum of both teams' scores. |
| `overtime` | numeric | Binary indicator of whether the game went to overtime. |
| `old_game_id` | numeric | The old id for the game assigned by the NFL. |
| `away_rest` | numeric | Days of rest the away team had before the game. |
| `home_rest` | numeric | Days of rest the home team had before the game. |
| `away_moneyline` | numeric | Odds for away team to win the game. |
| `home_moneyline` | numeric | Odds for home team to win the game. |
| `spread_line` | numeric | Game spread; positive favors home team. |
| `away_spread_odds` | numeric | Odds for away team to cover the spread. |
| `home_spread_odds` | numeric | Odds for home team to cover the spread. |
| `total_line` | numeric | The total line for the game. |
| `under_odds` | numeric | Odds that total score would be under the total_line. |
| `over_odds` | numeric | Odds that total score would be over the total_line. |
| `div_game` | numeric | Binary indicator of whether game was between two division rivals. |
| `roof` | character | Stadium roof status: outdoors, open, closed, or dome. |
| `surface` | character | Type of ground the game was played on. |
| `temp` | numeric | Temperature at the stadium (outdoors and open games only). |
| `wind` | numeric | Wind speed in miles/hour (outdoors and open games only). |
| `away_qb_name` | numeric | Name of away team starting QB. |
| `home_qb_name` | numeric | Name of home team starting QB. |
| `away_coach` | character | Name of the head coach of the away team. |
| `home_coach` | character | Name of the head coach of the home team. |
| `referee` | character | Name of the game's referee. |
| `stadium_id` | character | ID of the stadium where the game took place. |
| `stadium` | character | Name of the stadium. |

[Back to Top](#table-of-contents)

---

### Nextgen Stats

**Description:** Next Gen Stats tracking data for passing, rushing, and receiving

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html](https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html)

**Fields:** 48

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `season_type` | character | Either REG or POST |
| `player_display_name` | character | Full name of the player |
| `player_position` | character | Position of the player according to NGS |
| `team_abbr` | character | Official team abbreviation |
| `player_gsis_id` | character | Unique player identifier |
| `player_first_name` | character | Player's first name |
| `player_last_name` | character | Player's last name |
| `player_short_name` | character | Shortened version of player's name |
| `season` | numeric | NFL season year |
| `week` | numeric | Week number in NFL season |
| `avg_time_to_throw` | numeric | Average time elapsed from the time of snap to throw on every pass attempt |
| `avg_completed_air_yards` | numeric | Mean air yards on successful passes |
| `avg_intended_air_yards` | numeric | Mean air yards across all pass attempts |
| `avg_air_yards_differential` | numeric | Difference between completed and intended air yards |
| `aggressiveness` | numeric | % of attempts into tight windows (defender within 1 yard of receiver) |
| `max_completed_air_distance` | numeric | Maximum actual distance ball traveled on completed pass |
| `avg_air_yards_to_sticks` | numeric | Air yards relative to first down marker on all attempts |
| `attempts` | numeric | Total pass attempts |
| `pass_yards` | numeric | Yards gained on pass plays |
| `pass_touchdowns` | numeric | Touchdowns scored on pass plays |
| `interceptions` | numeric | Passes intercepted |
| `passer_rating` | numeric | Overall NFL passer rating |
| `completions` | numeric | Completed passes |
| `completion_percentage` | numeric | Percentage of completed passes |
| `expected_completion_percentage` | numeric | Projected completion percentage based on completion probability |
| `completion_percentage_above_expectation` | numeric | Actual vs. expected completion percentage |
| `avg_cushion` | numeric | Yards between receiver and defender at snap (WR/TE only) |
| `avg_separation` | numeric | Distance measured between a WR/TE and the nearest defender at the time of catch |
| `percent_share_of_intended_air_yards` | numeric | Receiver's intended air yards as team percentage |
| `receptions` | numeric | Completed passes for receiver |
| `targets` | numeric | Passes thrown to receiver |
| `catch_percentage` | numeric | Percentage of caught passes relative to targets |
| `yards` | numeric | Receiving yards |
| `rec_touchdowns` | numeric | Touchdown receptions |
| `avg_yac` | numeric | Mean yards after catch by receiver |
| `avg_expected_yac` | numeric | Projected yards after catch based on tracking data factors |
| `avg_yac_above_expectation` | numeric | Actual vs. expected yards after catch |
| `efficiency` | numeric | Total distance traveled per rushing yards gained (lower = more vertical) |
| `percent_attempts_gte_eight_defenders` | numeric | % of rushes facing 8+ defenders in box |
| `avg_time_to_los` | numeric | Average seconds before crossing line of scrimmage on rushes |
| `rush_attempts` | numeric | Total rushing attempts |
| `rush_yards` | numeric | Rushing yards gained |
| `expected_rush_yards` | numeric | Projected rushing yards via NGS model |
| `rush_yards_over_expected` | numeric | Actual vs. expected rushing yards |
| `avg_rush_yards` | numeric | Mean rushing yards per attempt |
| `rush_yards_over_expected_per_att` | numeric | Average rush yards above expectation per carry |
| `rush_pct_over_expected` | numeric | Rushing percentage above expectation |
| `rush_touchdowns` | numeric | Rushing touchdowns scored |

[Back to Top](#table-of-contents)

---

### Pfr Passing

**Description:** Pro Football Reference advanced passing statistics

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_pfr_passing.html](https://nflreadr.nflverse.com/articles/dictionary_pfr_passing.html)

**Fields:** 28

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `player` | character | Player name |
| `team` | character | Player team |
| `pfr_id` | character | PFR player ID |
| `pass_attempts` | numeric | Pass attempts |
| `batted_balls` | numeric | Batted balls |
| `throwaways` | numeric | Throwaways |
| `spikes` | numeric | Spikes |
| `drops` | numeric | Throws dropped |
| `drop_pct` | numeric | Percent of throws dropped |
| `bad_throws` | numeric | Bad throws |
| `bad_throw_pct` | numeric | Percent of throws that were bad |
| `on_tgt_throws` | numeric | On target throws |
| `on_tgt_pct` | numeric | Percent of throws on target |
| `season` | numeric | Season |
| `pocket_time` | numeric | Average time in pocket |
| `times_blitzed` | numeric | Number of times blitzed |
| `times_hurried` | numeric | Number of times hurried |
| `times_hit` | numeric | Number of times hit |
| `times_pressured` | numeric | Number of times pressured |
| `pressure_pct` | numeric | Percent of the time pressured |
| `rpo_plays` | numeric | Number of RPO plays |
| `rpo_yards` | numeric | Yards on RPOs |
| `rpo_pass_att` | numeric | Number of pass attempts on RPOs |
| `rpo_pass_yards` | numeric | Passing yards on RPOs |
| `rpo_rush_att` | numeric | Rush attempts on RPOs |
| `rpo_rush_yards` | numeric | Rushing yards on RPOs |
| `pa_pass_att` | numeric | Play action pass attempts |
| `pa_pass_yards` | numeric | Play action passing yards |

[Back to Top](#table-of-contents)

---

### Snap Counts

**Description:** Player snap counts by game and unit (offense/defense/special teams)

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html](https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html)

**Fields:** 16

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `game_id` | character | nflfastR game ID |
| `pfr_game_id` | character | PFR game ID |
| `season` | numeric | Season of game |
| `game_type` | character | Type of game (regular or postseason) |
| `week` | numeric | Week of game in NFL season |
| `player` | character | Player name |
| `pfr_player_id` | character | Player PFR ID |
| `position` | character | Position of player |
| `team` | character | Team of player |
| `opponent` | character | Opposing team of player |
| `offense_snaps` | numeric | Number of snaps on offense |
| `offense_pct` | numeric | Percent of offensive snaps taken |
| `defense_snaps` | numeric | Number of snaps on defense |
| `defense_pct` | numeric | Percent of defensive snaps taken |
| `st_snaps` | numeric | Number of snaps on special teams |
| `st_pct` | numeric | Percent of special teams snaps taken |

[Back to Top](#table-of-contents)

---

### Team Stats

**Description:** Weekly team statistics aggregated from play-by-play data

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_team_stats.html](https://nflreadr.nflverse.com/articles/dictionary_team_stats.html)

**Fields:** 28

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `season` | numeric | Official NFL season |
| `week` | numeric | Game week number |
| `team` | character | Abbreviation of team |
| `season_type` | character | REG for regular season, POST for postseason |
| `opponent_team` | character | Abbreviation of opponent team |
| `completions` | numeric | The number of completed passes. |
| `attempts` | numeric | The number of pass attempts as defined by the NFL. |
| `passing_yards` | numeric | Yards gained on pass plays. |
| `passing_tds` | numeric | The number of passing touchdowns. |
| `passing_interceptions` | numeric | Number of passing interceptions |
| `sacks_suffered` | numeric | Number of sacks taken as a QB |
| `passing_epa` | numeric | Expected points added on passing plays and sacks |
| `passing_cpoe` | numeric | Completion percentage over expected for this team. |
| `carries` | numeric | Official rushing attempts including scrambles |
| `rushing_yards` | numeric | Yards gained on rushing plays including scrambles |
| `rushing_tds` | numeric | The number of rushing touchdowns (incl. scrambles). |
| `rushing_epa` | numeric | Expected points added on rushing attempts |
| `receptions` | numeric | The number of pass receptions. |
| `targets` | numeric | The number of pass plays where the player was the targeted receiver. |
| `receiving_yards` | numeric | Yards gained after a pass reception. |
| `receiving_tds` | numeric | Touchdowns following pass reception |
| `receiving_epa` | numeric | Total EPA on plays where this receiver was targeted |
| `special_teams_tds` | numeric | Total number of kick/punt return touchdowns |
| `def_tackles_solo` | numeric | Total number of solo tackles for this team |
| `def_sacks` | numeric | Number of sacks by this team |
| `def_interceptions` | numeric | Number of interceptions forced by this team |
| `def_pass_defended` | numeric | Number of passes defended/broken up by this team |
| `def_tds` | numeric | Number of defensive touchdowns scored by this team |

[Back to Top](#table-of-contents)

---

### Combine

**Description:** NFL Combine athletic testing results

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_combine.html](https://nflreadr.nflverse.com/articles/dictionary_combine.html)

**Fields:** 18

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `season` | numeric | Season(s) the specified combine occurred |
| `draft_year` | numeric | Year that player was drafted |
| `draft_team` | character | Team that drafted player |
| `draft_round` | numeric | Round that player was drafted in |
| `draft_ovr` | numeric | Pick number of player |
| `pfr_id` | numeric | Pro-Football-Reference ID for player |
| `cfb_id` | numeric | Sports Reference (CFB) ID for player |
| `player_name` | character | Full name of player |
| `pos` | character | Position of player |
| `school` | character | College of player |
| `ht` | numeric | Height of player (feet and inches) |
| `wt` | numeric | Weight of player (lbs) |
| `forty` | numeric | Player's 40 yard dash time at combine (seconds) |
| `bench` | numeric | Reps benched by player at combine |
| `vertical` | numeric | Player's vertical jump at combine (inches) |
| `broad_jump` | numeric | Player's broad jump at combine (inches) |
| `cone` | numeric | Player's 3 cone drill time at combine (seconds) |
| `shuttle` | numeric | Player's shuttle run time at combine (seconds) |

[Back to Top](#table-of-contents)

---

### Contracts

**Description:** NFL player contract information from Over The Cap

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_contracts.html](https://nflreadr.nflverse.com/articles/dictionary_contracts.html)

**Fields:** 15

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `player` | character | Player name |
| `position` | character | Player's position |
| `team` | character | Player's team |
| `is_active` | logical | Active contract |
| `year_signed` | numeric | Year the contract was signed |
| `years` | numeric | Contract length |
| `value` | numeric | Total contract value |
| `apy` | numeric | Average money per contract year |
| `guaranteed` | numeric | Total guaranteed money |
| `apy_cap_pct` | numeric | Average money per year as percentage of salary cap at signing |
| `inflated_value` | numeric | Total contract value inflated for salary cap rise |
| `inflated_apy` | numeric | Average money per year inflated for salary cap rise |
| `inflated_guaranteed` | numeric | Total guaranteed money inflated for salary cap rise |
| `player_page` | character | Player's OverTheCap url |
| `otc_id` | numeric | Player's OverTheCap ID |

[Back to Top](#table-of-contents)

---

### Depth Charts

**Description:** NFL team depth charts with player positions and rankings

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_depth_charts.html](https://nflreadr.nflverse.com/articles/dictionary_depth_charts.html)

**Fields:** 12

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `dt` | character | Timestamp indicating when the data record was loaded |
| `team` | character | Team that depth chart belongs to |
| `player_name` | character | Full name of player |
| `espn_id` | character | ESPN Player ID |
| `gsis_id` | character | Game Stats and Info Service ID |
| `pos_grp_id` | character | Player position group identifier |
| `pos_grp` | character | Player position group: formation of offense, defense, or special teams |
| `pos_id` | character | Player position identifier |
| `pos_name` | character | Player position name |
| `pos_abb` | character | Player position abbreviation |
| `pos_slot` | numeric | A number assigned to each position in a formation |
| `pos_rank` | numeric | Player's rank on depth chart grouped by pos_slot |

[Back to Top](#table-of-contents)

---

### Draft Picks

**Description:** Historical NFL draft pick data with career statistics

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_draft_picks.html](https://nflreadr.nflverse.com/articles/dictionary_draft_picks.html)

**Fields:** 36

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `season` | integer | Draft Year |
| `round` | integer | Draft round |
| `pick` | integer | Draft overall pick |
| `team` | character | Draft team |
| `gsis_id` | character | ID for joining with nflverse data |
| `pfr_player_id` | character | ID from Pro Football Reference |
| `cfb_player_id` | character | ID from College Football Reference |
| `pfr_player_name` | character | Player's name as recorded by PFR |
| `hof` | logical | Whether player has been selected to the Pro Football Hall of Fame |
| `position` | character | Player position as recorded by PFR |
| `category` | character | Broader category of player positions |
| `side` | character | O for offense, D for defense, S for special teams |
| `college` | character | College attended in final year |
| `age` | integer | Player age as of draft |
| `to` | integer | Final season played in NFL |
| `allpro` | numeric | Number of AP First Team All-Pro selections |
| `probowls` | numeric | Number of Pro Bowls |
| `seasons_started` | numeric | Number of seasons recorded as primary starter |
| `w_av` | numeric | Weighted Approximate Value |
| `car_av` | numeric | Career Approximate Value |
| `dr_av` | numeric | Draft Approximate Value |
| `games` | numeric | Games played in career |
| `pass_completions` | numeric | Career pass completions |
| `pass_attempts` | numeric | Career pass attempts |
| `pass_yards` | numeric | Career pass yards thrown |
| `pass_tds` | numeric | Career pass touchdowns thrown |
| `pass_ints` | numeric | Career pass interceptions thrown |
| `rush_atts` | numeric | Career rushing attempts |
| `rush_yards` | numeric | Career rushing yards |
| `rush_tds` | numeric | Career rushing touchdowns |
| `receptions` | numeric | Career receptions |
| `rec_yards` | numeric | Career receiving yards |
| `rec_tds` | numeric | Career receiving touchdowns |
| `def_solo_tackles` | numeric | Career solo tackles |
| `def_ints` | numeric | Career interceptions |
| `def_sacks` | numeric | Career sacks |

[Back to Top](#table-of-contents)

---

### Espn Qbr

**Description:** ESPN's Total QBR quarterback rating metric

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_espn_qbr.html](https://nflreadr.nflverse.com/articles/dictionary_espn_qbr.html)

**Fields:** 23

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `season` | numeric | Season(s) the specified timeframe belongs to. |
| `season_type` | character | REG or POST indicating regular or post season. |
| `game_week` | numeric | Season week |
| `team_abb` | character | Abbreviation of Team of Player |
| `player_id` | numeric | ESPN Player ID |
| `name_short` | character | Short name of player (First Initial, Last Name) |
| `rank` | numeric | QBR Rank in specified timeframe |
| `qbr_total` | numeric | Adjusted Total QBR, adjusts quarterback play on 0-100 scale |
| `pts_added` | numeric | Points contributed by quarterback above average level QB |
| `qb_plays` | numeric | Total dropbacks for the quarterback (excludes handoffs) |
| `epa_total` | numeric | Total Expected Points Added by quarterback |
| `pass` | numeric | Expected Points Added on pass plays |
| `run` | numeric | Expected Points Added on run plays |
| `exp_sack` | numeric | Expected EPA Added on Sacks |
| `penalty` | numeric | Expected Points Added on penalties |
| `qbr_raw` | numeric | Raw total QBR, not adjusted for opponent strength |
| `sack` | numeric | Expected Points Added on sacks |
| `name_first` | character | First Name of Quarterback |
| `name_last` | character | Last Name of Quarterback |
| `name_display` | character | Full Name of Quarterback |
| `headshot_href` | character | Link to ESPN Headshot of Player |
| `team` | character | Full Team Name of Player |
| `qualified` | character | Indicator of whether player meets minimum play requirement |

[Back to Top](#table-of-contents)

---

### Injuries

**Description:** NFL injury reports with practice status and game designations

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_injuries.html](https://nflreadr.nflverse.com/articles/dictionary_injuries.html)

**Fields:** 16

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `season` | numeric | Season(s) the specified timeframe belongs to. |
| `season_type` | numeric | REG or POST indicating regular or post season timeframe. |
| `team` | character | Team of injured player |
| `week` | numeric | Week that injury occurred |
| `gsis_id` | numeric | Game Stats and Info Service ID |
| `position` | character | Position of injured player |
| `full_name` | character | Full name of player |
| `first_name` | character | First name of player |
| `last_name` | character | Last name of player |
| `report_primary_injury` | character | Primary injury listed on official injury report |
| `report_secondary_injury` | character | Secondary injury listed on official injury report |
| `report_status` | character | Player's status for game on official injury report |
| `practice_primary_injury` | character | Primary injury listed on practice injury report |
| `practice_secondary_injury` | character | Secondary injury listed on practice injury report |
| `practice_status` | character | Player's participation in practice |
| `date_modified` | character | Date and time that injury information was updated |

[Back to Top](#table-of-contents)

---

### Participation

**Description:** Play-level participation data including formations and coverage

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_participation.html](https://nflreadr.nflverse.com/articles/dictionary_participation.html)

**Fields:** 23

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `nflverse_game_id` | character | nflverse identifier for games |
| `old_game_id` | character | Legacy NFL game ID. |
| `play_id` | integer | Numeric play id for unique play identification |
| `possession_team` | character | Team abbreviation holding possession. |
| `offense_formation` | character | Formation the offense lines up in to snap the ball. |
| `offense_personnel` | character | The positions of the offensive personnel |
| `defenders_in_box` | integer | Number of defensive players lined up in the box at the snap |
| `defense_personnel` | character | The positions of the defensive personnel |
| `number_of_pass_rushers` | integer | Number of defensive player who rushed the passer |
| `players_on_play` | character | A list of every player on the field for the play |
| `offense_players` | character | A list of every offensive player on the field |
| `defense_players` | character | A list of every defensive player on the field |
| `n_offense` | integer | Number of offensive players on the field. |
| `n_defense` | integer | Number of defensive players on the field. |
| `time_to_throw` | double | Duration between snap and release |
| `was_pressure` | logical | Whether the QB was pressured on a play. |
| `route` | character | Primary receiver route type (CORNER, GO, SLANT, WHEEL, etc.). |
| `defense_man_zone_type` | character | Defense coverage approach (man or zone). |
| `defense_coverage_type` | character | Defense coverage type (COVER_0 through COVER_9, COMBO, BLOWN). |
| `offense_names` | character | Offensive player names in gsis_id order. |
| `defense_names` | character | Defensive player names in gsis_id order. |
| `offense_positions` | character | Offensive player positions in gsis_id order. |
| `defense_positions` | character | Defensive player positions in gsis_id order. |

[Back to Top](#table-of-contents)

---

### Ftn Charting

**Description:** FTN Data advanced charting for play-action, RPO, and more

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html](https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html)

**Fields:** 28

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `ftn_game_id` | numeric | FTN game ID |
| `nflverse_game_id` | character | Game ID used by nflverse |
| `season` | numeric | NFL season starting year. Data available from 2022 onwards. |
| `week` | numeric | NFL week number. |
| `ftn_play_id` | numeric | FTN play ID |
| `nflverse_play_id` | numeric | Play ID used by nflverse, corresponds to GSIS play ID |
| `starting_hash` | character | Hash the ball was placed (L, M, R) |
| `qb_location` | character | Pre-snap position of quarterback (U, S, P) |
| `n_offense_backfield` | numeric | Number of players in the backfield at the snap |
| `is_no_huddle` | logical | No huddle |
| `is_motion` | logical | Motion occurred on the play before or at the time of the snap |
| `is_play_action` | logical | Play-action pass |
| `is_screen_pass` | logical | Screen pass |
| `is_rpo` | logical | Play is considered run-pass option |
| `is_trick_play` | logical | Trick play |
| `is_qb_out_of_pocket` | logical | Quarterback moved out of pocket |
| `is_interception_worthy` | logical | Interception worthy pass |
| `is_throw_away` | logical | Quarterback thrown away |
| `read_thrown` | character | Read the ball was thrown |
| `is_catchable_ball` | logical | Catchable ball (throws generally on target not defended away) |
| `is_contested_ball` | logical | Contested ball (receiver facing physical contact at catch) |
| `is_created_reception` | logical | Created reception (only occurs due to exceptional receiver play) |
| `is_drop` | logical | Receiver drop |
| `is_qb_sneak` | logical | Quarterback sneak |
| `n_blitzers` | numeric | Number of blitzers |
| `n_pass_rushers` | numeric | Number of pass rushers |
| `is_qb_fault_sack` | logical | Sack that is the fault of the quarterback |
| `date_pulled` | POSIXct | Date the data was retrieved from the FTN Data API |

[Back to Top](#table-of-contents)

---

### Ff Playerids

**Description:** Cross-platform player ID mappings for fantasy football

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_ff_playerids.html](https://nflreadr.nflverse.com/articles/dictionary_ff_playerids.html)

**Fields:** 32

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `mfl_id` | character | MyFantasyLeague.com ID - primary key |
| `sportradar_id` | character | SportRadar identifier (UUID format) |
| `fantasypros_id` | character | FantasyPros.com identifier |
| `gsis_id` | character | NFL Game Stats and Information Services identifier |
| `pff_id` | character | Pro Football Focus identifier |
| `sleeper_id` | character | Sleeper platform identifier |
| `nfl_id` | character | NFL.com identifier |
| `espn_id` | character | ESPN identifier |
| `yahoo_id` | character | Yahoo identifier |
| `fleaflicker_id` | character | Fleaflicker identifier |
| `cbs_id` | character | CBS identifier |
| `rotowire_id` | character | Rotowire identifier |
| `rotoworld_id` | character | Rotoworld identifier |
| `ktc_id` | character | KeepTradeCut identifier |
| `pfr_id` | character | Pro Football Reference identifier |
| `cfbref_id` | character | College Football Reference identifier |
| `stats_id` | character | Stats identifier |
| `stats_global_id` | character | Stats Global identifier |
| `fantasy_data_id` | character | FantasyData identifier |
| `name` | character | Player name in FirstName LastName format |
| `merge_name` | character | Name formatted for matching |
| `position` | character | Player position |
| `team` | character | Player team |
| `birthdate` | date | Player birth date |
| `age` | numeric | Current age rounded to one decimal |
| `draft_year` | numeric | NFL draft year |
| `draft_round` | numeric | Draft round number |
| `draft_pick` | numeric | Pick number within the specific round |
| `draft_ovr` | character | Overall draft selection number |
| `height` | numeric | Height measurement in inches |
| `weight` | numeric | Weight measurement in pounds |
| `college` | character | College institution attended |

[Back to Top](#table-of-contents)

---

### Ff Rankings

**Description:** Fantasy football expert consensus rankings

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_ff_rankings.html](https://nflreadr.nflverse.com/articles/dictionary_ff_rankings.html)

**Fields:** 23

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `fp_page` | character | The relative url the data was scraped from |
| `page_type` | character | Two word identifier for ranking type and position |
| `ecr_type` | character | Two letter identifier for ranking and position type |
| `player` | character | Player name |
| `id` | character | FantasyPros ID |
| `pos` | character | Position as tracked by FP |
| `team` | character | NFL team the player plays for |
| `sportsdata_id` | character | ID - also known as sportradar_id |
| `player_filename` | character | Base URL for player on fantasypros.com |
| `yahoo_id` | character | Yahoo ID |
| `cbs_id` | character | CBS ID |
| `player_image_url` | character | An image of the player |
| `mergename` | character | Player name cleaned for matching |
| `scrape_date` | Date | Date this dataframe was last updated |
| `ecr` | numeric | Average expert ranking for this player |
| `sd` | numeric | Standard deviation of expert rankings |
| `best` | numeric | Highest ranking given by any expert |
| `worst` | numeric | Lowest ranking given by any expert |
| `player_owned_avg` | numeric | Average percentage rostered across ESPN and Yahoo |
| `player_owned_espn` | numeric | Percentage rostered in ESPN leagues |
| `player_owned_yahoo` | numeric | Percentage rostered in Yahoo leagues |
| `rank_delta` | numeric | Change in ranks over a recent period |
| `bye` | numeric | NFL bye week |

[Back to Top](#table-of-contents)

---

### Trades

**Description:** NFL trade transaction data

**Source:** [https://nflreadr.nflverse.com/articles/dictionary_trades.html](https://nflreadr.nflverse.com/articles/dictionary_trades.html)

**Fields:** 11

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `trade_id` | numeric | ID of Trade |
| `season` | numeric | Season the specified trade occurred |
| `trade_date` | numeric | Exact date that trade occurred |
| `gave` | character | Team that gave pick/player in row |
| `received` | character | Team that received pick/player in row |
| `pick_season` | numeric | Draft in which traded pick was in |
| `pick_round` | numeric | Round in which traded pick was in |
| `pick_number` | numeric | Pick number of traded pick |
| `conditional` | numeric | Binary indicator of whether traded pick was conditional |
| `pfr_id` | character | Pro-Football-Reference ID of traded player |
| `pfr_name` | character | Full name of traded player |

[Back to Top](#table-of-contents)

---

