#!/usr/bin/env python3
"""
Fetch NFL injury reports from Sleeper API
Sleeper has reliable, up-to-date injury data for fantasy football
"""

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_sleeper_players():
    """
    Fetch all NFL players from Sleeper API.
    Returns player data including injury status.
    """
    logger.info("Fetching player data from Sleeper API...")

    url = "https://api.sleeper.app/v1/players/nfl"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        players = response.json()

        logger.info(f"✅ Retrieved {len(players):,} players from Sleeper")
        return players

    except Exception as e:
        logger.error(f"❌ Failed to fetch from Sleeper: {e}")
        return None


def process_injury_data(players):
    """
    Extract injury information from Sleeper player data.

    Sleeper injury status values:
    - Questionable (Q)
    - Doubtful (D)
    - Out (O)
    - IR (Injured Reserve)
    - PUP (Physically Unable to Perform)
    - COV (COVID-19 list)
    """
    injury_list = []

    for player_id, player_data in players.items():
        # Skip if not active NFL player
        if player_data.get('status') != 'Active':
            continue

        # Check for injury status
        injury_status = player_data.get('injury_status')

        if injury_status:
            injury_list.append({
                'player_id': player_id,
                'player_name': f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}".strip(),
                'team': player_data.get('team', ''),
                'position': player_data.get('position', ''),
                'injury_status': injury_status,
                'injury_body_part': player_data.get('injury_body_part', ''),
                'injury_start_date': player_data.get('injury_start_date', ''),
                'injury_notes': player_data.get('injury_notes', ''),
                'fantasy_positions': ','.join(player_data.get('fantasy_positions', []) or []),
                'years_exp': player_data.get('years_exp', 0),
                'last_updated': datetime.now().isoformat()
            })

    return pd.DataFrame(injury_list)


def map_injury_status_to_availability(status):
    """
    Map Sleeper injury status to practice participation / availability.

    Returns:
    - availability: Likely game status
    - practice_status: Expected practice participation
    """
    status_map = {
        'Questionable': {
            'availability': 'Questionable',
            'practice_status': 'Limited',
            'game_probability': 0.5
        },
        'Doubtful': {
            'availability': 'Doubtful',
            'practice_status': 'DNP',
            'game_probability': 0.25
        },
        'Out': {
            'availability': 'Out',
            'practice_status': 'DNP',
            'game_probability': 0.0
        },
        'IR': {
            'availability': 'Out (IR)',
            'practice_status': 'DNP',
            'game_probability': 0.0
        },
        'PUP': {
            'availability': 'Out (PUP)',
            'practice_status': 'DNP',
            'game_probability': 0.0
        },
        'COV': {
            'availability': 'Out (COVID)',
            'practice_status': 'DNP',
            'game_probability': 0.0
        }
    }

    return status_map.get(status, {
        'availability': status,
        'practice_status': 'Unknown',
        'game_probability': 0.5
    })


def generate_injury_report(injuries_df):
    """Generate formatted injury report."""

    if len(injuries_df) == 0:
        logger.warning("⚠️  No injuries found")
        return

    # Add availability mapping
    injuries_df['game_probability'] = injuries_df['injury_status'].apply(
        lambda x: map_injury_status_to_availability(x)['game_probability']
    )

    # Sort by team and status severity
    status_order = {'Out': 0, 'IR': 0, 'PUP': 0, 'Doubtful': 1, 'Questionable': 2, 'COV': 0}
    injuries_df['status_severity'] = injuries_df['injury_status'].map(status_order)
    injuries_df = injuries_df.sort_values(['team', 'status_severity', 'player_name'])

    print("\n" + "="*80)
    print("NFL INJURY REPORT (Sleeper API)")
    print("="*80)
    print(f"Timestamp: {datetime.now()}")
    print(f"Total Injured Players: {len(injuries_df)}")
    print("="*80)
    print()

    # Group by team
    for team, team_injuries in injuries_df.groupby('team'):
        if pd.isna(team) or team == '':
            continue

        print(f"\n{team}:")
        print("-" * 80)

        for _, player in team_injuries.iterrows():
            status_emoji = {
                'Out': '🔴',
                'IR': '🔴',
                'PUP': '🔴',
                'Doubtful': '🟡',
                'Questionable': '🟢',
                'COV': '🔴'
            }.get(player['injury_status'], '⚪')

            body_part = f" ({player['injury_body_part']})" if player['injury_body_part'] else ""

            print(f"  {status_emoji} {player['player_name']} ({player['position']}) - "
                  f"{player['injury_status']}{body_part}")

            if player['injury_notes']:
                print(f"     Note: {player['injury_notes']}")

    print("\n" + "="*80)
    print(f"\n📊 Status Breakdown:")
    status_counts = injuries_df['injury_status'].value_counts()
    for status, count in status_counts.items():
        print(f"   {status}: {count}")

    print("\n" + "="*80)
    print("⚠️  ALWAYS verify injury status before betting!")
    print("   Updates: Check Sleeper or official team reports")
    print("="*80)


def save_injury_data(injuries_df, output_dir='data/injuries'):
    """Save injury data to CSV for use in predictions."""

    if len(injuries_df) == 0:
        logger.warning("No injury data to save")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full injury report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_path = output_path / f'injuries_sleeper_{timestamp}.csv'
    injuries_df.to_csv(full_path, index=False)
    logger.info(f"💾 Saved full report: {full_path}")

    # Save latest (for predictions to use)
    latest_path = output_path / 'injuries_latest.csv'
    injuries_df.to_csv(latest_path, index=False)
    logger.info(f"💾 Saved latest: {latest_path}")

    # Update canonical current injuries file used across the pipeline
    current_path = output_path / 'current_injuries.csv'
    injuries_df.to_csv(current_path, index=False)
    logger.info(f"💾 Synced current injuries: {current_path}")

    # Create simple lookup format for model
    lookup_df = injuries_df[['player_name', 'team', 'position', 'injury_status', 'game_probability']].copy()
    lookup_path = output_path / 'injury_status_lookup.csv'
    lookup_df.to_csv(lookup_path, index=False)
    logger.info(f"💾 Saved lookup: {lookup_path}")

    return latest_path


def main():
    """Main execution."""

    # Fetch players from Sleeper
    players = fetch_sleeper_players()

    if not players:
        logger.error("Failed to fetch player data")
        return

    # Process injury data
    injuries_df = process_injury_data(players)

    # Generate report
    generate_injury_report(injuries_df)

    # Save data
    if len(injuries_df) > 0:
        save_injury_data(injuries_df)
    else:
        logger.warning("No injuries to save")

    logger.info("\n✅ Injury fetch complete!")


if __name__ == "__main__":
    main()
