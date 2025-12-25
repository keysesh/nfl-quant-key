"""
Causal Narrative Generator - Tells a COMPLETE, COHESIVE story.

Generates natural language narratives that explain WHY a pick was made,
incorporating all Monte Carlo simulation inputs and model factors.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


class CausalNarrativeGenerator:
    """Generates story-driven narratives showing the complete model reasoning."""

    def generate_html(self, row: pd.Series) -> str:
        """Generate a flowing narrative that tells the complete story."""

        # Extract core data
        player = str(row.get('player', row.get('nflverse_name', 'Player')))
        pick = str(row.get('pick', '')).upper()
        line = float(row.get('line', 0) or 0)
        trailing = float(row.get('trailing_stat', 0) or 0)
        projection = float(row.get('model_projection', trailing) or trailing)
        model_prob = float(row.get('model_prob', 0.5) or 0.5)
        market = str(row.get('market', ''))
        market_display = str(row.get('market_display', market))
        opponent = str(row.get('opponent', '') or '')
        position = str(row.get('position', '') or '')
        team = str(row.get('team', '') or '')

        # Calculate metrics
        edge_pct = ((projection - line) / line * 100) if line > 0 else 0
        lvt = float(row.get('line_vs_trailing', 0) or 0)
        hist_over_rate = float(row.get('hist_over_rate', 0) or 0)
        hist_count = int(row.get('hist_count', 0) or 0)

        # Determine market type
        is_receiving = 'recep' in market.lower() or 'receiving' in market.lower()
        is_rushing = 'rush' in market.lower()
        is_passing = 'pass' in market.lower()

        # Get all the MC simulation inputs
        snap_share = float(row.get('snap_share', 0) or 0)
        targets = float(row.get('targets_mean', 0) or 0)
        receptions = float(row.get('receptions_mean', 0) or 0)
        rec_yards = float(row.get('receiving_yards_mean', 0) or 0)
        rec_std = float(row.get('receiving_yards_std', 0) or 0)
        rush_att = float(row.get('rushing_attempts_mean', 0) or 0)
        rush_yards = float(row.get('rushing_yards_mean', 0) or 0)
        rush_std = float(row.get('rushing_yards_std', 0) or 0)
        pass_att = float(row.get('passing_attempts_mean', 0) or 0)
        pass_yards = float(row.get('passing_yards_mean', 0) or 0)
        completions = float(row.get('passing_completions_mean', 0) or 0)

        # Context factors
        def_epa = float(row.get('opponent_def_epa', 0) or 0)
        team_pass_att = float(row.get('team_pass_attempts', 0) or 0)
        team_rush_att = float(row.get('team_rush_attempts', 0) or 0)
        rz_target_share = float(row.get('redzone_target_share', 0) or 0)
        rz_carry_share = float(row.get('redzone_carry_share', 0) or 0)
        gl_carry_share = float(row.get('goalline_carry_share', 0) or 0)
        game_script = float(row.get('game_script_dynamic', 0) or 0)

        # Game info
        is_primetime = row.get('is_primetime_game', False)
        primetime_type = str(row.get('primetime_type', '') or '')
        is_home = str(row.get('team', '')) in str(row.get('game', '').split('@')[1] if '@' in str(row.get('game', '')) else '')

        # Injury context
        qb_status = str(row.get('injury_qb_status', '') or '')
        player_status = str(row.get('severity', '') or '')

        # Build the narrative
        paragraphs = []

        # === PARAGRAPH 1: THE SETUP ===
        setup = self._build_setup_paragraph(
            player, position, team, opponent, market_display, line, trailing,
            hist_over_rate, hist_count, pick, is_primetime, primetime_type
        )
        paragraphs.append(setup)

        # === PARAGRAPH 2: THE VOLUME STORY ===
        if is_receiving:
            volume = self._build_receiving_volume_paragraph(
                player, targets, receptions, snap_share, team_pass_att, rz_target_share
            )
        elif is_rushing:
            volume = self._build_rushing_volume_paragraph(
                player, rush_att, snap_share, team_rush_att, rz_carry_share, gl_carry_share
            )
        elif is_passing:
            volume = self._build_passing_volume_paragraph(
                player, pass_att, completions, snap_share
            )
        else:
            volume = ""
        if volume:
            paragraphs.append(volume)

        # === PARAGRAPH 3: THE EFFICIENCY STORY ===
        if is_receiving and targets > 0:
            efficiency = self._build_receiving_efficiency_paragraph(
                player, rec_yards, targets, rec_std, line
            )
        elif is_rushing and rush_att > 0:
            efficiency = self._build_rushing_efficiency_paragraph(
                player, rush_yards, rush_att, rush_std, line
            )
        elif is_passing and pass_att > 0:
            efficiency = self._build_passing_efficiency_paragraph(
                player, pass_yards, completions, pass_att, line
            )
        else:
            efficiency = ""
        if efficiency:
            paragraphs.append(efficiency)

        # === PARAGRAPH 4: THE MATCHUP STORY ===
        matchup = self._build_matchup_paragraph(
            opponent, def_epa, game_script, is_receiving, is_rushing, is_passing
        )
        if matchup:
            paragraphs.append(matchup)

        # === PARAGRAPH 5: THE PROJECTION & VERDICT ===
        verdict = self._build_verdict_paragraph(
            player, projection, line, pick, edge_pct, model_prob, trailing
        )
        paragraphs.append(verdict)

        # Build the final HTML
        narrative_html = ''.join([f'<p class="narrative-paragraph">{p}</p>' for p in paragraphs if p])

        # Add the key stats bar
        stats_bar = self._build_stats_bar(
            projection, line, model_prob, edge_pct, pick, trailing, snap_share,
            targets if is_receiving else rush_att if is_rushing else pass_att
        )

        return f'''
        <div class="narrative-story">
            <div class="narrative-content">
                {narrative_html}
            </div>
            {stats_bar}
        </div>
        '''

    def _build_setup_paragraph(self, player, position, team, opponent, market_display,
                                line, trailing, hist_over_rate, hist_count, pick,
                                is_primetime, primetime_type) -> str:
        """Opening paragraph setting up the bet context."""

        # Line vs trailing context
        if trailing > 0:
            line_diff = line - trailing
            if line_diff > 2:
                line_context = f"set {line_diff:.1f} points above his recent average of {trailing:.1f}"
            elif line_diff < -2:
                line_context = f"set {abs(line_diff):.1f} points below his recent average of {trailing:.1f}"
            else:
                line_context = f"right at his recent average of {trailing:.1f}"
        else:
            line_context = "without much recent history to compare"

        # Historical tendency
        if hist_count >= 5:
            under_rate = 1 - hist_over_rate
            if hist_over_rate >= 0.60:
                hist_context = f"He's cleared this number in {hist_over_rate:.0%} of his last {hist_count} games"
            elif under_rate >= 0.60:
                hist_context = f"He's gone under similar lines in {under_rate:.0%} of his last {hist_count} games"
            else:
                hist_context = f"He's been roughly 50/50 on similar lines over his last {hist_count} games"
        else:
            hist_context = ""

        # Primetime context
        if is_primetime and primetime_type:
            game_context = f" in tonight's {primetime_type} showdown"
        else:
            game_context = ""

        # Build the paragraph
        parts = [
            f"<strong>{player}</strong> ({position}, {team}) faces a <strong>{market_display}</strong> line of <strong>{line}</strong>{game_context} against {opponent}.",
            f"The line is {line_context}."
        ]

        if hist_context:
            parts.append(hist_context + ".")

        return " ".join(parts)

    def _build_receiving_volume_paragraph(self, player, targets, receptions, snap_share,
                                           team_pass_att, rz_target_share) -> str:
        """Build the receiving volume story."""
        parts = []

        # Snap share context
        if snap_share >= 0.90:
            parts.append(f"As a {snap_share:.0%} snap share player, {player.split()[0]} is a true workhorse in this offense.")
        elif snap_share >= 0.75:
            parts.append(f"Playing on {snap_share:.0%} of snaps, he's firmly entrenched as a primary option.")
        elif snap_share >= 0.60:
            parts.append(f"With a {snap_share:.0%} snap share, he's in a rotational role but gets consistent work.")
        elif snap_share > 0:
            parts.append(f"His {snap_share:.0%} snap share indicates a limited or situational role.")

        # Target volume
        if targets >= 10:
            parts.append(f"Our simulation expects roughly <strong>{targets:.1f} targets</strong> — elite volume that creates a high floor.")
        elif targets >= 7:
            parts.append(f"We project approximately <strong>{targets:.1f} targets</strong>, solid volume for consistent production.")
        elif targets >= 5:
            parts.append(f"The model expects around <strong>{targets:.1f} targets</strong>, moderate volume that makes outcomes variable.")
        elif targets > 0:
            parts.append(f"With only <strong>{targets:.1f} projected targets</strong>, opportunity is a concern.")

        # Catch rate
        if targets > 0 and receptions > 0:
            catch_rate = receptions / targets
            if catch_rate >= 0.75:
                parts.append(f"He's catching {catch_rate:.0%} of his targets, showing reliable hands.")
            elif catch_rate < 0.55:
                parts.append(f"His {catch_rate:.0%} catch rate introduces volatility.")

        # Red zone role
        if rz_target_share > 0.20:
            parts.append(f"He commands {rz_target_share:.0%} of red zone targets, making him a primary scoring threat.")
        elif rz_target_share > 0.10:
            parts.append(f"He sees {rz_target_share:.0%} of red zone targets — involved but not dominant.")

        # Team context
        if team_pass_att >= 38:
            parts.append(f"The team's pass-heavy approach ({team_pass_att:.0f} attempts/game) supports volume.")
        elif team_pass_att <= 30:
            parts.append(f"A run-oriented offense ({team_pass_att:.0f} pass attempts/game) limits upside.")

        return " ".join(parts)

    def _build_rushing_volume_paragraph(self, player, rush_att, snap_share, team_rush_att,
                                         rz_carry_share, gl_carry_share) -> str:
        """Build the rushing volume story."""
        parts = []

        # Carry volume
        if rush_att >= 18:
            parts.append(f"As a workhorse back with <strong>{rush_att:.1f} carries</strong> per game projected, {player.split()[0]} is the clear lead runner.")
        elif rush_att >= 12:
            parts.append(f"Projected for <strong>{rush_att:.1f} carries</strong>, he's the primary back in this offense.")
        elif rush_att >= 8:
            parts.append(f"With <strong>{rush_att:.1f} expected carries</strong>, he leads a committee but faces competition.")
        elif rush_att > 0:
            parts.append(f"Only <strong>{rush_att:.1f} projected carries</strong> limits his workload significantly.")

        # Snap share
        if snap_share >= 0.70:
            parts.append(f"His {snap_share:.0%} snap share shows three-down usage.")
        elif snap_share >= 0.50:
            parts.append(f"A {snap_share:.0%} snap share suggests early-down work with some passing down involvement.")
        elif snap_share > 0:
            parts.append(f"Limited to {snap_share:.0%} of snaps — likely a change-of-pace role.")

        # Scoring roles
        if gl_carry_share > 0.40:
            parts.append(f"He dominates goal-line work ({gl_carry_share:.0%} of carries inside the 5), which is TD-positive.")
        if rz_carry_share > 0.30:
            parts.append(f"With {rz_carry_share:.0%} of red zone carries, he's the short-yardage option.")

        return " ".join(parts)

    def _build_passing_volume_paragraph(self, player, pass_att, completions, snap_share) -> str:
        """Build the passing volume story."""
        parts = []

        if pass_att >= 38:
            parts.append(f"In a pass-heavy scheme, we project <strong>{pass_att:.1f} attempts</strong> — elite volume.")
        elif pass_att >= 32:
            parts.append(f"The model expects around <strong>{pass_att:.1f} pass attempts</strong>, solid volume.")
        elif pass_att >= 25:
            parts.append(f"With <strong>{pass_att:.1f} projected attempts</strong>, the offense is balanced.")
        elif pass_att > 0:
            parts.append(f"Only <strong>{pass_att:.1f} expected attempts</strong> suggests a run-heavy game script.")

        if completions > 0 and pass_att > 0:
            comp_pct = completions / pass_att
            if comp_pct >= 0.68:
                parts.append(f"His {comp_pct:.0%} completion rate shows accuracy and quick-game efficiency.")
            elif comp_pct < 0.60:
                parts.append(f"A {comp_pct:.0%} completion rate introduces volatility to yardage totals.")

        return " ".join(parts)

    def _build_receiving_efficiency_paragraph(self, player, rec_yards, targets, rec_std, line) -> str:
        """Build the receiving efficiency story."""
        if targets <= 0:
            return ""

        ypt = rec_yards / targets
        cv = rec_std / rec_yards if rec_yards > 0 else 0.5

        parts = []

        # Y/T context
        if ypt >= 12:
            parts.append(f"His <strong>{ypt:.1f} yards per target</strong> indicates a deep threat role — high upside but variable.")
        elif ypt >= 9:
            parts.append(f"At <strong>{ypt:.1f} yards per target</strong>, he's an efficient intermediate option.")
        elif ypt >= 7:
            parts.append(f"His <strong>{ypt:.1f} yards per target</strong> reflects a possession receiver profile — steady but limited ceiling.")
        elif ypt > 0:
            parts.append(f"A low <strong>{ypt:.1f} yards per target</strong> suggests short routes or YAC-dependent production.")

        # Variance context
        if cv > 0.7:
            parts.append(f"High variance (±{rec_std:.0f} yards) means outcomes swing widely game-to-game.")
        elif cv < 0.4:
            parts.append(f"Low variance (±{rec_std:.0f} yards) provides consistency around the projection.")

        # Math breakdown
        parts.append(f"<span class='math-callout'>The math: {targets:.1f} targets × {ypt:.1f} Y/T = <strong>{rec_yards:.1f}</strong> projected yards</span>")

        return " ".join(parts)

    def _build_rushing_efficiency_paragraph(self, player, rush_yards, rush_att, rush_std, line) -> str:
        """Build the rushing efficiency story."""
        if rush_att <= 0:
            return ""

        ypc = rush_yards / rush_att

        parts = []

        if ypc >= 5.0:
            parts.append(f"Averaging <strong>{ypc:.1f} yards per carry</strong>, he's been explosive and could break a big one.")
        elif ypc >= 4.2:
            parts.append(f"At <strong>{ypc:.1f} YPC</strong>, he's running efficiently behind this line.")
        elif ypc >= 3.5:
            parts.append(f"His <strong>{ypc:.1f} YPC</strong> is workmanlike — he'll need volume to produce.")
        elif ypc > 0:
            parts.append(f"Struggling at <strong>{ypc:.1f} YPC</strong> — tough sledding so far.")

        parts.append(f"<span class='math-callout'>The math: {rush_att:.1f} carries × {ypc:.1f} YPC = <strong>{rush_yards:.1f}</strong> projected yards</span>")

        return " ".join(parts)

    def _build_passing_efficiency_paragraph(self, player, pass_yards, completions, pass_att, line) -> str:
        """Build the passing efficiency story."""
        if completions <= 0:
            return ""

        ypc = pass_yards / completions

        parts = []

        if ypc >= 12:
            parts.append(f"His <strong>{ypc:.1f} yards per completion</strong> shows a downfield attack.")
        elif ypc >= 10:
            parts.append(f"At <strong>{ypc:.1f} Y/C</strong>, it's a balanced passing scheme.")
        elif ypc > 0:
            parts.append(f"A short <strong>{ypc:.1f} Y/C</strong> indicates a dink-and-dunk approach.")

        parts.append(f"<span class='math-callout'>The math: {completions:.1f} completions × {ypc:.1f} Y/C = <strong>{pass_yards:.1f}</strong> projected yards</span>")

        return " ".join(parts)

    def _build_matchup_paragraph(self, opponent, def_epa, game_script,
                                  is_receiving, is_rushing, is_passing) -> str:
        """Build the matchup context story."""
        parts = []

        # Defense strength
        if def_epa > 0.05:
            parts.append(f"The matchup is favorable — {opponent}'s defense ranks as a bottom-tier unit (EPA: {def_epa:+.2f}), prone to giving up yards.")
        elif def_epa > 0.02:
            parts.append(f"{opponent}'s defense is below average (EPA: {def_epa:+.2f}), a soft matchup.")
        elif def_epa < -0.08:
            parts.append(f"This is a tough draw — {opponent} fields an elite defense (EPA: {def_epa:.2f}) that limits production.")
        elif def_epa < -0.04:
            parts.append(f"{opponent}'s defense is above average (EPA: {def_epa:.2f}), creating a tougher path to production.")
        else:
            parts.append(f"{opponent}'s defense grades as league average, a neutral matchup factor.")

        # Game script
        if game_script < -0.15:
            if is_passing or is_receiving:
                parts.append("Projected to trail suggests more passing volume — a boost for receiving props.")
            elif is_rushing:
                parts.append("A likely negative game script could limit rushing attempts as the team chases points.")
        elif game_script > 0.15:
            if is_rushing:
                parts.append("Expected to lead should mean more rushing attempts to milk the clock.")
            elif is_passing:
                parts.append("A projected lead could reduce passing volume as the team runs more.")

        return " ".join(parts)

    def _build_verdict_paragraph(self, player, projection, line, pick, edge_pct,
                                  model_prob, trailing) -> str:
        """Build the final verdict and recommendation."""

        parts = []

        # Projection vs line
        diff = projection - line
        if projection > 0 and abs(diff) > 0.1:
            if pick == 'OVER':
                parts.append(f"The model projects <strong>{projection:.1f}</strong> — {abs(diff):.1f} points above the {line} line.")
            else:
                parts.append(f"The model projects <strong>{projection:.1f}</strong> — {abs(diff):.1f} points below the {line} line.")

        # Confidence explanation
        if model_prob >= 0.70:
            parts.append(f"At {model_prob:.0%} confidence, this is a high-conviction play.")
        elif model_prob >= 0.60:
            parts.append(f"The {model_prob:.0%} confidence represents a solid edge worth capturing.")
        else:
            parts.append(f"At {model_prob:.0%}, confidence is moderate — consider sizing accordingly.")

        # Edge context
        if abs(edge_pct) >= 15:
            parts.append(f"The <strong>{edge_pct:+.0f}%</strong> edge is substantial — a clear value opportunity.")
        elif abs(edge_pct) >= 8:
            parts.append(f"An <strong>{edge_pct:+.0f}%</strong> edge provides meaningful value.")
        elif abs(edge_pct) >= 5:
            parts.append(f"The <strong>{edge_pct:+.0f}%</strong> edge is thin but playable.")

        return " ".join(parts)

    def _build_stats_bar(self, projection, line, model_prob, edge_pct, pick, trailing,
                          snap_share, volume_metric) -> str:
        """Build the bottom stats bar."""

        pick_class = 'pick-over' if pick == 'OVER' else 'pick-under'
        prob_class = 'high' if model_prob >= 0.70 else 'medium' if model_prob >= 0.60 else 'low'
        edge_class = 'positive' if edge_pct > 0 else 'negative'

        return f'''
        <div class="stats-bar">
            <div class="stat-item">
                <span class="stat-label">Projection</span>
                <span class="stat-value projection">{projection:.1f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Line</span>
                <span class="stat-value">{line}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Recent Avg</span>
                <span class="stat-value">{trailing:.1f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Snap%</span>
                <span class="stat-value">{snap_share:.0%}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Volume</span>
                <span class="stat-value">{volume_metric:.1f}</span>
            </div>
            <div class="stat-item verdict">
                <span class="pick-badge {pick_class}">{pick}</span>
                <span class="prob {prob_class}">{model_prob:.0%}</span>
                <span class="edge {edge_class}">{edge_pct:+.0f}%</span>
            </div>
        </div>
        '''


def build_edge_narrative(row: pd.Series) -> str:
    """Build narrative for edge-based picks (LVT, Player Bias, TD Poisson, Game Lines)."""
    source = str(row.get('source', '')).upper()
    player = str(row.get('player', 'Unknown'))
    market = str(row.get('market', ''))
    market_display = str(row.get('market_display', market))
    line = float(row.get('line', 0) or 0)
    pick = str(row.get('pick', row.get('direction', ''))).upper()
    confidence = float(row.get('combined_confidence', row.get('model_prob', 0.5)) or 0.5)
    reasoning = str(row.get('reasoning', '') or '')
    team = str(row.get('team', '') or '')

    paragraphs = []

    # Game Line picks have different structure
    if 'GAME_LINE' in source:
        game = str(row.get('game', player))
        bet_type = str(row.get('bet_type', 'spread'))
        edge_pct = float(row.get('edge_pct', 0) or 0)

        if bet_type == 'spread':
            home_team = game.split(' @ ')[1] if ' @ ' in game else game
            away_team = game.split(' @ ')[0] if ' @ ' in game else ''
            direction = str(row.get('direction', '')).upper()

            setup = f"<strong>{game}</strong> — The model identifies value on the <strong>{direction}</strong> side of this spread."

            if direction == 'HOME':
                analysis = f"EPA-based power ratings suggest {home_team} is undervalued by the market. "
            else:
                analysis = f"EPA-based power ratings suggest {away_team} is undervalued by the market. "

            # Add rest/divisional context from reasoning
            if 'Rest:' in reasoning:
                analysis += reasoning.split('|')[1].strip() + ". " if '|' in reasoning else ""
            if 'Divisional' in reasoning:
                analysis += "As a divisional game, spreads tend to be closer than expected."

            paragraphs.append(setup)
            paragraphs.append(analysis)
            paragraphs.append(f"Edge: <strong>{edge_pct:.1f} points</strong> toward {direction}. Confidence: <strong>{confidence:.0%}</strong>.")

        else:  # total
            setup = f"<strong>{game}</strong> — The model projects the total to go <strong>{pick}</strong> {line}."
            lvt_edge = float(row.get('edge_pct', 0) or 0)

            if pick == 'OVER':
                analysis = f"Combined team scoring projections suggest this total is set {abs(lvt_edge):.1f}% too low."
            else:
                analysis = f"Combined team scoring projections suggest this total is set {abs(lvt_edge):.1f}% too high."

            paragraphs.append(setup)
            paragraphs.append(analysis)
            paragraphs.append(f"LVT Edge: <strong>{abs(lvt_edge):.1f}%</strong>. Confidence: <strong>{confidence:.0%}</strong>.")

    # TD Poisson picks
    elif 'TD_POISSON' in source or 'POISSON' in source:
        expected_tds = float(row.get('expected_tds', 0) or 0)
        p_over = float(row.get('p_over', 0) or 0)
        p_under = float(row.get('p_under', 0) or 0)

        setup = f"<strong>{player}</strong> ({team}) — {market_display} line of <strong>{line}</strong>."

        if expected_tds > 0:
            analysis = f"Poisson regression projects <strong>{expected_tds:.2f} TDs</strong> based on historical rates and matchup factors. "
            if pick == 'OVER':
                analysis += f"P(Over {line}): <strong>{p_over:.0%}</strong>."
            else:
                analysis += f"P(Under {line}): <strong>{p_under:.0%}</strong>."
        else:
            analysis = reasoning

        paragraphs.append(setup)
        paragraphs.append(analysis)
        paragraphs.append(f"Confidence: <strong>{confidence:.0%}</strong>.")

    # LVT or Player Bias picks
    elif 'LVT' in source or 'PLAYER_BIAS' in source or 'ENSEMBLE' in source:
        lvt_conf = float(row.get('lvt_confidence', 0) or 0)
        bias_conf = float(row.get('player_bias_confidence', 0) or 0)
        under_rate = float(row.get('player_under_rate', 0.5) or 0.5)
        bet_count = int(row.get('player_bet_count', 0) or 0)
        trailing = float(row.get('trailing_stat', 0) or 0)

        setup = f"<strong>{player}</strong> ({team}) — {market_display} line of <strong>{line}</strong>."
        paragraphs.append(setup)

        # LVT analysis
        if lvt_conf > 0.5:
            lvt_pct = (lvt_conf - 0.5) * 200  # Convert to percentage edge
            if trailing > 0:
                diff = line - trailing
                if diff > 0:
                    paragraphs.append(f"Line is set {diff:.1f} above recent average ({trailing:.1f}). LVT edge: <strong>{lvt_pct:.0f}%</strong> toward {pick}.")
                else:
                    paragraphs.append(f"Line is set {abs(diff):.1f} below recent average ({trailing:.1f}). LVT edge: <strong>{lvt_pct:.0f}%</strong> toward {pick}.")

        # Player Bias analysis
        if bias_conf > 0.5 and bet_count >= 5:
            if pick == 'UNDER':
                paragraphs.append(f"Historical tendency: goes UNDER in <strong>{under_rate:.0%}</strong> of games ({bet_count} samples).")
            else:
                over_rate = 1 - under_rate
                paragraphs.append(f"Historical tendency: goes OVER in <strong>{over_rate:.0%}</strong> of games ({bet_count} samples).")

        paragraphs.append(f"Combined confidence: <strong>{confidence:.0%}</strong>.")

    else:
        # Fallback for unknown sources
        paragraphs.append(f"<strong>{player}</strong> — {market_display} {pick} {line}.")
        if reasoning:
            paragraphs.append(reasoning)
        paragraphs.append(f"Confidence: <strong>{confidence:.0%}</strong>.")

    # Build HTML
    narrative_html = ''.join([f'<p class="narrative-paragraph">{p}</p>' for p in paragraphs])

    return f'''
    <div class="narrative-story edge-narrative">
        <div class="narrative-content">
            {narrative_html}
        </div>
    </div>
    '''


def build_model_narrative(row: pd.Series, pick: str) -> str:
    """Build causal narrative HTML for dashboard."""
    try:
        # Check if this is an edge-based pick
        source = str(row.get('source', '')).upper()
        if any(x in source for x in ['LVT', 'PLAYER_BIAS', 'POISSON', 'GAME_LINE', 'TD_']):
            return build_edge_narrative(row)

        # Fall back to MC-based narrative for old model
        generator = CausalNarrativeGenerator()
        return generator.generate_html(row)
    except Exception as e:
        return f'<div class="narrative-story error">Narrative error: {str(e)[:100]}</div>'


def _highlight_narrative(text: str, pick: str) -> str:
    """
    Add highlight spans to key values in narrative text.

    - Percentages get .hl-pct (green for high confidence)
    - Key stats/numbers get .hl-stat (accent color)
    - Pick direction gets .hl-pick with over/under class
    - Edge values get .hl-edge with positive/negative class
    """
    import re

    # Highlight percentages (e.g., "79%", "85%")
    text = re.sub(
        r'(\d+\.?\d*%)',
        r'<span class="hl-pct">\1</span>',
        text
    )

    # Highlight edge values (e.g., "+25%", "-19%")
    text = re.sub(
        r'(\+\d+\.?\d*%)',
        r'<span class="hl-edge positive">\1</span>',
        text
    )
    text = re.sub(
        r'(-\d+\.?\d*%)',
        r'<span class="hl-edge negative">\1</span>',
        text
    )

    # Highlight key projection/line numbers in context
    # "projects 44.4" or "projection of 35.5"
    text = re.sub(
        r'projects? (\d+\.?\d*)',
        r'projects <span class="hl-stat">\1</span>',
        text
    )

    # Highlight "line of X"
    text = re.sub(
        r'line of (\d+\.?\d*)',
        r'line of <span class="hl-stat">\1</span>',
        text
    )

    # Highlight "average of X"
    text = re.sub(
        r'average of (\d+\.?\d*)',
        r'average of <span class="hl-stat">\1</span>',
        text
    )

    # Highlight targets/carries/attempts numbers
    text = re.sub(
        r'(\d+\.?\d*) targets',
        r'<span class="hl-stat">\1</span> targets',
        text
    )
    text = re.sub(
        r'(\d+\.?\d*) carries',
        r'<span class="hl-stat">\1</span> carries',
        text
    )
    text = re.sub(
        r'(\d+\.?\d*) (expected )?attempts',
        r'<span class="hl-stat">\1</span> \2attempts',
        text
    )

    # Highlight yards per target/carry
    text = re.sub(
        r'(\d+\.?\d*) yards per',
        r'<span class="hl-stat">\1</span> yards per',
        text
    )
    text = re.sub(
        r'(\d+\.?\d*) YPC',
        r'<span class="hl-stat">\1</span> YPC',
        text
    )
    text = re.sub(
        r'(\d+\.?\d*) Y/T',
        r'<span class="hl-stat">\1</span> Y/T',
        text
    )
    text = re.sub(
        r'(\d+\.?\d*) Y/C',
        r'<span class="hl-stat">\1</span> Y/C',
        text
    )

    # Highlight "projected yards" result
    text = re.sub(
        r'= (\d+\.?\d*) projected',
        r'= <span class="hl-stat hl-result">\1</span> projected',
        text
    )

    # Highlight confidence levels
    text = re.sub(
        r'high-conviction',
        r'<span class="hl-confidence high">high-conviction</span>',
        text
    )
    text = re.sub(
        r'solid edge',
        r'<span class="hl-confidence medium">solid edge</span>',
        text
    )

    # Highlight pick direction in verdict
    pick_upper = pick.upper()
    pick_class = 'over' if pick_upper == 'OVER' else 'under'

    return text


def generate_prose_only(row: pd.Series, styled: bool = True) -> str:
    """
    Generate ONLY the narrative prose paragraphs without the stats bar.

    Args:
        row: DataFrame row with pick data
        styled: If True, add highlight spans for visual styling

    Returns:
        HTML with paragraph breaks and optional highlight spans.
        This is used for modal/expansion displays where stats are already shown in cards.
    """
    try:
        generator = CausalNarrativeGenerator()

        # Extract core data (same as generate_html)
        player = str(row.get('player', row.get('nflverse_name', 'Player')))
        pick = str(row.get('pick', '')).upper()
        line = float(row.get('line', 0) or 0)
        trailing = float(row.get('trailing_stat', 0) or 0)
        projection = float(row.get('model_projection', trailing) or trailing)
        model_prob = float(row.get('model_prob', 0.5) or 0.5)
        market = str(row.get('market', ''))
        market_display = str(row.get('market_display', market))
        opponent = str(row.get('opponent', '') or '')
        position = str(row.get('position', '') or '')
        team = str(row.get('team', '') or '')

        # Calculate metrics
        edge_pct = ((projection - line) / line * 100) if line > 0 else 0
        lvt = float(row.get('line_vs_trailing', 0) or 0)
        hist_over_rate = float(row.get('hist_over_rate', 0) or 0)
        hist_count = int(row.get('hist_count', 0) or 0)

        # Determine market type
        is_receiving = 'recep' in market.lower() or 'receiving' in market.lower()
        is_rushing = 'rush' in market.lower()
        is_passing = 'pass' in market.lower()

        # Get MC simulation inputs
        snap_share = float(row.get('snap_share', 0) or 0)
        targets = float(row.get('targets_mean', 0) or 0)
        receptions = float(row.get('receptions_mean', 0) or 0)
        rec_yards = float(row.get('receiving_yards_mean', 0) or 0)
        rec_std = float(row.get('receiving_yards_std', 0) or 0)
        rush_att = float(row.get('rushing_attempts_mean', 0) or 0)
        rush_yards = float(row.get('rushing_yards_mean', 0) or 0)
        rush_std = float(row.get('rushing_yards_std', 0) or 0)
        pass_att = float(row.get('passing_attempts_mean', 0) or 0)
        pass_yards = float(row.get('passing_yards_mean', 0) or 0)
        completions = float(row.get('passing_completions_mean', 0) or 0)

        # Context factors
        def_epa = float(row.get('opponent_def_epa', 0) or 0)
        team_pass_att = float(row.get('team_pass_attempts', 0) or 0)
        team_rush_att = float(row.get('team_rush_attempts', 0) or 0)
        rz_target_share = float(row.get('redzone_target_share', 0) or 0)
        rz_carry_share = float(row.get('redzone_carry_share', 0) or 0)
        gl_carry_share = float(row.get('goalline_carry_share', 0) or 0)
        game_script = float(row.get('game_script_dynamic', 0) or 0)

        # Game info
        is_primetime = row.get('is_primetime_game', False)
        primetime_type = str(row.get('primetime_type', '') or '')

        # Build paragraphs (same logic as generate_html, but no stats bar)
        paragraphs = []

        # Setup paragraph
        setup = generator._build_setup_paragraph(
            player, position, team, opponent, market_display, line, trailing,
            hist_over_rate, hist_count, pick, is_primetime, primetime_type
        )
        paragraphs.append(('setup', setup))

        # Volume paragraph
        if is_receiving:
            volume = generator._build_receiving_volume_paragraph(
                player, targets, receptions, snap_share, team_pass_att, rz_target_share
            )
        elif is_rushing:
            volume = generator._build_rushing_volume_paragraph(
                player, rush_att, snap_share, team_rush_att, rz_carry_share, gl_carry_share
            )
        elif is_passing:
            volume = generator._build_passing_volume_paragraph(
                player, pass_att, completions, snap_share
            )
        else:
            volume = ""
        if volume:
            paragraphs.append(('volume', volume))

        # Efficiency paragraph
        if is_receiving and targets > 0:
            efficiency = generator._build_receiving_efficiency_paragraph(
                player, rec_yards, targets, rec_std, line
            )
        elif is_rushing and rush_att > 0:
            efficiency = generator._build_rushing_efficiency_paragraph(
                player, rush_yards, rush_att, rush_std, line
            )
        elif is_passing and pass_att > 0:
            efficiency = generator._build_passing_efficiency_paragraph(
                player, pass_yards, completions, pass_att, line
            )
        else:
            efficiency = ""
        if efficiency:
            paragraphs.append(('efficiency', efficiency))

        # Matchup paragraph
        matchup = generator._build_matchup_paragraph(
            opponent, def_epa, game_script, is_receiving, is_rushing, is_passing
        )
        if matchup:
            paragraphs.append(('matchup', matchup))

        # Verdict paragraph
        verdict = generator._build_verdict_paragraph(
            player, projection, line, pick, edge_pct, model_prob, trailing
        )
        paragraphs.append(('verdict', verdict))

        # Process paragraphs
        import re
        html_parts = []
        for i, (para_type, p) in enumerate(paragraphs):
            if p:
                # Remove existing HTML tags but preserve content
                clean = re.sub(r'<[^>]+>', '', p).strip()

                if styled:
                    # Add highlight spans
                    clean = _highlight_narrative(clean, pick)

                    # Wrap in paragraph div with type class
                    is_last = (i == len(paragraphs) - 1)
                    para_class = f"narrative-para {para_type}" + (" verdict-para" if is_last else "")
                    html_parts.append(f'<div class="{para_class}">{clean}</div>')
                else:
                    html_parts.append(clean)

        # Join paragraphs
        if styled:
            return ''.join(html_parts)
        else:
            return '<br><br>'.join(html_parts)

    except Exception as e:
        return f'Narrative error: {str(e)[:100]}'


def generate_causal_narrative(row: pd.Series):
    """Compatibility wrapper."""
    generator = CausalNarrativeGenerator()
    return generator.generate_html(row)


# CSS for narrative
CAUSAL_NARRATIVE_CSS = '''
/* Story-driven Narrative Styles */
.narrative-story {
    margin-top: 16px;
    background: var(--bg-tertiary);
    border-radius: 12px;
    overflow: hidden;
}

.narrative-content {
    padding: 20px;
    font-size: 13px;
    line-height: 1.7;
    color: var(--text-primary);
}

.narrative-paragraph {
    margin-bottom: 14px;
}

.narrative-paragraph:last-child {
    margin-bottom: 0;
}

.narrative-paragraph strong {
    color: #fff;
    font-weight: 600;
}

.math-callout {
    display: block;
    margin-top: 8px;
    padding: 10px 14px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.08));
    border-left: 3px solid #6366f1;
    border-radius: 0 6px 6px 0;
    font-family: 'SF Mono', monospace;
    font-size: 12px;
    color: #a5b4fc;
}

.math-callout strong {
    color: #6366f1;
    font-size: 14px;
}

/* Stats Bar */
.stats-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: rgba(0, 0, 0, 0.3);
    border-top: 1px solid var(--border-color);
    gap: 16px;
    flex-wrap: wrap;
}

.stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
}

.stat-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
}

.stat-value {
    font-size: 15px;
    font-weight: 700;
    font-family: 'SF Mono', monospace;
    color: var(--text-primary);
}

.stat-value.projection {
    color: #6366f1;
    font-size: 18px;
}

.stat-item.verdict {
    flex-direction: row;
    gap: 10px;
    margin-left: auto;
    padding-left: 20px;
    border-left: 1px solid var(--border-color);
}

.pick-badge {
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 13px;
    letter-spacing: 0.5px;
}

.pick-badge.pick-over {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1));
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.pick-badge.pick-under {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.2), rgba(244, 63, 94, 0.1));
    color: #f43f5e;
    border: 1px solid rgba(244, 63, 94, 0.3);
}

.prob {
    font-weight: 600;
    font-size: 14px;
}

.prob.high { color: #10b981; }
.prob.medium { color: #f59e0b; }
.prob.low { color: var(--text-muted); }

.edge {
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 13px;
    font-family: 'SF Mono', monospace;
}

.edge.positive {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
}

.edge.negative {
    background: rgba(244, 63, 94, 0.15);
    color: #f43f5e;
}

/* Mobile responsive */
@media (max-width: 768px) {
    .stats-bar {
        flex-direction: column;
        gap: 12px;
    }

    .stat-item.verdict {
        margin-left: 0;
        padding-left: 0;
        border-left: none;
        padding-top: 12px;
        border-top: 1px solid var(--border-color);
        width: 100%;
        justify-content: center;
    }

    .narrative-content {
        padding: 16px;
        font-size: 12px;
    }
}
'''
