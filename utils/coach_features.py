import pandas as pd


def build_coach_stats(kg_compact, kg_coaches, kg_seeds, kg_teams):
    """
    Build per-coach career NCAA tournament stats up to (but not including) each season.

    Parameters are expected to be the Kaggle-style NCAA tournament tables:
    - kg_compact: historical tournament game results (with DayNum, WTeamID/LTeamID)
    - kg_coaches: coach history table with Season/TeamID/LastDayNum/CoachName

    Notes
    -----
    We intentionally use only information from *earlier seasons* when generating
    a coach's season-level career features to avoid leakage.
    """
    if kg_coaches is None:
        print("[SKIP] MTeamCoaches.csv not available; coach features will be absent.")
        return None

    # ── 1. Identify head coach for each team-season ────────────────────────────
    # MTeamCoaches can list multiple coaches per team per year (mid-season change).
    # We pick the coach who was there through the end of the season (largest LastDayNum).
    coaches = kg_coaches.copy()
    coaches.columns = [c.strip() for c in coaches.columns]
    coaches = (
        coaches
        .sort_values(['Season', 'TeamID', 'LastDayNum'], ascending=[True, True, False])
        .drop_duplicates(subset=['Season', 'TeamID'], keep='first')
        [['Season', 'TeamID', 'CoachName']]
    )

    # ── 2. Build tournament round lookup from compact results ─────────────────
    # DayNum in the NCAA data encodes which round the game was in.
    tourney = kg_compact.copy()
    tourney['Round'] = pd.cut(
        tourney['DayNum'],
        bins=[0, 135, 137, 143, 145, 152, 154, 999],
        labels=['First Four', 'R64', 'R32', 'Sweet16', 'Elite8', 'Final4', 'Champion'],
    )

    # Winners
    w_tourney = tourney[['Season', 'WTeamID', 'Round']].rename(columns={'WTeamID': 'TeamID'}).copy()
    w_tourney['won'] = 1

    # Losers
    l_tourney = tourney[['Season', 'LTeamID', 'Round']].rename(columns={'LTeamID': 'TeamID'}).copy()
    l_tourney['won'] = 0

    team_games = pd.concat([w_tourney, l_tourney], ignore_index=True)
    team_games = team_games.merge(coaches, on=['Season', 'TeamID'], how='left')

    # ── 3. Aggregate per-coach career stats (cumulative) ──────────────────────
    coach_season = (
        team_games
        .groupby(['CoachName', 'Season'])
        .agg(
            team_id=('TeamID', 'first'),
            tourn_wins=('won', 'sum'),
            tourn_games=('won', 'count'),
            final_four=('Round', lambda x: int(any(r in ['Final4', 'Champion'] for r in x))),
            champion=('Round', lambda x: int(any(r == 'Champion' for r in x))),
        )
        .reset_index()
    )
    coach_season['appearance'] = 1

    # For each (coach, season) we only sum stats from *earlier* seasons.
    coach_career = []
    for coach, grp in coach_season.groupby('CoachName'):
        grp = grp.sort_values('Season').reset_index(drop=True)
        for _, row in grp.iterrows():
            past = grp[grp['Season'] < row['Season']]
            coach_career.append({
                'CoachName': coach,
                'Season': row['Season'],
                'TeamID': row['team_id'],
                'coach_appearances': int(past['appearance'].sum()),
                'coach_tourn_wins': int(past['tourn_wins'].sum()),
                'coach_final_fours': int(past['final_four'].sum()),
                'coach_championships': int(past['champion'].sum()),
                'coach_win_rate': (
                    past['tourn_wins'].sum() / past['tourn_games'].sum()
                    if past['tourn_games'].sum() > 0 else 0.0
                ),
            })

    coach_stats = pd.DataFrame(coach_career)

    # Join back to get TeamID + stats for joining to matchup / 2026 data
    result = coaches.merge(
        coach_stats[[
            'CoachName', 'Season', 'coach_appearances', 'coach_tourn_wins',
            'coach_final_fours', 'coach_championships', 'coach_win_rate',
        ]],
        on=['CoachName', 'Season'],
        how='left',
    )

    # Fill NaN for coaches with no prior tournament history
    for col in ['coach_appearances', 'coach_tourn_wins', 'coach_final_fours', 'coach_championships']:
        result[col] = result[col].fillna(0).astype(int)
    result['coach_win_rate'] = result['coach_win_rate'].fillna(0.0)

    return result

