"""
Visualization and display helpers.

Utility functions for EDA chart routing, bracket display formatting,
upset explanation text, and team name lookups. These are display-layer
functions that don't affect modeling logic.
"""


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def is_boolean_col(series):
    """Detect whether a numeric series contains only 0/1 values.

    Used during EDA to route columns to the right chart type: boolean
    columns get bar charts instead of histograms/KDE.

    Parameters
    ----------
    series : pd.Series
        A numeric column to check.

    Returns
    -------
    bool
    """
    vals = series.dropna().unique()
    return len(vals) > 0 and set(vals).issubset({0, 1, 0.0, 1.0})


# ---------------------------------------------------------------------------
# Team name lookup
# ---------------------------------------------------------------------------

def team_name(tid, kg_teams):
    """Look up a team's display name from their Kaggle TeamID.

    Parameters
    ----------
    tid : int
        Kaggle TeamID.
    kg_teams : pd.DataFrame
        Kaggle ``MTeams.csv`` with ``TeamID`` and ``TeamName`` columns.

    Returns
    -------
    str
    """
    n = kg_teams.loc[kg_teams['TeamID'] == tid, 'TeamName'].values
    return n[0] if len(n) else str(tid)


# ---------------------------------------------------------------------------
# Bracket display
# ---------------------------------------------------------------------------

def get_upset_threshold(s_a, s_b, rnd_idx,
                        r64_thresholds=None,
                        r32_threshold=0.42,
                        late_threshold=0.45):
    """Return the upset-pick probability threshold for a given matchup and round.

    Thresholds are tiered by seed matchup (R64) and by round (R32+).
    A model probability exceeding the threshold triggers an upset pick.

    Parameters
    ----------
    s_a, s_b : int
        Seeds of the two teams.
    rnd_idx : int
        Round index (0 = R64, 1 = R32, 2+ = Sweet 16 and beyond).
    r64_thresholds : dict, optional
        Mapping of ``(hi_seed, lo_seed) → threshold`` for R64.
    r32_threshold : float
        Threshold for Round of 32 games.
    late_threshold : float
        Threshold for Sweet 16 and beyond.

    Returns
    -------
    float
    """
    if r64_thresholds is None:
        r64_thresholds = {
            (1, 16): 0.50, (2, 15): 0.50, (3, 14): 0.50, (4, 13): 0.50,
            (5, 12): 0.37, (6, 11): 0.40, (7, 10): 0.40, (8, 9): 0.50,
        }
    if rnd_idx == 0:
        return r64_thresholds.get((min(s_a, s_b), max(s_a, s_b)), 0.50)
    if rnd_idx == 1:
        return r32_threshold
    return late_threshold


def upset_flag(p_dog, threshold, close_margin=0.03):
    """Return a string annotation for upset/close-call picks.

    Parameters
    ----------
    p_dog : float
        Model's predicted win probability for the underdog.
    threshold : float
        The upset threshold for this matchup.
    close_margin : float
        How close to threshold counts as a "close call".

    Returns
    -------
    str
        ``'  << UPSET PICK'``, ``'  << CLOSE CALL'``, or ``''``.
    """
    if p_dog > threshold:
        return '  << UPSET PICK'
    elif p_dog > threshold - close_margin:
        return '  << CLOSE CALL'
    return ''


def print_game(winner, loser, win_pct, kg_teams, rnd_idx=None,
               r64_thresholds=None, close_margin=0.03):
    """Print one formatted bracket game line with upset tagging.

    Parameters
    ----------
    winner : tuple
        ``(TeamID, seed)`` of the predicted winner.
    loser : tuple
        ``(TeamID, seed)`` of the predicted loser.
    win_pct : float
        Model's win probability (0–100 scale).
    kg_teams : pd.DataFrame
        Team names lookup table.
    rnd_idx : int or None
        Round index for threshold lookup.

    Returns
    -------
    bool
        True if this game is an upset (winner has higher seed number).
    """
    w = team_name(winner[0], kg_teams) if winner[0] else 'TBD'
    l = team_name(loser[0], kg_teams) if loser[0] else 'TBD'
    is_upset = winner[1] > loser[1]
    tag = '  *** UPSET ***' if is_upset else ''
    if rnd_idx is not None and rnd_idx > 0 and not is_upset:
        threshold = get_upset_threshold(winner[1], loser[1], rnd_idx,
                                        r64_thresholds=r64_thresholds)
        if min(win_pct, 100 - win_pct) / 100 > threshold - close_margin:
            tag = '  << CLOSE CALL'
    print(f'    #{winner[1]:<2} {w:<22} over #{loser[1]:<2} {l:<22}  '
          f'({win_pct:.1f}% model){tag}')
    return is_upset


# ---------------------------------------------------------------------------
# Upset explanation
# ---------------------------------------------------------------------------

# Feature labels for explaining upset picks. Keys are feature column names;
# values are (short_label, human_readable_detail) tuples.
FEAT_LABELS = {
    'adj_em_diff':            ('efficiency margin gap',    'underdog rates closer in net efficiency than seed implies'),
    'barthag_diff':           ('power rating',             'underdog win-probability rating closer than seed implies'),
    'adj_o_diff':             ('offensive efficiency',     'underdog scores more efficiently per possession'),
    'adj_d_diff':             ('defensive efficiency',     'underdog holds opponents to fewer points per possession'),
    'wab_diff':               ('wins above bubble',        'underdog beat stronger teams relative to bubble'),
    'sos_diff':               ('strength of schedule',     'underdog played a harder schedule'),
    'seed_disagreement':      ('seed disagreement',        'model thinks favorite is overseeded for their efficiency'),
    'consensus_disagreement': ('computer consensus',       'ranking systems rate underdog higher than their seed'),
    'ft_pct_diff':            ('free throw accuracy',      'underdog shoots free throws better'),
    'ast_rate_diff':          ('assist rate',              'underdog moves the ball more effectively'),
    'blk_rate_diff':          ('block rate',               'underdog protects the rim better'),
    'stl_rate_diff':          ('steal rate',               'underdog applies more perimeter pressure'),
    'coach_tourn_wins_diff':  ('coaching tournament wins', 'underdog coach has more tournament wins'),
    'coach_appearances_diff': ('coaching experience',      'underdog coach has more tournament appearances'),
}


def explain_upset(feat_dict, nm_hi, nm_lo, n=3):
    """Format human-readable text explaining which feature edges drove an upset pick.

    All ``_diff`` features are (Team A / higher seed) minus (Team B / lower seed).
    Negative values indicate the underdog has the advantage on that metric.
    We surface the N strongest underdog edges.

    Parameters
    ----------
    feat_dict : dict
        Feature values for the matchup.
    nm_hi : str
        Name of the favored (higher-seeded) team.
    nm_lo : str
        Name of the underdog.
    n : int
        Number of top edges to display.

    Returns
    -------
    str
        Multi-line explanation string.
    """
    edges = []
    for col, (label, detail) in FEAT_LABELS.items():
        val = feat_dict.get(col, float('nan'))
        if val != val:  # nan check
            continue
        if val < 0:
            edges.append((abs(val), label, detail))
    edges.sort(reverse=True)
    if not edges:
        return f'      Why: {nm_lo} rates closer to {nm_hi} than the seed gap suggests'
    lines = [f'      Why ({nm_lo} upset pick):']
    for _, label, detail in edges[:n]:
        lines.append(f'        · {label}: {detail}')
    return chr(10).join(lines)
