# What the models are looking at

Every model on this site picks a winner by comparing two teams across some subset of the features below. Features ending in `_diff` are **Team A minus Team B**, where Team A is always the lower (favored) seed — so a positive value means the favorite looks stronger on that axis.

For each feature we list: what it is, how to read it in basketball terms, and why it matters for March.

---

## 1. Efficiency metrics — "how good is this team per possession?"

The bedrock of modern college basketball analytics. These numbers are pace-adjusted, which matters because a team scoring 85 at a slow pace is not the same as a team scoring 85 at a fast pace.

### `adj_em_diff` — Adjusted Efficiency Margin
The difference between a team's *adjusted offensive efficiency* (points scored per 100 possessions) and *adjusted defensive efficiency* (points allowed per 100 possessions). Positive = good team. This is generally the **single most predictive** number in college basketball, which is why **Comparative Metrics** leans 51% of its weight here.

### `adj_o_diff`, `adj_d_diff` — Adjusted Offensive / Defensive Efficiency
The two halves of `adj_em`, broken out. Sometimes a team's offense is carrying it (think high-variance shooters) and sometimes a team's defense is (think veteran, physical team). Breaking them out lets the model see the shape of the talent, not just the sum.

### `barthag_diff` — Barttorvik Rating
A team's estimated probability of beating an average Division-I team on a neutral court. Think of it as a "true-team-strength" scalar: 0.95 is a Final Four candidate, 0.70 is a bubble team, 0.30 is a bid-stealer. `barthag_diff` is how much more likely Team A is to beat the neutral-court average than Team B is.

### `adj_t_diff` — Adjusted Tempo
Possessions per 40 minutes, pace-adjusted. Does *not* mean "fast team good" — some great teams play slow. But **tempo mismatches** can be weapons. A fast team playing a slow team often tries to force transition possessions the slow team isn't built for.

---

## 2. Schedule-quality metrics — "who have they actually played?"

Efficiency numbers can be inflated by beating up on weak schedules. These features ask the résumé question.

### `sos_diff` — Strength of Schedule
The average Barthag rating of a team's opponents. Higher = played tougher opponents. A 25-5 record against the 300th-ranked SOS is not the same as 25-5 against the 15th-ranked SOS. **Greg_v1** puts its single largest weight on this feature.

### `ncsos_diff` — Non-Conference Strength of Schedule
How tough was a team's non-conference slate? This one is partially in the coach's control (scheduling), so teams that ducked tough games get flagged here even if their conference rescued their overall SOS.

### `consos_diff` — Conference Strength of Schedule
How strong the team's conference-only schedule was. A 13-5 conference record in the Big 12 means more than 13-5 in the MWC. Teams in power conferences get battle-tested every week; mid-major teams often don't face a top-30 opponent for two months.

### `elite_sos_diff` — Elite Strength of Schedule
Tightened version of SOS that only counts top-50-ish opponents. Gets around the problem that beating lots of mediocre teams can inflate regular SOS.

### `wab_diff` — Wins Above Bubble
One of the most underrated numbers in the sport. It asks: *how many more wins does this team have than a bubble team would have playing the same schedule?* A high WAB means the team is not just winning, but winning games a borderline tournament team would have lost. **Hindsight says this is one of the top three features for 2026.**

---

## 3. Quality-of-wins metrics — "how do they perform against good teams?"

### `qual_o_diff`, `qual_d_diff`, `qual_barthag_diff`
Offensive efficiency, defensive efficiency, and Barthag *restricted to games against top-tier opponents*. Season averages can hide a team's ceiling — Sweet 16 and beyond, everyone is a top-tier opponent, so the question becomes "how does this team play when they can't cruise?"

### `qual_games_diff`
Raw count of quality-opponent games played. Zero quality games = zero evidence of how they handle pressure. Some mid-majors with dazzling adj_em numbers simply haven't been stress-tested.

---

## 4. Shooting-style features — "how do they score?"

### `off_efg_diff`, `def_efg_diff` — Effective Field-Goal %
Field-goal percentage weighted to give 1.5× credit for three-point makes. Basketball's single best "shooting ability" stat.

### `off_to_diff`, `def_to_diff` — Turnover Rate
Turnovers per 100 possessions. Teams that don't turn it over tend to survive March's pressure; teams that force turnovers on defense create bonus possessions.

### `off_or_diff`, `def_or_diff` — Offensive / Defensive Rebound %
Second-chance opportunities (offensive glass) and defensive-glass lockdown. Big, athletic teams eat on the offensive glass; guard-heavy teams live and die on boards.

### `off_ftr_diff`, `def_ftr_diff` — Free-Throw Rate
Free-throws attempted per field-goal attempt. High `off_ftr` teams get to the line — a huge tournament edge in tight games.

### `fg2_pct_diff`, `fg3_pct_diff`
Two-point and three-point shooting percentage. Teams that live from three have higher variance; teams that live inside tend to be more stable.

### `ast_rate_diff`
Assist rate — what fraction of made field goals come off an assist. High assist rate = ball-movement offense; low assist rate = isolation-heavy. **Lean GB** weights this as one of its seven features.

---

## 5. Coach features — "who's running the team?"

### `coach_appearances_diff` — Career Tournament Appearances
How many times the head coach has been in the NCAA Tournament. Experienced coaches tend to survive first weekend; first-timers often bow out early.

### `coach_tourn_wins_diff` — Career Tournament Wins
Raw tournament wins across the coach's career. Strong signal for coaches who've won deep in the bracket before.

### `coach_final_fours_diff` — Career Final Fours
A big deal: Final Fours are made by coaches who've been there before. Disproportionately predictive for Elite 8 → Final Four games.

### `coach_win_rate_diff` — Career Tournament Win Rate
Proportion of tournament games the coach has won. Normalizes for appearances — catches elite coaches early and punishes perennials who always exit early.

---

## 6. Seed / structural features — "what does the bracket say?"

### `seed_diff`
Team A's seed minus Team B's seed. Always negative (Team A is the favorite by convention). A `seed_diff` of -15 is a 1-vs-16; -1 is a 7-vs-8.

### `min_seed`
The lower of the two seeds. Used by **Lean GB** to encode "is this a 1-seed game" vs "is this an 8-vs-9". High-seeded matchups (min_seed = 1 or 2) behave very differently from low-seeded ones.

### `is_big_gap`
Binary flag for when |seed_diff| >= 8. Distinguishes true mismatches (1-vs-16, 2-vs-15) from everything else.

### `is_late_round`
Binary flag for Sweet 16 and beyond. Later-round games reward different traits than first-weekend games — for example, coach experience and defensive efficiency both matter more once you're through to the regional semis.

### `seed_product`, `seed_sum`
Polynomial expansions of seed — helps Ridge regression capture non-linear seed effects. `seed_product` differentiates (1-vs-8) from (2-vs-4) which sum to the same `seed_sum`.

---

## 7. Engineered "disagreement" features

These are the most interesting features in the set because they encode *market vs model* tension.

### `seed_disagreement`
Residual of `adj_em_diff` regressed on `seed_diff`. In English: given how much the seeds differ, how much more or less efficient is Team A than we'd expect? Positive = Team A is better than its seed suggests. **This is Hindsight's single most important feature for 2026** — teams whose efficiency numbers outran their seeding went further than the committee predicted.

### `consensus_disagreement`
Similar idea, but against Massey Consensus rankings rather than raw efficiency. Captures "this team is ranked differently than the market thinks they should be." Used by some variants of **Lean GB**.

---

## 8. Geographic & stylistic features (new this cycle)

### `teamA_home_state_adv`
Binary flag for whether Team A's home state is the same as the game's host city's state. Plays to Dance mythology — Kentucky in Louisville, BYU in Denver — but the real-world effect is modest. Tested as a model variant this year; did not improve accuracy enough to include in the lean baseline.

### `ast_vs_stl_clash_adv`
Interaction term: does one team's assist-heavy offense meet the other's steal-heavy defense? Captures the stylistic friction that season averages miss. Like `teamA_home_state_adv`, tested as a variant; useful as a "show your work" exercise more than a predictive lift.

---

## Which features which models use

| Model | Features |
|---|---|
| **Seeding Only** | Seed only (no stats) |
| **Comparative Metrics** | 10 features: mostly adj_em, barthag, seed, coach |
| **Greg_v1** | 28 features across all of the above categories |
| **Lean GB** (all 3 variants) | 7 features: seed_disagreement, wab, adj_em, barthag, ast_rate, min_seed, is_big_gap |
| **Animal Kingdom** | None — just mascot names fed to Claude |
| **Vegas Odds** | None — just spread + total |
| **Hindsight** | Same ~25-feature base as Comparative Metrics, trained on the 67 tournament games themselves |

Different feature sets mean each model has a different "theory" of what matters. That's the whole point of the **What Mattered** tab — you can see at a glance which theory 2026 actually rewarded.
