/* ═══════════════════════════════════════════════════════════════════════════
   March Madness 2026 — client.

   All data is pre-computed and served as static JSON.  This script:
     - Switches between tabs
     - Renders the retrospective leaderboard, heatmap, upsets
     - Renders a full bracket for any selected model
     - Renders the hindsight feature-importance chart
     - Renders markdown docs via marked.js
   ═══════════════════════════════════════════════════════════════════════════ */

const API = {
    manifest: () => fetch('/api/manifest').then(r => r.json()),
    bracket: (slug) => fetch(`/api/bracket/${slug}`).then(r => r.json()),
    retrospective: () => fetch('/api/retrospective').then(r => r.json()),
    hindsight: () => fetch('/api/hindsight').then(r => r.json()),
    doc: (slug) => fetch(`/api/docs/${slug}`).then(r => r.text()),
};

const REGION_NAMES = { W: 'East', X: 'South', Y: 'Midwest', Z: 'West' };

const state = {
    manifest: null,
    currentModel: null,
    currentDoc: 'report',
    bracketCache: {},
};

// ─────────────────────────────────────────────────────────────────────────
// Tab switching
// ─────────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
        const target = btn.dataset.tab;
        document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.tab-panel').forEach(p => {
            p.classList.toggle('active', p.id === `panel-${target}`);
        });
        // Plotly needs a resize nudge when a hidden tab becomes visible
        window.dispatchEvent(new Event('resize'));
    });
});

// ─────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────
const fmtPct = (v) => v == null || isNaN(v) ? '—' : `${(v * 100).toFixed(1)}%`;
const fmtSeed = (v) => v == null ? '?' : String(Math.round(v));

function el(tag, props = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
        if (k === 'class') e.className = v;
        else if (k === 'html') e.innerHTML = v;
        else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
        else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
        else e.setAttribute(k, v);
    }
    for (const c of children.flat()) {
        if (c == null || c === false) continue;
        e.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return e;
}

// ─────────────────────────────────────────────────────────────────────────
// Masthead — champion strip
// ─────────────────────────────────────────────────────────────────────────
function renderResultStrip(manifest) {
    const strip = document.getElementById('result-strip');
    strip.innerHTML = '';
    if (!manifest.champion) {
        strip.append(el('span', {}, 'Tournament results not yet loaded.'));
        return;
    }
    strip.append(
        el('span', { class: 'trophy-label' }, '2026 Champion'),
        el('span', { class: 'champion' }, manifest.champion),
        el('span', { class: 'vs' }, `defeated ${manifest.runner_up || '—'}`),
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Retrospective — leaderboard
// ─────────────────────────────────────────────────────────────────────────
function renderLeaderboard(retro, manifest) {
    const root = document.getElementById('leaderboard');
    root.innerHTML = '';

    const sorted = [...retro.summary].sort((a, b) => (b.pick_accuracy ?? 0) - (a.pick_accuracy ?? 0));
    sorted.forEach((s, i) => {
        const card = el('div', {
            class: 'lb-card' + (i === 0 ? ' winner' : ''),
            style: { borderTopColor: s.color },
        });
        card.append(
            el('span', { class: 'rank' }, String(i + 1)),
            el('div', { class: 'model-name' }, s.model),
            el('div', { class: 'accuracy' },
                fmtPct(s.pick_accuracy),
                el('span', { class: 'label' }, `of ${s.games_graded} games`),
            ),
            el('div', { class: 'champion-line' },
                'Picked ',
                el('strong', {}, s.predicted_champion || '—'),
                ' · ',
                el('span', { class: `mark ${s.champion_correct ? 'ok' : 'bad'}` },
                    s.champion_correct ? '✓ champion' : '✗ champ wrong'),
            ),
            el('div', { class: 'blurb' }, s.blurb),
        );
        root.append(card);
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Retrospective — per-round heatmap (Plotly)
// ─────────────────────────────────────────────────────────────────────────
function renderRoundHeatmap(retro) {
    const perRound = retro.per_round.filter(r => r.round_num > 0);  // skip First Four
    const modelNames = retro.summary.map(s => s.model);

    const z = modelNames.map(m => perRound.map(r => r[m] ?? null));
    const x = perRound.map(r => r.round_label);
    const y = modelNames;

    const text = z.map(row => row.map(v => v == null ? '' : `${(v * 100).toFixed(0)}%`));

    const data = [{
        type: 'heatmap',
        x, y, z,
        text, texttemplate: '%{text}',
        textfont: { family: 'JetBrains Mono, ui-monospace, monospace', size: 13, color: '#171717' },
        colorscale: [
            [0.00, '#f6e7e5'],
            [0.35, '#f6e7e5'],
            [0.50, '#fbfaf7'],
            [0.65, '#e0ecdf'],
            [0.90, '#9cc7ac'],
            [1.00, '#2c7a5c'],
        ],
        zmin: 0, zmax: 1,
        showscale: false,
        xgap: 2,
        ygap: 2,
        hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.1%}<extra></extra>',
    }];

    const layout = {
        margin: { l: 170, r: 20, t: 10, b: 50 },
        height: 40 + modelNames.length * 44 + 60,
        paper_bgcolor: '#fbfaf7',
        plot_bgcolor: '#fbfaf7',
        font: { family: 'Inter, sans-serif', size: 13, color: '#171717' },
        xaxis: {
            side: 'bottom',
            showgrid: false,
            tickfont: { size: 12 },
            ticklen: 0,
        },
        yaxis: {
            autorange: 'reversed',
            showgrid: false,
            tickfont: { size: 12 },
            ticklen: 0,
        },
    };
    Plotly.newPlot('round-heatmap', data, layout, { displayModeBar: false, responsive: true });
}

// ─────────────────────────────────────────────────────────────────────────
// Retrospective — champion picks list
// ─────────────────────────────────────────────────────────────────────────
function renderChampionPicks(retro, manifest) {
    const root = document.getElementById('champion-picks');
    root.innerHTML = '';
    const sorted = [...retro.summary].sort((a, b) => (b.pick_accuracy ?? 0) - (a.pick_accuracy ?? 0));
    sorted.forEach(s => {
        root.append(el('div', { class: 'ch-row' },
            el('span', { class: 'model' }, s.model),
            el('span', {},
                el('span', { class: 'pick' }, s.predicted_champion || '—'),
                el('span', { class: `mark ${s.champion_correct ? 'ok' : 'bad'}` },
                    s.champion_correct ? ' ✓' : ' ✗'),
            ),
        ));
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Retrospective — upsets
// ─────────────────────────────────────────────────────────────────────────
function renderUpsets(retro) {
    const root = document.getElementById('upsets-list');
    root.innerHTML = '';
    retro.upsets.forEach(u => {
        const story = el('div', { class: 'story' },
            el('span', { class: 'seed winner' }, `#${u.winner_seed}`),
            el('span', { class: 'w' }, u.winner),
            ' beat ',
            el('span', { class: 'seed' }, `#${u.loser_seed}`),
            u.loser,
            u.winner_score != null ? `, ${u.winner_score}–${u.loser_score}` : '',
        );
        const called = el('div', { class: 'called-by' });
        if (u.called_by.length === 0) {
            called.append(el('span', { class: 'none' }, 'No model saw it coming.'));
        } else {
            u.called_by.forEach(m => called.append(el('span', { class: 'chip' }, m)));
        }
        root.append(el('div', { class: 'upset' },
            el('div', { class: 'round' }, u.round_label),
            story,
            called,
        ));
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Brackets — model switcher + layout
// ─────────────────────────────────────────────────────────────────────────
function renderModelSwitcher(manifest) {
    const root = document.getElementById('model-switcher');
    root.innerHTML = '';
    manifest.models.forEach((m, i) => {
        const btn = el('button', {
            class: 'model-pill' + (m.slug === state.currentModel ? ' active' : ''),
            onclick: () => {
                state.currentModel = m.slug;
                document.querySelectorAll('.model-pill').forEach(b =>
                    b.classList.toggle('active', b.dataset.slug === m.slug));
                loadBracket(m.slug);
            },
        }, m.name);
        btn.dataset.slug = m.slug;
        if (m.predicted_champion) {
            btn.append(el('span', { class: 'champ' }, `→ ${m.predicted_champion}`));
        }
        root.append(btn);
    });
}

async function loadBracket(slug) {
    let data = state.bracketCache[slug];
    if (!data) {
        data = await API.bracket(slug);
        state.bracketCache[slug] = data;
    }
    renderBracketMeta(data);
    renderBracket(data);
}

function renderBracketMeta(data) {
    const root = document.getElementById('bracket-meta');
    root.innerHTML = '';

    const games = data.games.filter(g => g.result_winner_id != null && g.pred_winner_id != null);
    const correct = games.filter(g => g.pred_winner_id === g.result_winner_id).length;
    const total = games.length;
    const acc = total ? correct / total : null;

    const ch = data.games.find(g => g.round_num === 6);
    const champCorrect = ch && ch.pred_winner_id && ch.result_winner_id && ch.pred_winner_id === ch.result_winner_id;

    root.append(
        el('div', { class: 'kv' },
            el('span', { class: 'k' }, 'Model'),
            el('span', { class: 'v' }, data.model),
        ),
        el('div', { class: 'kv' },
            el('span', { class: 'k' }, 'Pick accuracy'),
            el('span', { class: 'v' }, fmtPct(acc)),
        ),
        el('div', { class: 'kv' },
            el('span', { class: 'k' }, 'Predicted champion'),
            el('span', { class: `v ${ch ? (champCorrect ? 'ok' : 'bad') : ''}` },
                (ch && ch.pred_winner) || '—',
                ch ? ` ${champCorrect ? '✓' : '✗'}` : '',
            ),
        ),
        el('div', { class: 'kv' },
            el('span', { class: 'k' }, 'Graded games'),
            el('span', { class: 'v' }, `${correct} / ${total}`),
        ),
    );
}

// Kaggle visual slot order (so R1 pairings land next to their R2 parent)
const R1_SUFFIXES = [1, 8, 2, 7, 3, 6, 4, 5];

function gameCard(g) {
    if (!g) return el('div', { class: 'game tbd' });

    const isActual = g.result_winner_id != null;
    const predId = g.pred_winner_id;
    const actualId = g.result_winner_id;
    const correct = isActual && predId != null && actualId === predId;

    const strongWins = predId != null && predId === g.strong_team_id;
    const weakWins   = predId != null && predId === g.weak_team_id;

    const cls = isActual ? (correct ? 'correct' : 'wrong') : '';

    const card = el('div', { class: `game ${cls}` });

    // Predicted row 1 (strong)
    card.append(
        el('div', { class: 'team' + (strongWins ? ' winner' : '') },
            el('span', { class: 'seed' }, fmtSeed(g.strong_seed)),
            el('span', { class: 'name' }, g.strong_team || 'TBD'),
            el('span', { class: 'score' },
                g.strong_pred_score != null ? Math.round(g.strong_pred_score) : ''),
        ),
        el('div', { class: 'team' + (weakWins ? ' winner' : '') },
            el('span', { class: 'seed' }, fmtSeed(g.weak_seed)),
            el('span', { class: 'name' }, g.weak_team || 'TBD'),
            el('span', { class: 'score' },
                g.weak_pred_score != null ? Math.round(g.weak_pred_score) : ''),
        ),
    );

    if (isActual) {
        const resultText = g.result_winner ? g.result_winner : '';
        let scoreText = '';
        if (g.result_strong_score != null && g.result_weak_score != null) {
            scoreText = `${Math.round(g.result_strong_score)}–${Math.round(g.result_weak_score)}`;
        }
        card.append(
            el('div', { class: 'actual-strip' },
                el('span', { class: `tag ${correct ? 'ok' : 'bad'}` },
                    correct ? '✓ actual' : '✗ actual'),
                el('span', { class: 'score' }, `${resultText} ${scoreText}`),
            ),
        );
    }
    return card;
}

function renderRegion(games, regionCode, rtl) {
    const regionGames = games.filter(g => g.region === regionCode);
    const byId = {};
    regionGames.forEach(g => byId[g.slot_id] = g);

    const wrap = el('div', {
        class: 'region-bracket' + (rtl ? ' rtl' : ''),
    });

    // R64
    R1_SUFFIXES.forEach((suf, i) => {
        const g = byId[`R1${regionCode}${suf}`];
        const slot = el('div', {
            class: 'slot',
            style: {
                gridColumn: rtl ? '4' : '1',
                gridRow: String(i + 1),
            }
        }, gameCard(g));
        slot.dataset.round = '1';
        wrap.append(slot);
    });
    // R32
    [1, 2, 3, 4].forEach((suf, i) => {
        const g = byId[`R2${regionCode}${suf}`];
        const slot = el('div', {
            class: 'slot',
            style: {
                gridColumn: rtl ? '3' : '2',
                gridRow: `${i * 2 + 1} / span 2`,
            }
        }, gameCard(g));
        slot.dataset.round = '2';
        wrap.append(slot);
    });
    // Sweet 16
    [1, 2].forEach((suf, i) => {
        const g = byId[`R3${regionCode}${suf}`];
        const slot = el('div', {
            class: 'slot',
            style: {
                gridColumn: rtl ? '2' : '3',
                gridRow: `${i * 4 + 1} / span 4`,
            }
        }, gameCard(g));
        slot.dataset.round = '3';
        wrap.append(slot);
    });
    // Elite 8
    {
        const g = byId[`R4${regionCode}1`];
        const slot = el('div', {
            class: 'slot',
            style: {
                gridColumn: rtl ? '1' : '4',
                gridRow: '1 / span 8',
            }
        }, gameCard(g));
        slot.dataset.round = '4';
        wrap.append(slot);
    }
    return wrap;
}

function renderRegionBlock(games, regionCode, rtl) {
    const labels = ['R64', 'R32', 'Sweet 16', 'Elite 8'];
    if (rtl) labels.reverse();
    const regionName = REGION_NAMES[regionCode] || regionCode;
    const cls = regionName.toLowerCase();
    return el('div', {},
        el('div', { class: `region-header ${cls}` }, regionName),
        el('div', { class: 'bracket-round-labels' + (rtl ? ' rtl' : '') },
            ...labels.map(l => el('div', {}, l))),
        renderRegion(games, regionCode, rtl),
    );
}

function renderFinalFour(games) {
    const ff = games.filter(g => g.round_num === 5).sort((a, b) => a.slot_id.localeCompare(b.slot_id));
    const ch = games.filter(g => g.round_num === 6);
    return el('div', { class: 'final-four' },
        el('div', { class: 'round-stack' },
            el('h4', {}, 'Final Four'),
            ...ff.map(g => gameCard(g)),
        ),
        el('div', { class: 'round-stack' },
            el('h4', {}, 'Championship'),
            ...ch.map(g => gameCard(g)),
        ),
    );
}

function renderBracket(data) {
    const root = document.getElementById('bracket-view');
    root.innerHTML = '';

    // First Four
    const ff = data.games.filter(g => g.round_num === 0);
    if (ff.length) {
        const grid = el('div', { class: 'first-four-grid' },
            ...ff.map(g => gameCard(g)));
        root.append(el('div', { class: 'first-four' },
            el('h3', {}, 'First Four'),
            grid,
        ));
    }

    root.append(el('div', { class: 'bracket-outer' },
        el('div', { class: 'bracket-col' },
            renderRegionBlock(data.games, 'W', false),
            renderRegionBlock(data.games, 'Y', false),
        ),
        el('div', { class: 'bracket-col center' },
            renderFinalFour(data.games),
        ),
        el('div', { class: 'bracket-col' },
            renderRegionBlock(data.games, 'X', true),
            renderRegionBlock(data.games, 'Z', true),
        ),
    ));
}

// ─────────────────────────────────────────────────────────────────────────
// Hindsight tab
// ─────────────────────────────────────────────────────────────────────────
const hindsightState = { data: null, compareWith: null };

async function loadHindsight() {
    const h = await API.hindsight();
    hindsightState.data = h;

    const stats = document.getElementById('hindsight-stats');
    stats.innerHTML = '';
    const cells = [
        ['Training games', h.train_games, 'the 67 tournament games'],
        ['Training win accuracy', fmtPct(h.train_win_acc), 'by construction — this is the fit, not a test'],
        ['Training margin MAE', h.train_margin_mae.toFixed(2), 'points of spread error'],
        ['Training total MAE', h.train_total_mae.toFixed(2), 'points of combined-score error'],
    ];
    cells.forEach(([k, v, sub]) => {
        stats.append(el('div', { class: 'stat-cell' },
            el('div', { class: 'k' }, k),
            el('div', { class: 'v' }, String(v)),
            el('div', { class: 'sub' }, sub),
        ));
    });

    renderCompareSwitcher(h);
    renderImportanceChart();
}

function renderCompareSwitcher(h) {
    const root = document.getElementById('compare-switcher');
    root.innerHTML = '';
    root.append(el('span', { class: 'label' }, 'Overlay:'));

    const noneBtn = el('button', {
        class: 'pill' + (hindsightState.compareWith == null ? ' active' : ''),
        onclick: () => { hindsightState.compareWith = null; renderCompareSwitcher(h); renderImportanceChart(); },
    }, 'Hindsight only');
    root.append(noneBtn);

    const models = Object.keys(h.comparison_models || {});
    models.forEach(m => {
        root.append(el('button', {
            class: 'pill' + (hindsightState.compareWith === m ? ' active' : ''),
            onclick: () => { hindsightState.compareWith = m; renderCompareSwitcher(h); renderImportanceChart(); },
        }, m));
    });
}

function renderNarrative() {
    const h = hindsightState.data;
    const compare = hindsightState.compareWith;
    const narrative = compare
        ? (h.comparison_models[compare]?.narrative || {})
        : (h.narrative || {});
    document.getElementById('narrative-headline').textContent =
        narrative.headline || 'Reading this chart';
    document.getElementById('narrative-body').innerHTML =
        narrative.body || '';
}

function renderImportanceChart() {
    renderNarrative();
    const h = hindsightState.data;
    const compare = hindsightState.compareWith;
    const hindLookup = Object.fromEntries(h.importances.map(d => [d.feature, d.importance]));

    let features;
    let traces;
    let layoutNote;

    if (!compare) {
        // Hindsight alone
        const top = h.importances.slice(0, 15);
        features = top.map(d => d.feature);
        traces = [{
            type: 'bar',
            orientation: 'h',
            x: top.map(d => d.importance),
            y: features,
            marker: { color: '#c13b2a' },
            name: 'Hindsight (2026 actuals)',
            hovertemplate: '<b>%{y}</b><br>%{x:.1%} of total importance<extra>Hindsight</extra>',
        }];
        layoutNote = h.method;
    } else {
        const cmp = h.comparison_models[compare];
        const cmpLookup = Object.fromEntries(cmp.importances.map(d => [d.feature, d.importance]));

        // Union of top-10 from each side, then sort by Hindsight importance
        const hTop = h.importances.slice(0, 10).map(d => d.feature);
        const cTop = cmp.importances.slice(0, 10).map(d => d.feature);
        const unionSet = new Set([...hTop, ...cTop]);
        features = Array.from(unionSet).sort((a, b) =>
            (hindLookup[b] ?? 0) - (hindLookup[a] ?? 0)
        );

        traces = [
            {
                type: 'bar',
                orientation: 'h',
                x: features.map(f => hindLookup[f] ?? 0),
                y: features,
                marker: { color: '#c13b2a' },
                name: 'Hindsight — what 2026 rewarded',
                hovertemplate: '<b>%{y}</b><br>%{x:.1%} — Hindsight<extra></extra>',
            },
            {
                type: 'bar',
                orientation: 'h',
                x: features.map(f => cmpLookup[f] ?? 0),
                y: features,
                marker: { color: '#2b6aa3' },
                name: `${compare} — pre-tournament prior`,
                hovertemplate: `<b>%{y}</b><br>%{x:.1%} — ${compare}<extra></extra>`,
            },
        ];
        layoutNote = `${h.method}  ·  ${cmp.method}`;
    }

    // Reverse so top feature sits at top of chart
    features = features.slice().reverse();
    traces = traces.map(t => ({
        ...t,
        x: t.x.slice().reverse(),
        y: features,
    }));

    const height = Math.max(360, features.length * 36 + 120);
    const layout = {
        margin: { l: 220, r: 40, t: 20, b: 60 },
        height,
        paper_bgcolor: '#fbfaf7',
        plot_bgcolor: '#fbfaf7',
        font: { family: 'Inter, sans-serif', size: 13, color: '#171717' },
        barmode: 'group',
        bargap: 0.25,
        bargroupgap: 0.08,
        showlegend: !!compare,
        legend: {
            orientation: 'h',
            x: 0, y: -0.08,
            xanchor: 'left', yanchor: 'top',
            font: { size: 12, color: '#3b3b3b' },
        },
        xaxis: {
            title: { text: layoutNote, standoff: 14, font: { size: 11, color: '#6b6b6b' } },
            tickfont: { size: 11, color: '#6b6b6b' },
            showgrid: true,
            gridcolor: '#e3dfd6',
            zeroline: false,
            tickformat: '.0%',
        },
        yaxis: {
            tickfont: { family: 'JetBrains Mono, ui-monospace, monospace', size: 12, color: '#171717' },
            showgrid: false,
            zeroline: false,
        },
    };
    Plotly.newPlot('hindsight-importances', traces, layout, { displayModeBar: false, responsive: true });
}

// ─────────────────────────────────────────────────────────────────────────
// Docs tab
// ─────────────────────────────────────────────────────────────────────────
async function loadDoc(slug) {
    const body = document.getElementById('doc-body');
    body.textContent = 'Loading…';
    try {
        const md = await API.doc(slug);
        body.innerHTML = marked.parse(md);
    } catch (e) {
        body.textContent = `Could not load ${slug}.md`;
    }
}

document.querySelectorAll('.doc-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.doc-btn').forEach(b => b.classList.toggle('active', b === btn));
        state.currentDoc = btn.dataset.doc;
        loadDoc(btn.dataset.doc);
    });
});

// ─────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────
async function init() {
    try {
        const [manifest, retro] = await Promise.all([API.manifest(), API.retrospective()]);
        state.manifest = manifest;
        state.currentModel = manifest.models[0]?.slug;

        renderResultStrip(manifest);
        renderLeaderboard(retro, manifest);
        renderRoundHeatmap(retro);
        renderChampionPicks(retro, manifest);
        renderUpsets(retro);

        renderModelSwitcher(manifest);
        if (state.currentModel) loadBracket(state.currentModel);

        loadHindsight();
        loadDoc(state.currentDoc);
    } catch (err) {
        console.error(err);
        document.body.insertAdjacentHTML('afterbegin',
            `<div style="padding:20px;background:#fbe9e7;color:#b72;font-family:system-ui;">Failed to load data: ${err.message}. Did you run <code>prepare_data.py</code>?</div>`);
    }
}

init();
