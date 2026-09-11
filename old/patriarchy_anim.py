"""
patriarchy_anim.py — Animated single-run of the Patriarchy Simulation.

Visualises 100 people as nodes in a network, coloured by gender (fill)
and current job (border). Cross-gender friendship edges are highlighted
in amber. Three live metric charts run alongside.

The dynamic and no_friends scenarios share a percentile layout: red/blue dots on
y = job-trait percentile; Job A/B/C bands fixed at the 85th / 50th percentile lines.

Usage:
  python patriarchy_anim.py                      # fixed similarity friends (default)
  python patriarchy_anim.py --scenario dynamic   # dynamic friends (job-band layout)
  python patriarchy_anim.py --scenario dynamic --highlight --save  # Emma/Rachel tracking
  python patriarchy_anim.py --scenario gender_cutoff  # 0.7 same-gender, 0.9 cross-gender
  python patriarchy_anim.py --scenario gender_cutoff --highlight --save
  python patriarchy_anim.py --scenario no_friends # trial run (same layout as dynamic)
  python patriarchy_anim.py --scenario reservation # 40% Job A quota
  python patriarchy_anim.py --scenario reservation --highlight --save
  python patriarchy_anim.py --scenario reservation_ab # 40% Job A + 40% Job B quota
  python patriarchy_anim.py --scenario reservation_ab --highlight --save
  python patriarchy_anim.py --scenario ff_bias   # dynamic + female-female bias
  python patriarchy_anim.py --scenario random    # random childhood friends
  python patriarchy_anim.py --save               # write patriarchy_anim_<scenario>.mp4
  python patriarchy_anim.py --iters 8000         # run 8000 iterations (default)
  python patriarchy_anim.py --step 5             # animate every 5th iter (default)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

# ─── Parameters ───────────────────────────────────────────────────────────────
N_MALE, N_FEMALE  = 50, 50
N_PEOPLE          = N_MALE + N_FEMALE
N_TRAITS          = 10
JOB_TRAITS        = [0, 1, 2]
JOB_A_COUNT       = 15
JOB_B_COUNT       = 35
ALPHA             = 0.10
NOISE_STD         = 0.08   # life shocks
RAND_INFL_STD     = 9.0    # spread of random influence draw
INNATE_MATCH_PWR  = 2.0
MISALIGNED_GAIN   = 0.04
OPPOSED_GAIN      = 0.015
INNATE_STICK_PWR  = 3.0    # higher = stickier when close to innate
INNATE_STICK_FLOOR = 0.015
SEED              = 42
BASE_CUTOFF       = 0.75
SAME_GENDER_CUTOFF  = 0.70   # friendship threshold within gender
CROSS_GENDER_CUTOFF = 0.90   # friendship threshold across genders
FF_CUTOFF_FACTOR  = 0.70
FF_INFLUENCE_MULT = 2.0
FRIEND_WEIGHT     = 0.5   # friend vs job-peer influence when you have friends
JOB_PAY           = {0: 1000, 1: 300, 2: 100}

genders     = np.array([0] * N_MALE + [1] * N_FEMALE)
male_mask   = genders == 0
female_mask = genders == 1

# Job A / B gender quotas: 40% of each tier reserved for women
JOB_A_RESERVATION_F = 6    # 6 of 15 Job A slots
JOB_B_RESERVATION_F = 14   # 14 of 35 Job B slots

# ─── Colour palette ───────────────────────────────────────────────────────────
C_BG     = '#0D0D0D'
C_MALE   = '#4A90D9'
C_FEMALE = '#E8705A'
C_JOB_A  = '#FFD700'
C_JOB_B  = '#C0C0C0'
C_JOB_C  = '#555555'
C_BRIDGE = '#F5C842'
C_HIGHLIGHT_RING = '#F5C842'
C_RESERVED_A_GLOW = '#FFD700'   # gold — Job A quota women
C_RESERVED_B_GLOW = '#5BC0EB'   # cyan — Job B quota women
C_RESERVED_A_RING = '#FF6B9D'
C_RESERVED_B_RING = '#5BC0EB'
C_RESERVED_A_FILL = '#2a1520'
C_RESERVED_B_FILL = '#102030'
C_EDGE   = '#FFFFFF'
C_TEXT   = '#FFFFFF'
C_MUTED  = '#888888'

JOB_COLORS = [C_JOB_A, C_JOB_B, C_JOB_C]
DUMMY_SEG  = [[[1e9, 1e9], [1e9, 1e9]]]   # off-screen placeholder for empty collections

# ─── Reproducible initial state ───────────────────────────────────────────────
np.random.seed(SEED)
_t = np.zeros((N_PEOPLE, N_TRAITS))
_t[:N_MALE, :3] = np.random.normal(57, 15, (N_MALE, 3))   # male job-trait advantage
_t[:N_MALE, 3:] = np.random.normal(50, 12, (N_MALE, 7))
_t[N_MALE:, :3] = np.random.normal(43, 15, (N_FEMALE, 3)) # female job-trait disadvantage
_t[N_MALE:, 3:] = np.random.normal(50, 12, (N_FEMALE, 7))
INIT_TRAITS   = np.clip(_t, 1.0, 99.0)

# Sample innates for men only; each woman gets the same innate profile as a paired man
_male_innate = np.clip(np.random.normal(50, 12, (N_MALE, N_TRAITS)), 1.0, 99.0)
_pairing = np.random.permutation(N_MALE)   # woman i shares innates with man _pairing[i]
INNATE_TRAITS = np.zeros((N_PEOPLE, N_TRAITS))
INNATE_TRAITS[male_mask] = _male_innate
INNATE_TRAITS[female_mask] = _male_innate[_pairing]

# ─── Simulation functions ─────────────────────────────────────────────────────

def assign_jobs(traits, reservation_a_f=0, reservation_b_f=0):
    scores = traits[:, JOB_TRAITS].sum(axis=1)
    rank   = list(np.argsort(-scores))
    jobs   = np.full(N_PEOPLE, 2, dtype=int)

    if reservation_a_f == 0 and reservation_b_f == 0:
        jobs[rank[:JOB_A_COUNT]]                          = 0
        jobs[rank[JOB_A_COUNT:JOB_A_COUNT + JOB_B_COUNT]] = 1
        return jobs

    female_by_score = [i for i in rank if female_mask[i]]
    reserved_a = female_by_score[:reservation_a_f]
    reserved_b = female_by_score[reservation_a_f:reservation_a_f + reservation_b_f]
    quota_set  = set(reserved_a) | set(reserved_b)
    open_pool  = [i for i in rank if i not in quota_set]

    job_a = list(reserved_a) + open_pool[:JOB_A_COUNT - reservation_a_f]
    placed = set(job_a)

    job_b = list(reserved_b)
    for p in rank:
        if len(job_b) >= JOB_B_COUNT:
            break
        if p not in placed and p not in job_b:
            job_b.append(p)

    jobs[job_a] = 0
    jobs[job_b] = 1
    return jobs


def trait_similarity(traits):
    c   = traits - 50.0
    nrm = np.linalg.norm(c, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-6, 1e-6, nrm)
    sim = (c / nrm) @ (c / nrm).T
    np.fill_diagonal(sim, 0.0)
    return sim


def cosine_adj_gender_thresholds(traits, same_cutoff=SAME_GENDER_CUTOFF,
                                 cross_cutoff=CROSS_GENDER_CUTOFF):
    """Same-gender pairs use a lower cutoff; cross-gender pairs use a higher one."""
    sim = trait_similarity(traits)
    same = genders[:, None] == genders[None, :]
    cross = ~same
    np.fill_diagonal(same, False)
    np.fill_diagonal(cross, False)

    adj = np.zeros((N_PEOPLE, N_PEOPLE), dtype=bool)
    adj[same]  = sim[same]  >= same_cutoff
    adj[cross] = sim[cross] >= cross_cutoff
    adj |= adj.T
    np.fill_diagonal(adj, False)
    return adj


def cosine_adj(traits, cutoff, female_bias=False):
    """Friendships wherever cosine similarity meets cutoff (symmetrised)."""
    sim = trait_similarity(traits)
    adj = sim >= cutoff
    if female_bias:
        ff = np.outer(female_mask, female_mask)
        adj[ff] = sim[ff] >= cutoff * FF_CUTOFF_FACTOR
    np.fill_diagonal(adj, False)
    adj |= adj.T
    np.fill_diagonal(adj, False)
    return adj


def similarity_fixed_friends(traits, n=4):
    """Top-n most similar peers at t=0, frozen for life."""
    sim = trait_similarity(traits)
    np.fill_diagonal(sim, -999.0)
    adj = np.zeros((N_PEOPLE, N_PEOPLE), dtype=bool)
    for i in range(N_PEOPLE):
        adj[i, np.argsort(-sim[i])[:n]] = True
    adj |= adj.T
    np.fill_diagonal(adj, False)
    return adj


def random_fixed_friends(n=4, seed=None):
    """n random peers at t=0, frozen for life."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((N_PEOPLE, N_PEOPLE), dtype=bool)
    for i in range(N_PEOPLE):
        pool = np.delete(np.arange(N_PEOPLE), i)
        adj[i, rng.choice(pool, n, replace=False)] = True
    adj |= adj.T
    np.fill_diagonal(adj, False)
    return adj


def innate_learn_rate(influence, traits):
    """Eager learning when influence matches innate interests; strong resistance otherwise."""
    match = np.clip(1.0 - np.abs(influence - INNATE_TRAITS) / 99.0, 0.0, 1.0)
    learn = np.where(
        match > 0.5,
        0.5 + match ** INNATE_MATCH_PWR,
        MISALIGNED_GAIN * match ** (INNATE_MATCH_PWR + 1),
    )
    toward_innate = INNATE_TRAITS - traits
    with_infl = influence - traits
    opposed = (np.sign(toward_innate) != np.sign(with_infl)) & (np.abs(toward_innate) > 2)
    learn = np.where(opposed, learn * OPPOSED_GAIN, learn)
    return learn


def innate_stickiness(traits):
    """Mobility toward influence: low when near innate (settled), higher when far."""
    dist = np.abs(traits - INNATE_TRAITS) / 99.0
    mobility = np.clip(dist * 2.0, 0.0, 1.0) ** INNATE_STICK_PWR
    return np.maximum(mobility, INNATE_STICK_FLOOR)


def update_traits(traits, adj, jobs, female_bias=False):
    fc = adj.sum(1)
    hf = fc > 0

    if female_bias:
        w   = np.ones((N_PEOPLE, N_PEOPLE))
        w[np.ix_(female_mask, female_mask)] = FF_INFLUENCE_MULT
        wf  = adj * w
        wfc = wf.sum(1)
        fa  = traits.copy()
        act = wfc > 0
        fa[act] = (wf @ traits)[act] / wfc[act, None]
    else:
        fa = traits.copy()
        fa[hf] = (adj @ traits)[hf] / fc[hf, None]

    ja = np.zeros_like(traits)
    for jid in range(3):
        m = jobs == jid; cnt = m.sum()
        if cnt > 1:
            ja[m] = (traits[m].mean(0) * cnt - traits[m]) / (cnt - 1)
        elif cnt == 1:
            ja[m] = traits[m]

    fw = np.where(hf[:, None], FRIEND_WEIGHT, 0.0)
    social_target = fw * fa + (1.0 - fw) * ja
    pull = social_target - traits
    gain = (innate_learn_rate(social_target, traits)
            * innate_stickiness(traits))
    delta = pull * gain
    scale = np.where(delta > 0, (100 - traits) / 100, traits / 100)
    new = traits + ALPHA * delta * scale
    new += np.random.normal(0, NOISE_STD, traits.shape)
    return np.clip(new, 0.0, 100.0)


def update_traits_random_only(traits):
    """Random influence each round — no friends, no job peers (trial run)."""
    rand_infl = np.clip(np.random.normal(50, RAND_INFL_STD, traits.shape), 1.0, 99.0)
    pull = rand_infl - traits
    gain = innate_learn_rate(rand_infl, traits) * innate_stickiness(traits)
    delta = pull * gain
    scale = np.where(delta > 0, (100 - traits) / 100, traits / 100)
    new = traits + ALPHA * delta * scale
    new += np.random.normal(0, NOISE_STD, traits.shape)
    return np.clip(new, 0.0, 100.0)

# ─── Reservation helpers ──────────────────────────────────────────────────────

def reserved_women_ids(traits, reservation_a_f=0, reservation_b_f=0):
    """Top-scoring women filling reserved Job A and/or Job B slots."""
    scores = traits[:, JOB_TRAITS].sum(axis=1)
    rank   = list(np.argsort(-scores))
    female_by_score = [i for i in rank if female_mask[i]]
    reserved_a = female_by_score[:reservation_a_f]
    reserved_b = female_by_score[reservation_a_f:reservation_a_f + reservation_b_f]
    return reserved_a, reserved_b


def friends_in_job(adj, person, jobs, job_id):
    peers = (jobs == job_id) & (np.arange(N_PEOPLE) != person)
    return int(adj[person, peers].sum())

# ─── Pre-run and record all frames ────────────────────────────────────────────

def precompute(scenario, n_iters):
    traits = INIT_TRAITS.copy()
    f_adj  = None
    if scenario == 'fixed':
        f_adj = similarity_fixed_friends(traits)
    elif scenario == 'random':
        f_adj = random_fixed_friends(n=4, seed=SEED)

    if scenario == 'reservation':
        reservation_a_f, reservation_b_f = JOB_A_RESERVATION_F, 0
    elif scenario == 'reservation_ab':
        reservation_a_f, reservation_b_f = JOB_A_RESERVATION_F, JOB_B_RESERVATION_F
    else:
        reservation_a_f, reservation_b_f = 0, 0

    frames = []
    for _ in range(n_iters):
        jobs = assign_jobs(traits, reservation_a_f, reservation_b_f)

        if scenario == 'no_friends':
            adj = np.zeros((N_PEOPLE, N_PEOPLE), dtype=bool)
        elif scenario == 'gender_cutoff':
            adj = cosine_adj_gender_thresholds(traits)
        elif f_adj is not None:
            adj = f_adj
        elif scenario == 'ff_bias':
            adj = cosine_adj(traits, BASE_CUTOFF, female_bias=True)
        else:
            adj = cosine_adj(traits, BASE_CUTOFF)

        cross = (np.outer(male_mask, female_mask) |
                 np.outer(female_mask, male_mask)) & adj

        pay    = np.array([JOB_PAY[j] for j in jobs], dtype=float)
        mp, fp = pay[male_mask].mean(), pay[female_mask].mean()

        frame = {
            'traits':  traits.copy(),
            'jobs':    jobs.copy(),
            'adj':     adj.copy(),
            'cross':   cross.copy(),
            'job_a_f': (jobs[female_mask] == 0).mean() * 100,
            'job_b_f': (jobs[female_mask] == 1).mean() * 100,
            'pay_pct': fp / mp * 100 if mp > 0 else 100.0,
            'avg_fri': adj.sum() / N_PEOPLE,
            'n_cross': cross.sum() // 2,
        }
        if reservation_a_f > 0 or reservation_b_f > 0:
            reserved_a, reserved_b = reserved_women_ids(
                traits, reservation_a_f, reservation_b_f)
            frame['reserved_a'] = reserved_a
            frame['reserved_b'] = reserved_b
            frame['reserved'] = reserved_a + reserved_b
            frame['reserved_in_a'] = [i for i in reserved_a if jobs[i] == 0]
            frame['reserved_in_b'] = [i for i in reserved_b if jobs[i] == 1]
        frames.append(frame)
        traits = (update_traits_random_only(traits) if scenario == 'no_friends'
                  else update_traits(traits, adj, jobs,
                                     female_bias=(scenario == 'ff_bias')))
    return frames


# ─── Node layout (PCA of initial traits → natural gender clusters) ─────────────

def make_layout():
    c = INIT_TRAITS - INIT_TRAITS.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    pos = c @ Vt[:2].T
    return pos / (np.abs(pos).max() + 1e-6) * 0.87   # normalise to ~[-0.87, 0.87]


# ─── Adjacency matrix → edge segment lists ────────────────────────────────────

def adj_to_segs(adj, pos, cross=None):
    rows, cols = np.where(np.triu(adj, k=1))
    normal, bridges = [], []
    for r, c_ in zip(rows, cols):
        seg = [pos[r].tolist(), pos[c_].tolist()]
        if cross is not None and cross[r, c_]:
            bridges.append(seg)
        else:
            normal.append(seg)
    return normal, bridges


# ─── Build the animation ──────────────────────────────────────────────────────

SCENARIO_LABELS = {
    'fixed':      'Simulation 1 — Fixed Similarity Friends',
    'dynamic':    'Simulation 2 — Dynamic Friends',
    'gender_cutoff': 'Gender Cutoffs — 0.7 Same-Gender / 0.9 Cross-Gender',
    'no_friends': 'Trial Run — Random Influence Only',
    'reservation': 'Gender Quota — 40% Job A Reserved for Women',
    'reservation_ab': 'Gender Quota — 40% Job A & Job B Reserved for Women',
    'ff_bias':    'Simulation — Dynamic + Female-Female Bias',
    'random':     'Simulation — Random Childhood Friends',
}

JOB_BAND_SCENARIOS = {'dynamic', 'gender_cutoff', 'no_friends', 'reservation', 'reservation_ab'}

# Script characters (person index → label)
HIGHLIGHT_CHARS = {91: 'Emma', 54: 'Rachel'}


def all_edge_segs(adj, pos):
    rows, cols = np.where(np.triu(adj, k=1))
    return [[pos[r].tolist(), pos[c].tolist()] for r, c in zip(rows, cols)]


def build_animation(scenario, frames, frame_step=2):
    anim_idxs = list(range(0, len(frames), frame_step))
    pos       = make_layout()
    n_iters   = len(frames)

    # ── Figure / axes layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    gs  = GridSpec(3, 10, figure=fig,
                   left=0.03, right=0.97, top=0.91, bottom=0.08,
                   wspace=0.6, hspace=0.7)

    ax_net = fig.add_subplot(gs[:, :6])   # network — left 60%
    ax_a   = fig.add_subplot(gs[0, 6:])  # Job A female %
    ax_b   = fig.add_subplot(gs[1, 6:])  # Job B female %
    ax_p   = fig.add_subplot(gs[2, 6:])  # Pay parity

    for ax in (ax_net, ax_a, ax_b, ax_p):
        ax.set_facecolor(C_BG)
        ax.tick_params(colors=C_MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a2a')

    ax_net.set_aspect('equal')
    ax_net.set_xlim(-1.0, 1.0)
    ax_net.set_ylim(-1.0, 1.0)
    ax_net.set_xticks([])
    ax_net.set_yticks([])

    fig.suptitle(SCENARIO_LABELS.get(scenario, scenario),
                 color=C_TEXT, fontsize=12, fontweight='bold', y=0.97)

    # ── Edge collections ──────────────────────────────────────────────────────
    f0             = frames[0]
    n_segs, b_segs = adj_to_segs(f0['adj'], pos, f0['cross'])

    lc_normal = LineCollection(
        n_segs or DUMMY_SEG,
        colors=C_EDGE, linewidths=0.3, alpha=0.12, zorder=1)
    lc_bridges = LineCollection(
        b_segs or DUMMY_SEG,
        colors=C_BRIDGE, linewidths=0.9, alpha=0.55, zorder=2)
    ax_net.add_collection(lc_normal)
    ax_net.add_collection(lc_bridges)

    # ── Node scatter ─────────────────────────────────────────────────────────
    # Use global score range so node sizes are comparable across all frames
    score_min = min(f['traits'][:, JOB_TRAITS].sum(1).min() for f in frames)
    score_max = max(f['traits'][:, JOB_TRAITS].sum(1).max() for f in frames)

    def node_sizes(traits):
        sc = traits[:, JOB_TRAITS].sum(1)
        return 22 + (sc - score_min) / (score_max - score_min + 1e-6) * 95

    face_colors = [C_MALE if m else C_FEMALE for m in male_mask]
    edge_colors = [JOB_COLORS[j] for j in f0['jobs']]

    sc = ax_net.scatter(
        pos[:, 0], pos[:, 1],
        s=node_sizes(f0['traits']),
        c=face_colors,
        edgecolors=edge_colors,
        linewidths=1.7,
        zorder=3)

    # ── Text overlays on network ──────────────────────────────────────────────
    iter_lbl = ax_net.text(
        0.02, 0.98, 'Iter   1',
        transform=ax_net.transAxes,
        color=C_TEXT, fontsize=11, va='top',
        fontfamily='monospace', fontweight='bold')

    stats_lbl = ax_net.text(
        0.02, 0.02,
        f"\u2640 Job A: {f0['job_a_f']:.0f}%  "
        f"Pay: {f0['pay_pct']:.0f}%  "
        f"Avg friends: {f0['avg_fri']:.1f}  "
        f"Bridges: {f0['n_cross']}",
        transform=ax_net.transAxes,
        color=C_BRIDGE, fontsize=8.5, va='bottom',
        fontfamily='monospace')

    # ── Legend ────────────────────────────────────────────────────────────────
    leg_items = [
        mpatches.Patch(facecolor=C_MALE,   label='Male node'),
        mpatches.Patch(facecolor=C_FEMALE, label='Female node'),
        mpatches.Patch(facecolor=C_JOB_A,  label='Job A  (pay 1000)'),
        mpatches.Patch(facecolor=C_JOB_B,  label='Job B  (pay 300)'),
        mpatches.Patch(facecolor=C_JOB_C,  label='Job C  (pay 100)'),
        mpatches.Patch(facecolor=C_BRIDGE, label='Cross-gender link'),
    ]
    ax_net.legend(
        handles=leg_items, loc='lower right',
        fontsize=7.5, framealpha=0.25,
        labelcolor=C_TEXT, facecolor='#1a1a1a', edgecolor='#333333')

    # ── Metric charts ─────────────────────────────────────────────────────────
    def style_ax(ax, title, ylabel, ylim, hline=None):
        ax.set_title(title, color=C_MUTED, fontsize=8, pad=3)
        ax.set_ylabel(ylabel, color=C_MUTED, fontsize=7)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, n_iters)
        ax.set_xlabel('Iteration', color=C_MUTED, fontsize=7)
        ax.grid(True, alpha=0.1, color='#444444', linewidth=0.5)
        if hline is not None:
            ax.axhline(hline, color='#3a3a3a', ls='--', lw=0.8)

    style_ax(ax_a, 'Women in Job A', '%', (0, 65),  hline=50)
    style_ax(ax_b, 'Women in Job B', '%', (0, 65),  hline=50)
    style_ax(ax_p, 'Female pay / Male pay', '%', (20, 115), hline=100)

    line_a, = ax_a.plot([], [], color=C_FEMALE,  lw=1.8)
    line_b, = ax_b.plot([], [], color='#9B59B6', lw=1.8)
    line_p, = ax_p.plot([], [], color=C_JOB_A,  lw=1.8)

    # Moving dot = current value on each chart
    dot_a = ax_a.scatter([], [], s=28, color=C_FEMALE,  zorder=5)
    dot_b = ax_b.scatter([], [], s=28, color='#9B59B6', zorder=5)
    dot_p = ax_p.scatter([], [], s=28, color=C_JOB_A,   zorder=5)

    # ── Animation update function ─────────────────────────────────────────────
    x_hist, ya_hist, yb_hist, yp_hist = [], [], [], []

    def update_frame(fi):
        idx = anim_idxs[fi]
        f   = frames[idx]
        it  = idx + 1

        # Edges
        ns, bs = adj_to_segs(f['adj'], pos, f['cross'])
        lc_normal.set_segments(ns  if ns  else DUMMY_SEG)
        lc_bridges.set_segments(bs if bs  else DUMMY_SEG)

        # Nodes: border = job, size = job-trait score (gender fill never changes)
        sc.set_edgecolor([JOB_COLORS[j] for j in f['jobs']])
        sc.set_sizes(node_sizes(f['traits']))

        # Text
        iter_lbl.set_text(f'Iter {it:>3d}')
        stats_lbl.set_text(
            f"\u2640 Job A: {f['job_a_f']:.0f}%  "
            f"Pay: {f['pay_pct']:.0f}%  "
            f"Avg friends: {f['avg_fri']:.1f}  "
            f"Bridges: {f['n_cross']}"
        )

        # Charts
        x_hist.append(it)
        ya_hist.append(f['job_a_f'])
        yb_hist.append(f['job_b_f'])
        yp_hist.append(f['pay_pct'])

        line_a.set_data(x_hist, ya_hist)
        line_b.set_data(x_hist, yb_hist)
        line_p.set_data(x_hist, yp_hist)

        dot_a.set_offsets([[it, f['job_a_f']]])
        dot_b.set_offsets([[it, f['job_b_f']]])
        dot_p.set_offsets([[it, f['pay_pct']]])

        return (lc_normal, lc_bridges, sc,
                iter_lbl, stats_lbl,
                line_a, line_b, line_p,
                dot_a, dot_b, dot_p)

    ani = animation.FuncAnimation(
        fig, update_frame,
        frames=len(anim_idxs),
        interval=33,          # ~30 fps (live preview)
        blit=True,
        repeat=False,
    )
    return fig, ani


# ─── Job-band layout (dynamic scenario) ───────────────────────────────────────

C_RED  = '#E53935'
C_BLUE = '#1E88E5'

Y_LO, Y_HI = 0.06, 0.94
Y_SPAN = Y_HI - Y_LO

BAND_META = [
    {'label': 'Job A', 'pay': 1000, 'fill': '#1a1a0a', 'edge': C_JOB_A},
    {'label': 'Job B', 'pay': 300,  'fill': '#121212', 'edge': C_JOB_B},
    {'label': 'Job C', 'pay': 100,  'fill': '#0a0a0a', 'edge': C_JOB_C},
]
JOB_LABELS = [b['label'] for b in BAND_META]


# Fixed job-band boundaries on the percentile axis (top 15% / top 50%)
PCT_JOB_A = 100.0 * (N_PEOPLE - JOB_A_COUNT) / N_PEOPLE
PCT_JOB_B = 100.0 * (N_PEOPLE - JOB_A_COUNT - JOB_B_COUNT) / N_PEOPLE


def pct_to_y(pct):
    return Y_LO + np.clip(pct / 100.0, 0.0, 1.0) * Y_SPAN


def job_trait_avg(traits):
    """Mean of the three job-relevant traits (hiring score / 3)."""
    return traits[:, JOB_TRAITS].mean(axis=1)


def trait_percentiles(traits):
    """Population percentile of each person's avg job-trait score (0–100)."""
    avgs = job_trait_avg(traits)
    order = np.argsort(avgs, kind='mergesort')
    ranks = np.empty(N_PEOPLE, dtype=float)
    ranks[order] = np.arange(N_PEOPLE)
    for val in np.unique(avgs):
        mask = avgs == val
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks / (N_PEOPLE - 1) * 100.0


def percentile_positions(traits):
    """Fixed x slot; y = job-trait percentile within the population."""
    pos = np.zeros((N_PEOPLE, 2))
    pos[:, 0] = np.linspace(-0.88, 0.88, N_PEOPLE)
    pos[:, 1] = pct_to_y(trait_percentiles(traits))
    return pos


def fixed_band_rects():
    """Job A/B/C bands at fixed percentile cutoffs."""
    y_a = pct_to_y(PCT_JOB_A)
    y_b = pct_to_y(PCT_JOB_B)
    return [
        (y_a, Y_HI - y_a),
        (y_b, y_a - y_b),
        (Y_LO, y_b - Y_LO),
    ]


# ─── Per-scenario animation profiles (keep pacing / overlays isolated) ─────────

def uniform_anim_indices(n_iters, step):
    """Linear stride — used by dynamic."""
    return list(range(0, n_iters, max(1, step)))


def accelerating_anim_indices(n_iters, target_frames, slow_first_n=100):
    """Linger on early iterations, then compress the rest (no_friends only)."""
    if n_iters <= 1:
        return [0]

    slow_first_n = min(slow_first_n, n_iters - 1)
    phase1 = list(range(slow_first_n))   # 1 animation frame per iter for opening stretch

    if slow_first_n >= n_iters - 1:
        return phase1

    remaining_budget = max(2, target_frames - len(phase1))
    t = np.linspace(0.0, 1.0, remaining_budget)
    tail = slow_first_n + (t ** 2 * (n_iters - 1 - slow_first_n)).astype(int)
    return sorted(set(phase1) | set(tail.tolist()))


JOBS_ANIM_PROFILES = {
    'dynamic': {
        'show_friend_edges': True,
        'allow_highlight': True,
        'highlight_mode': 'emma_rachel',
        'reservation_a_f': 0,
        'reservation_b_f': 0,
        'pick_indices': uniform_anim_indices,
    },
    'gender_cutoff': {
        'show_friend_edges': True,
        'allow_highlight': True,
        'highlight_mode': 'emma_rachel',
        'reservation_a_f': 0,
        'reservation_b_f': 0,
        'pick_indices': uniform_anim_indices,
    },
    'no_friends': {
        'show_friend_edges': False,
        'allow_highlight': False,
        'highlight_mode': None,
        'reservation_f': 0,
        'pick_indices': lambda n, step: accelerating_anim_indices(
            n, target_frames=max(200, n // max(1, step))),
    },
    'reservation': {
        'show_friend_edges': True,
        'allow_highlight': True,
        'highlight_mode': 'reserved',
        'reservation_a_f': JOB_A_RESERVATION_F,
        'reservation_b_f': 0,
        'pick_indices': uniform_anim_indices,
    },
    'reservation_ab': {
        'show_friend_edges': True,
        'allow_highlight': True,
        'highlight_mode': 'reserved',
        'reservation_a_f': JOB_A_RESERVATION_F,
        'reservation_b_f': JOB_B_RESERVATION_F,
        'pick_indices': uniform_anim_indices,
    },
}


def build_jobs_animation(scenario, frames, frame_step=5, highlight=False):
    """Dispatch to shared renderer with scenario-specific pacing config."""
    profile = JOBS_ANIM_PROFILES[scenario]
    if highlight and not profile['allow_highlight']:
        highlight = False
    anim_idxs = profile['pick_indices'](len(frames), frame_step)
    return _render_jobs_animation(
        scenario, frames, anim_idxs,
        show_friend_edges=profile['show_friend_edges'],
        highlight=highlight,
        highlight_mode=profile.get('highlight_mode'),
        reservation_a_f=profile.get('reservation_a_f', 0),
        reservation_b_f=profile.get('reservation_b_f', 0),
    )


def _add_quota_slot_markers(ax, band_idx, n_reserved, job_count, label, markers,
                            edge_color, fill_color):
    """Draw fenced-off quota seats on a job band."""
    if n_reserved <= 0:
        return
    y_band, h_band = fixed_band_rects()[band_idx]
    slot_h = h_band / job_count
    for slot in range(n_reserved):
        sy = y_band + h_band - (slot + 1) * slot_h
        rect = mpatches.Rectangle(
            (0.72, sy), 0.25, slot_h * 0.92,
            facecolor=fill_color, edgecolor=edge_color,
            linewidth=1.2, linestyle='--', alpha=0.75, zorder=1)
        ax.add_patch(rect)
        markers.append(rect)
        if slot == 0:
            lbl = ax.text(
                0.845, sy + slot_h * 0.46, label,
                color=edge_color, fontsize=7, ha='center', va='center',
                fontweight='bold', rotation=90, zorder=2)
            markers.append(lbl)


def _make_reserved_glow(ax, pts, color):
    """Layered halo + ring for a set of reserved women."""
    layers = []
    x = pts[:, 0] if len(pts) else []
    y = pts[:, 1] if len(pts) else []
    for size, alpha in ((340, 0.05), (240, 0.10), (160, 0.18)):
        layers.append(ax.scatter(
            x, y, s=size, c=color, alpha=alpha, linewidths=0, zorder=4))
    ring = ax.scatter(
        x, y, s=72, facecolors='none', edgecolors=color,
        linewidths=1.6, alpha=0.85, zorder=4)
    return layers, ring


def _update_glow_stack(layers, ring, display_pos, ids):
    pts = display_pos[ids] if ids else np.empty((0, 2))
    for layer in layers:
        layer.set_offsets(pts)
    if ring is not None:
        ring.set_offsets(pts)


def _render_jobs_animation(scenario, frames, anim_idxs, show_friend_edges, highlight,
                           highlight_mode=None, reservation_a_f=0, reservation_b_f=0):
    """Percentile layout shared by dynamic and no_friends."""
    total_frames = len(anim_idxs)

    fig, ax = plt.subplots(figsize=(12, 9), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([pct_to_y(v) for v in (0, 25, 50, 75, 100)])
    ax.set_yticklabels(['0', '25', '50', '75', '100'], color=C_MUTED, fontsize=8)
    ax.tick_params(axis='y', colors=C_MUTED, length=3, width=0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)

    y_axis_lbl = ax.text(
        -1.02, 0.5, 'Job-trait percentile',
        color=C_MUTED, fontsize=9, rotation=90,
        va='center', ha='center', zorder=1)

    # Job band backgrounds — fixed at 85th / 50th percentile cutoffs
    band_rect_patches = []
    band_label_artists = []
    band_artists = []
    f0 = frames[0]

    for meta, (yb, h) in zip(BAND_META, fixed_band_rects()):
        rect = mpatches.Rectangle(
            (-0.98, yb), 1.96, max(h, 0.001),
            facecolor=meta['fill'], edgecolor=meta['edge'],
            linewidth=1.5, alpha=0.55, zorder=0)
        ax.add_patch(rect)
        band_rect_patches.append(rect)
        band_artists.append(rect)
        cy = yb + h / 2
        lbl = ax.text(
            -0.99, cy, f"{meta['label']}  (pay {meta['pay']})",
            color=meta['edge'], fontsize=11, va='center', ha='left',
            fontweight='bold', zorder=1)
        band_label_artists.append(lbl)
        band_artists.append(lbl)

    # Reserved-slot markers on Job A / Job B bands
    reserved_markers = []
    _add_quota_slot_markers(
        ax, 0, reservation_a_f, JOB_A_COUNT, 'RESERVED', reserved_markers,
        C_RESERVED_A_RING, C_RESERVED_A_FILL)
    _add_quota_slot_markers(
        ax, 1, reservation_b_f, JOB_B_COUNT, 'RESERVED', reserved_markers,
        C_RESERVED_B_RING, C_RESERVED_B_FILL)

    face_colors = np.where(male_mask, C_BLUE, C_RED)
    pos0 = percentile_positions(f0['traits'])
    display_pos = pos0.copy()

    sc = ax.scatter(
        display_pos[:, 0], display_pos[:, 1],
        s=55, c=face_colors, edgecolors='#111111', linewidths=0.4, zorder=3)

    # Glow on reserved women — gold (Job A) and cyan (Job B)
    reserved_a_glow_layers, reserved_a_glow_ring = [], None
    reserved_b_glow_layers, reserved_b_glow_ring = [], None
    if reservation_a_f > 0:
        init_a = f0.get('reserved_a', f0.get('reserved', []))
        pts_a = display_pos[init_a] if init_a else np.empty((0, 2))
        reserved_a_glow_layers, reserved_a_glow_ring = _make_reserved_glow(
            ax, pts_a, C_RESERVED_A_GLOW)
    if reservation_b_f > 0:
        init_b = f0.get('reserved_b', [])
        pts_b = display_pos[init_b] if init_b else np.empty((0, 2))
        reserved_b_glow_layers, reserved_b_glow_ring = _make_reserved_glow(
            ax, pts_b, C_RESERVED_B_GLOW)

    # Friendship edges (none in no_friends scenario)
    lc_friend = None
    if show_friend_edges:
        lc_friend = LineCollection(
            DUMMY_SEG, colors=C_EDGE, linewidths=0.35, alpha=0.14, zorder=2)
        ax.add_collection(lc_friend)

    # Highlight overlays (Emma/Rachel or reserved women in Job A)
    hl_sc = None
    hl_labels = []
    hl_mode = highlight_mode if highlight else None

    def _init_highlight_rings(indices, ring_color, label_fn):
        nonlocal hl_sc
        if not indices:
            return
        hl_sc = ax.scatter(
            display_pos[indices, 0], display_pos[indices, 1],
            s=220, facecolors='none',
            edgecolors=ring_color, linewidths=2.8, zorder=5)
        for pid in indices:
            lbl = ax.text(
                0, 0, label_fn(pid),
                color=ring_color, fontsize=9, fontweight='bold',
                ha='center', va='bottom', zorder=6)
            hl_labels.append((pid, lbl))

    if hl_mode == 'emma_rachel':
        _init_highlight_rings(
            sorted(HIGHLIGHT_CHARS),
            C_HIGHLIGHT_RING,
            lambda pid: HIGHLIGHT_CHARS[pid],
        )
    elif hl_mode == 'reserved' and f0.get('reserved'):
        _init_highlight_rings(
            f0['reserved'],
            C_RESERVED_A_GLOW,
            lambda pid: HIGHLIGHT_CHARS.get(pid, 'Reserved'),
        )

    title = ax.set_title(
        SCENARIO_LABELS.get(scenario, scenario),
        color=C_TEXT, fontsize=13, fontweight='bold', pad=12)
    fig.subplots_adjust(top=0.92)

    iter_lbl = ax.text(
        0.98, 0.98, 'Iter   1',
        transform=ax.transAxes, color=C_TEXT, fontsize=12,
        va='top', ha='right', fontfamily='monospace', fontweight='bold')

    stats_lbl = ax.text(
        0.98, 0.04,
        '',
        transform=ax.transAxes, color=C_MUTED, fontsize=10,
        va='bottom', ha='right', fontfamily='monospace')

    leg_items = [
        mpatches.Patch(facecolor=C_BLUE, label='Blue (male)'),
        mpatches.Patch(facecolor=C_RED,  label='Red (female)'),
    ]
    if show_friend_edges:
        leg_items.append(mpatches.Patch(facecolor=C_EDGE, label='Friendship'))
    if reservation_a_f > 0:
        leg_items.append(
            mpatches.Patch(facecolor=C_RESERVED_A_GLOW, alpha=0.55,
                           label='Reserved Job A'))
    if reservation_b_f > 0:
        leg_items.append(
            mpatches.Patch(facecolor=C_RESERVED_B_GLOW, alpha=0.55,
                           label='Reserved Job B'))
    if hl_mode == 'emma_rachel':
        leg_items.append(
            mpatches.Patch(facecolor='none', edgecolor=C_HIGHLIGHT_RING,
                           linewidth=2, label='Emma / Rachel'))
    elif hl_mode == 'reserved':
        leg_items.append(
            mpatches.Patch(facecolor='none', edgecolor=C_RESERVED_A_GLOW,
                           linewidth=2, label='Reserved women'))
        if reservation_b_f > 0:
            leg_items.append(
                mpatches.Patch(facecolor=C_RESERVED_B_FILL, edgecolor=C_RESERVED_B_RING,
                               linewidth=1, linestyle='--', label='Job B quota slots'))
        leg_items.append(
            mpatches.Patch(facecolor=C_RESERVED_A_FILL, edgecolor=C_RESERVED_A_RING,
                           linewidth=1, linestyle='--', label='Job A quota slots'))
    leg = ax.legend(
        handles=leg_items, loc='upper left',
        fontsize=9, framealpha=0.25,
        labelcolor=C_TEXT, facecolor='#1a1a1a', edgecolor='#333333')

    def _update_friend_edges(adj, pos):
        if lc_friend is None:
            return
        segs = all_edge_segs(adj, pos)
        lc_friend.set_segments(segs if segs else DUMMY_SEG)

    def _place_highlight_labels():
        for pid, lbl in hl_labels:
            x, y = display_pos[pid]
            lbl.set_position((x, y + 0.045))
            if hl_mode == 'emma_rachel':
                lbl.set_text(HIGHLIGHT_CHARS[pid])
            else:
                lbl.set_text(HIGHLIGHT_CHARS.get(pid, 'Reserved'))

    def update_frame(fi):
        nonlocal display_pos

        idx = anim_idxs[fi]
        f   = frames[idx]
        it  = idx + 1

        target = percentile_positions(f['traits'])
        display_pos += (target - display_pos) * 0.35

        sc.set_offsets(display_pos)
        sc.set_facecolors(face_colors)
        sc.set_edgecolors('#111111')
        sc.set_sizes(np.full(N_PEOPLE, 55))

        _update_friend_edges(f['adj'], display_pos)

        if reservation_a_f > 0:
            _update_glow_stack(
                reserved_a_glow_layers, reserved_a_glow_ring,
                display_pos, f.get('reserved_a', []))
        if reservation_b_f > 0:
            _update_glow_stack(
                reserved_b_glow_layers, reserved_b_glow_ring,
                display_pos, f.get('reserved_b', []))

        if highlight and hl_mode == 'emma_rachel':
            hl_idx = sorted(HIGHLIGHT_CHARS)
            if hl_sc is not None:
                hl_sc.set_offsets(display_pos[hl_idx])
            _place_highlight_labels()
            title.set_text(SCENARIO_LABELS.get(scenario, scenario))
            leg.set_visible(True)

            emma_fri = f['adj'][91].sum()
            rachel_fri = f['adj'][54].sum()
            emma_job = JOB_LABELS[f['jobs'][91]]
            rachel_job = JOB_LABELS[f['jobs'][54]]
            extra = (f"   Emma→{emma_job} ({int(emma_fri)} friends)   "
                     f"Rachel→{rachel_job} ({int(rachel_fri)} friends)")
        elif highlight and hl_mode == 'reserved':
            tracked_reserved = f.get('reserved', [])
            if hl_sc is not None:
                if tracked_reserved:
                    hl_sc.set_offsets(display_pos[tracked_reserved])
                    hl_sc.set_edgecolors(C_RESERVED_A_GLOW)
                else:
                    hl_sc.set_offsets(np.empty((0, 2)))
            tracked = {pid for pid, _ in hl_labels}
            for pid in tracked - set(tracked_reserved):
                lbl = next(l for p, l in hl_labels if p == pid)
                lbl.remove()
            hl_labels[:] = [(p, l) for p, l in hl_labels if p in tracked_reserved]
            for pid in tracked_reserved:
                if pid not in {p for p, _ in hl_labels}:
                    lbl = ax.text(
                        0, 0, HIGHLIGHT_CHARS.get(pid, 'Reserved'),
                        color=C_RESERVED_A_GLOW, fontsize=9, fontweight='bold',
                        ha='center', va='bottom', zorder=6)
                    hl_labels.append((pid, lbl))
            _place_highlight_labels()
            title.set_text(SCENARIO_LABELS.get(scenario, scenario))
            leg.set_visible(True)

            in_a = f.get('reserved_in_a', [])
            in_b = f.get('reserved_in_b', [])
            if reservation_b_f > 0:
                extra = (f"   Reserved→A:{len(in_a)}  B:{len(in_b)}")
            elif in_a:
                pid = in_a[0]
                fri_a = friends_in_job(f['adj'], pid, f['jobs'], 0)
                fri_all = int(f['adj'][pid].sum())
                extra = (f"   Reserved→Job A  "
                         f"friends in A: {fri_a}  elsewhere: {fri_all - fri_a}")
            else:
                extra = ''
        else:
            extra = ''
            title.set_text(SCENARIO_LABELS.get(scenario, scenario))
            leg.set_visible(True)

        n_a = (f['jobs'] == 0).sum()
        n_b = (f['jobs'] == 1).sum()
        n_c = (f['jobs'] == 2).sum()
        f_a = (f['jobs'][female_mask] == 0).sum()
        f_b = (f['jobs'][female_mask] == 1).sum()
        f_c = (f['jobs'][female_mask] == 2).sum()

        iter_lbl.set_text(f'Iter {it:>3d}')
        stats_lbl.set_text(
            f"Job A: {n_a} ({f_a} red)   "
            f"Job B: {n_b} ({f_b} red)   "
            f"Job C: {n_c} ({f_c} red)   "
            f"Pay: {f['pay_pct']:.0f}%{extra}"
        )

        artists = [sc, iter_lbl, stats_lbl, title, y_axis_lbl]
        if lc_friend is not None:
            artists.append(lc_friend)
        artists.extend(band_rect_patches)
        artists.extend(band_label_artists)
        artists.extend(reserved_markers)
        artists.extend(reserved_a_glow_layers)
        if reserved_a_glow_ring is not None:
            artists.append(reserved_a_glow_ring)
        artists.extend(reserved_b_glow_layers)
        if reserved_b_glow_ring is not None:
            artists.append(reserved_b_glow_ring)
        if highlight:
            artists.append(hl_sc)
            artists.extend(lbl for _, lbl in hl_labels)
        return artists

    ani = animation.FuncAnimation(
        fig, update_frame,
        frames=total_frames,
        interval=33,
        blit=False,
        repeat=False,
    )
    return fig, ani


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Patriarchy Simulation — live animated network view.')
    parser.add_argument('--scenario', default='fixed',
                        choices=['fixed', 'dynamic', 'gender_cutoff', 'no_friends',
                                 'reservation', 'reservation_ab', 'ff_bias', 'random'],
                        help='Which scenario to animate (default: fixed)')
    parser.add_argument('--iters', type=int, default=8000,
                        help='Number of simulation iterations (default: 8000)')
    parser.add_argument('--step', type=int, default=5,
                        help='Animate every N iterations (default: 5)')
    parser.add_argument('--save', action='store_true',
                        help='Save to MP4 instead of displaying live')
    parser.add_argument('--highlight', action='store_true',
                        help='Track Emma/Rachel (dynamic) or reserved women (reservation*)')
    args = parser.parse_args()

    print(f'Pre-computing {args.iters} iterations  [scenario: {args.scenario}]...')
    frames = precompute(args.scenario, args.iters)
    print(f'Done. {len(frames)} frames recorded. Building animation...')

    if args.scenario in JOB_BAND_SCENARIOS:
        profile = JOBS_ANIM_PROFILES[args.scenario]
        use_highlight = args.highlight and profile['allow_highlight']
        if args.highlight and not profile['allow_highlight']:
            print('Note: --highlight not available for this scenario; ignoring.')
        fig, ani = build_jobs_animation(
            args.scenario, frames, frame_step=args.step, highlight=use_highlight)
    else:
        fig, ani = build_animation(args.scenario, frames, frame_step=args.step)

    if args.save:
        profile = JOBS_ANIM_PROFILES.get(args.scenario, {})
        suffix = '_highlight' if (args.highlight and profile.get('allow_highlight')) else ''
        out    = f'patriarchy_anim_{args.scenario}{suffix}.mp4'
        writer = animation.FFMpegWriter(fps=30, bitrate=2000)
        print(f'Saving → {out}')
        ani.save(out, writer=writer, dpi=120,
                 savefig_kwargs={'facecolor': C_BG})
        print('Saved.')
    else:
        plt.show()


if __name__ == '__main__':
    main()
