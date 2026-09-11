# -*- coding: utf-8 -*-
"""
Patriarchy Simulation - Multi-Scenario Comparison
==================================================
Scenario 1: Fixed 4 friends (static network — same circle throughout)
Scenario 2: Dynamic friendships + female-female formation & influence bias
Scenario 3: Dynamic friendships, no gender bias (baseline)
Scenario 4: Baseline + 10% Job A reservation for women (2 of 15 slots)
Scenario 5: Cutoff sweep — progressively more open-minded
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ─── Global parameters ────────────────────────────────────────────────────────
N_MALE            = 50
N_FEMALE          = 50
N_PEOPLE          = N_MALE + N_FEMALE
N_TRAITS          = 10
JOB_TRAITS        = [0, 1, 2]
JOB_A_COUNT       = 15
JOB_B_COUNT       = 35
ALPHA             = 0.10
NOISE_STD         = 0.08
N_ITERATIONS      = 8000
SEED              = 42
BASE_CUTOFF       = 0.75
SAME_GENDER_CUTOFF  = 0.70
CROSS_GENDER_CUTOFF = 0.90

# Female bias parameters (Scenario 2)
FF_CUTOFF_FACTOR  = 0.70     # F-F friendship threshold = cutoff * 0.70 (easier to befriend)
FF_INFLUENCE_MULT = 2.0      # female friends get 2x weight in influence avg for women

# Pay per job
JOB_PAY = {0: 1000, 1: 300, 2: 100}

# Job A reservation (Scenario 4)
JOB_A_RESERVATION_F = 2     # 2 of 15 slots reserved for women (~10%)

# Scenario 5 cutoffs (high = restrictive, low = open-minded)
S5_CUTOFFS = [0.80, 0.70, 0.60, 0.50, 0.40, 0.30]

OTHER_TRAITS = [i for i in range(N_TRAITS) if i not in JOB_TRAITS]
genders      = np.array([0]*N_MALE + [1]*N_FEMALE)
male_mask    = genders == 0
female_mask  = genders == 1

# ─── Generate shared initial state (same for all scenarios) ──────────────────
np.random.seed(SEED)

_t = np.zeros((N_PEOPLE, N_TRAITS))
_t[:N_MALE, :3]  = np.random.normal(57, 15, (N_MALE, 3))   # male bias on job traits
_t[:N_MALE, 3:]  = np.random.normal(50, 12, (N_MALE, 7))
_t[N_MALE:, :3]  = np.random.normal(43, 15, (N_FEMALE, 3)) # female bias against job traits
_t[N_MALE:, 3:]  = np.random.normal(50, 12, (N_FEMALE, 7))
INITIAL_TRAITS = np.clip(_t, 1.0, 99.0)

# Sample innates for men only; each woman gets the same innate profile as a paired man
_male_innate = np.clip(np.random.normal(50, 12, (N_MALE, N_TRAITS)), 1.0, 99.0)
_pairing = np.random.permutation(N_MALE)   # woman i shares innates with man _pairing[i]
INNATE_TRAITS = np.zeros((N_PEOPLE, N_TRAITS))
INNATE_TRAITS[male_mask] = _male_innate
INNATE_TRAITS[female_mask] = _male_innate[_pairing]

# ─── Core functions ───────────────────────────────────────────────────────────

def assign_jobs(traits, reservation_f=0):
    """Rank by job-trait sum. Top JOB_A_COUNT -> A, next JOB_B_COUNT -> B, rest -> C.
    Optional: reserve `reservation_f` Job A slots for top-scoring women."""
    scores = traits[:, JOB_TRAITS].sum(axis=1)
    rank   = list(np.argsort(-scores))
    jobs   = np.full(N_PEOPLE, 2, dtype=int)

    if reservation_f > 0:
        female_by_score = [i for i in rank if female_mask[i]]
        reserved        = female_by_score[:reservation_f]
        open_pool       = [i for i in rank if i not in reserved]
        job_a           = reserved + open_pool[:JOB_A_COUNT - reservation_f]
        job_b_pool      = [i for i in rank if i not in job_a]
        jobs[job_a]                   = 0
        jobs[job_b_pool[:JOB_B_COUNT]] = 1
    else:
        jobs[rank[:JOB_A_COUNT]]                         = 0
        jobs[rank[JOB_A_COUNT:JOB_A_COUNT + JOB_B_COUNT]] = 1
    return jobs


def compute_friendships_dynamic(traits, cutoff, female_bias=False):
    """Cosine similarity of centered traits. Optional lower threshold for F-F pairs."""
    centered  = traits - 50.0
    norms     = np.linalg.norm(centered, axis=1, keepdims=True)
    safe_norm = np.where(norms < 1e-6, 1e-6, norms)
    normed    = centered / safe_norm
    sim       = normed @ normed.T
    np.fill_diagonal(sim, 0.0)

    adj = sim >= cutoff
    if female_bias:
        ff_mask       = np.outer(female_mask, female_mask)
        adj[ff_mask]  = sim[ff_mask] >= (cutoff * FF_CUTOFF_FACTOR)
        np.fill_diagonal(adj, False)
    return adj


def compute_friendships_gender_cutoff(traits, same_cutoff=SAME_GENDER_CUTOFF,
                                     cross_cutoff=CROSS_GENDER_CUTOFF):
    """Same-gender pairs use a lower cutoff; cross-gender pairs use a higher one."""
    centered  = traits - 50.0
    norms     = np.linalg.norm(centered, axis=1, keepdims=True)
    safe_norm = np.where(norms < 1e-6, 1e-6, norms)
    normed    = centered / safe_norm
    sim       = normed @ normed.T
    np.fill_diagonal(sim, 0.0)

    same = genders[:, None] == genders[None, :]
    cross = ~same
    np.fill_diagonal(same, False)
    np.fill_diagonal(cross, False)

    adj = np.zeros((N_PEOPLE, N_PEOPLE), dtype=bool)
    adj[same]  = sim[same]  >= same_cutoff
    adj[cross] = sim[cross] >= cross_cutoff
    return adj


def compute_friendships_fixed(traits, cutoff):
    """Compute friendships from initial traits using the same cutoff as dynamic scenarios.
    Network is frozen at t=0 and never updated."""
    return compute_friendships_dynamic(traits, cutoff, female_bias=False)


def compute_childhood_friends(traits, n=4):
    """Each person's top-n most similar people at t=0, kept for life.
    Uses top-n directly (no cutoff) so everyone is guaranteed n friends."""
    centered  = traits - 50.0
    norms     = np.linalg.norm(centered, axis=1, keepdims=True)
    safe_norm = np.where(norms < 1e-6, 1e-6, norms)
    normed    = centered / safe_norm
    sim       = normed @ normed.T
    np.fill_diagonal(sim, -999.0)

    adj = np.zeros((N_PEOPLE, N_PEOPLE), dtype=bool)
    for i in range(N_PEOPLE):
        top = np.argsort(-sim[i])[:n]
        adj[i, top] = True
    adj = adj | adj.T   # symmetric: if A picked B, B also has A
    np.fill_diagonal(adj, False)
    return adj


def compute_random_childhood_friends(n=4, seed=None):
    """Each person's n childhood friends chosen uniformly at random (not by similarity).
    Network is frozen at t=0 and never updated."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((N_PEOPLE, N_PEOPLE), dtype=bool)
    for i in range(N_PEOPLE):
        candidates = [j for j in range(N_PEOPLE) if j != i]
        chosen = rng.choice(candidates, size=n, replace=False)
        adj[i, chosen] = True
    adj = adj | adj.T
    np.fill_diagonal(adj, False)
    return adj


def compute_friendships_disjoint_cliques(traits, clique_size=4):
    """
    Partition all N people into disjoint cliques of `clique_size` by similarity.
    Greedy: repeatedly find the unassigned person with the highest total similarity
    to other unassigned people, then pull in their most similar unassigned peers
    to fill the clique. No cross-clique edges — zero cross-pollination.
    """
    centered  = traits - 50.0
    norms     = np.linalg.norm(centered, axis=1, keepdims=True)
    safe_norm = np.where(norms < 1e-6, 1e-6, norms)
    normed    = centered / safe_norm
    sim       = normed @ normed.T
    np.fill_diagonal(sim, -999.0)   # exclude self

    unassigned = list(range(N_PEOPLE))
    adj        = np.zeros((N_PEOPLE, N_PEOPLE), dtype=bool)

    while len(unassigned) >= clique_size:
        # Pick seed: unassigned person with highest avg similarity to other unassigned
        sub_sim   = sim[np.ix_(unassigned, unassigned)]
        avg_sim   = sub_sim.mean(axis=1)
        seed_idx  = unassigned[int(np.argmax(avg_sim))]

        # Pick clique_size-1 most similar unassigned peers to seed
        peers     = [u for u in unassigned if u != seed_idx]
        peers_sim = sim[seed_idx, peers]
        top_peers = [peers[i] for i in np.argsort(-peers_sim)[:clique_size - 1]]

        clique = [seed_idx] + top_peers
        for i in clique:
            for j in clique:
                if i != j:
                    adj[i, j] = True
            unassigned.remove(i)

    # Any remainder (if N not divisible by clique_size) forms a smaller clique
    if len(unassigned) > 1:
        for i in unassigned:
            for j in unassigned:
                if i != j:
                    adj[i, j] = True

    return adj


def update_traits(traits, friendships, jobs, female_bias=False, job_weight=0.5):
    """friend influence + job-peer influence (split controlled by job_weight), with innate resistance and noise."""
    friend_counts = friendships.sum(axis=1)
    has_friends   = friend_counts > 0

    # Friend average (with optional F-F influence upweight for women)
    if female_bias:
        w              = np.ones((N_PEOPLE, N_PEOPLE))
        w[np.ix_(female_mask, female_mask)] = FF_INFLUENCE_MULT
        wf             = friendships * w
        wf_counts      = wf.sum(axis=1)
        friend_avg     = traits.copy()
        active         = wf_counts > 0
        friend_avg[active] = (wf @ traits)[active] / wf_counts[active, None]
    else:
        friend_avg = traits.copy()
        friend_avg[has_friends] = (
            (friendships @ traits)[has_friends] / friend_counts[has_friends, None]
        )

    # Job peer average (leave-one-out)
    job_avg = np.zeros_like(traits)
    for job_id in [0, 1, 2]:
        mask = jobs == job_id
        n    = mask.sum()
        if n > 1:
            gm = traits[mask].mean(axis=0)
            job_avg[mask] = (gm * n - traits[mask]) / (n - 1)
        elif n == 1:
            job_avg[mask] = traits[mask]

    # Combined social delta
    fw    = np.where(has_friends[:, None], 1.0 - job_weight, 0.0)
    jw    = 1.0 - fw
    delta = fw * (friend_avg - traits) + jw * (job_avg - traits)

    # Innate resistance: harder to drift further from innate baseline
    resistance = 1.0 - np.abs(traits - INNATE_TRAITS) / 99.0
    delta      = delta * resistance

    # Boundary dampening
    scale      = np.where(delta > 0, (100 - traits) / 100, traits / 100)
    new_traits = traits + ALPHA * delta * scale

    # Random life shocks
    new_traits += np.random.normal(0, NOISE_STD, traits.shape)
    return np.clip(new_traits, 0.0, 100.0)


# ─── Simulation runner ────────────────────────────────────────────────────────

def run_simulation(label, friendship_mode='dynamic', cutoff=BASE_CUTOFF,
                   female_bias=False, reservation_f=0, run_seed=None, job_weight=0.5):
    if run_seed is not None:
        np.random.seed(run_seed)

    traits = INITIAL_TRAITS.copy()

    fixed_adj    = None
    childhood_adj = None
    if friendship_mode == 'fixed':
        fixed_adj = compute_childhood_friends(traits, n=4)
    elif friendship_mode == 'random_fixed':
        fixed_adj = compute_random_childhood_friends(n=4, seed=run_seed)
    elif friendship_mode == 'cliques':
        fixed_adj = compute_friendships_disjoint_cliques(traits)
    elif friendship_mode == 'hybrid':
        childhood_adj = compute_childhood_friends(traits, n=4)

    hist = {
        'job_a_female': [], 'job_b_female': [], 'job_c_female': [],
        'male_pay': [], 'female_pay': [],
        'avg_friends': [],
    }

    for _ in range(N_ITERATIONS):
        jobs = assign_jobs(traits, reservation_f)

        if friendship_mode in ('fixed', 'random_fixed', 'cliques'):
            adj = fixed_adj
        elif friendship_mode == 'hybrid':
            adj = compute_friendships_dynamic(traits, cutoff, female_bias) | childhood_adj
        elif friendship_mode == 'gender_cutoff':
            adj = compute_friendships_gender_cutoff(traits)
        else:
            adj = compute_friendships_dynamic(traits, cutoff, female_bias)

        for job_id, key in [(0,'job_a'), (1,'job_b'), (2,'job_c')]:
            n_job = (jobs == job_id).sum()
            hist[f'{key}_female'].append(
                (jobs[female_mask] == job_id).sum() / n_job if n_job > 0 else 0)

        pay = np.array([JOB_PAY[j] for j in jobs])
        hist['male_pay'].append(pay[male_mask].mean())
        hist['female_pay'].append(pay[female_mask].mean())
        hist['avg_friends'].append(adj.sum(axis=1).mean())

        traits = update_traits(traits, adj, jobs, female_bias, job_weight)

    mp_i, fp_i = hist['male_pay'][0],  hist['female_pay'][0]
    mp_f, fp_f = hist['male_pay'][-1], hist['female_pay'][-1]
    pct_i = fp_i / mp_i * 100 if mp_i > 0 else 100
    pct_f = fp_f / mp_f * 100 if mp_f > 0 else 100
    print(f"  {label:<38}  "
          f"JobA F%: {hist['job_a_female'][0]*100:4.0f}%->{hist['job_a_female'][-1]*100:4.0f}%  "
          f"JobB F%: {hist['job_b_female'][0]*100:4.0f}%->{hist['job_b_female'][-1]*100:4.0f}%  |  "
          f"F pay% of M: {pct_i:.0f}%->{pct_f:.0f}%  |  "
          f"pay gap: {mp_i-fp_i:.0f}->{mp_f-fp_f:.0f}  |  "
          f"friends: {hist['avg_friends'][0]:.1f}->{hist['avg_friends'][-1]:.1f}")
    return hist


# ─── Run all scenarios ────────────────────────────────────────────────────────
print("=" * 80)
print("PATRIARCHY SIMULATION — MULTI-SCENARIO COMPARISON")
print("=" * 80)

s1_to_s4 = {}
s1_to_s4['Fixed-2-Childhood (no new)']        = run_simulation('Fixed-2-Childhood (no new)',        friendship_mode='fixed',        run_seed=101)
s1_to_s4['Fixed-Random-Childhood (no new)']   = run_simulation('Fixed-Random-Childhood (no new)',   friendship_mode='random_fixed', run_seed=101)
s1_to_s4['S1b: Fixed disjoint cliques of 4'] = run_simulation('S1b: Fixed disjoint cliques of 4', friendship_mode='cliques', run_seed=101)
s1_to_s4['S1c: 2 Childhood + dynamic']        = run_simulation('S1c: 2 Childhood + dynamic',        friendship_mode='hybrid',  run_seed=101)
s1_to_s4['S2: Dynamic + F-F bias']           = run_simulation('S2: Dynamic + F-F bias',           female_bias=True,          run_seed=102)
s1_to_s4['S3: Baseline']                     = run_simulation('S3: Baseline',                     run_seed=103)
s1_to_s4['S3b: Gender cutoffs 0.7/0.9']      = run_simulation('S3b: Gender cutoffs 0.7/0.9',      friendship_mode='gender_cutoff', run_seed=103)
s1_to_s4['S4: Baseline + reservation']       = run_simulation('S4: Baseline + reservation',       reservation_f=JOB_A_RESERVATION_F, run_seed=104)

s5 = {}
for c in S5_CUTOFFS:
    lbl = f'cutoff={c:.2f}'
    s5[lbl] = run_simulation(f'S5 {lbl}', cutoff=c, run_seed=200 + int(c*100))

print("=" * 80)

# ─── Plot ─────────────────────────────────────────────────────────────────────
iters      = np.arange(N_ITERATIONS)
col_s1s4   = ['steelblue', 'deepskyblue', 'royalblue', 'mediumpurple', 'darkorange', 'forestgreen', 'teal', 'crimson']
col_s5     = plt.cm.plasma(np.linspace(0.1, 0.85, len(S5_CUTOFFS)))

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Patriarchy Simulation — Multi-Scenario Comparison", fontsize=14, fontweight='bold')
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)

# Top-left: Job A female share, S1-S4
ax = fig.add_subplot(gs[0, 0])
for (lbl, h), c in zip(s1_to_s4.items(), col_s1s4):
    ax.plot(iters, np.array(h['job_a_female'])*100, color=c, lw=2, label=lbl)
ax.axhline(50, color='gray', ls=':', alpha=0.5, label='Equal (50%)')
ax.set(xlabel='Iteration', ylabel='% of Job A',
       title='Job A — Female Share\nScenarios 1–4', ylim=(-2, 102))
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# Bottom-left: Pay gap, S1-S4
ax = fig.add_subplot(gs[1, 0])
for (lbl, h), c in zip(s1_to_s4.items(), col_s1s4):
    female_pct = np.array(h['female_pay']) / np.array(h['male_pay']) * 100
    ax.plot(iters, female_pct, color=c, lw=2, label=lbl)
ax.axhline(100, color='gray', ls=':', alpha=0.5, label='Equal pay (100%)')
ax.set(xlabel='Iteration', ylabel='Female pay as % of male pay',
       title='Pay Gap (F pay / M pay)\nScenarios 1–4')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# Top-right: Job A female share, S5
ax = fig.add_subplot(gs[0, 1])
for (lbl, h), c in zip(s5.items(), col_s5):
    ax.plot(iters, np.array(h['job_a_female'])*100, color=c, lw=2, label=lbl)
ax.axhline(50, color='gray', ls=':', alpha=0.5, label='Equal (50%)')
ax.set(xlabel='Iteration', ylabel='% of Job A',
       title='Job A — Female Share\nScenario 5: Openness Sweep', ylim=(-2, 102))
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# Bottom-right: Pay gap, S5
ax = fig.add_subplot(gs[1, 1])
for (lbl, h), c in zip(s5.items(), col_s5):
    female_pct = np.array(h['female_pay']) / np.array(h['male_pay']) * 100
    ax.plot(iters, female_pct, color=c, lw=2, label=lbl)
ax.axhline(100, color='gray', ls=':', alpha=0.5, label='Equal pay (100%)')
ax.set(xlabel='Iteration', ylabel='Female pay as % of male pay',
       title='Pay Gap (F pay / M pay)\nScenario 5: Openness Sweep')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

plt.savefig('patriarchy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved -> patriarchy_comparison.png")

# ─── Final summary table (averaged over multiple seeds) ──────────────────────
CHECKPOINTS  = [20, 100, 300, 600]
N_RUNS       = 20   # independent runs per scenario

def snap(h, t):
    i  = min(t - 1, len(h['job_a_female']) - 1)
    mp = h['male_pay'][i]
    fp = h['female_pay'][i]
    return {
        'job_a': h['job_a_female'][i] * 100,
        'job_b': h['job_b_female'][i] * 100,
        'pay':   fp / mp * 100 if mp > 0 else 100,
    }

key_configs = [
    ('Fixed-2-Childhood',            dict(friendship_mode='fixed')),
    ('Fixed-Random-Childhood',       dict(friendship_mode='random_fixed')),
    ('Childhood + Dynamic',          dict(friendship_mode='hybrid')),
    ('Only Dynamic',                 dict()),
    ('Dynamic + F-Bias',             dict(female_bias=True)),
    ('Dynamic + Reservation',        dict(reservation_f=JOB_A_RESERVATION_F)),
    ('Child + Dyn + Res',            dict(friendship_mode='hybrid', reservation_f=JOB_A_RESERVATION_F)),
    ('Child + Dyn + Res + Open-60',  dict(friendship_mode='hybrid', reservation_f=JOB_A_RESERVATION_F, cutoff=0.60)),
    ('Child + Dyn + Res + Open-40',  dict(friendship_mode='hybrid', reservation_f=JOB_A_RESERVATION_F, cutoff=0.40)),
    ('Child + Dyn + Res + Open-20',  dict(friendship_mode='hybrid', reservation_f=JOB_A_RESERVATION_F, cutoff=0.20)),
    ('Child+Dyn+20%Res+Open-60',     dict(friendship_mode='hybrid', reservation_f=3, cutoff=0.60)),
    ('Child+Dyn+20%Res+Open-40',     dict(friendship_mode='hybrid', reservation_f=3, cutoff=0.40)),
    ('Child+Dyn+20%Res+Open-20',     dict(friendship_mode='hybrid', reservation_f=3, cutoff=0.20)),
]

print('\nAveraging over', N_RUNS, 'independent runs per scenario...')
averaged = {}
for name, kwargs in key_configs:
    runs = [run_simulation(f'{name} run {s}', run_seed=1000+s, **kwargs)
            for s in range(N_RUNS)]
    # Average across runs for each checkpoint metric
    averaged[name] = {}
    for t in CHECKPOINTS:
        snaps = [snap(r, t) for r in runs]
        averaged[name][t] = {
            'job_a': np.mean([s['job_a'] for s in snaps]),
            'job_b': np.mean([s['job_b'] for s in snaps]),
            'pay':   np.mean([s['pay']   for s in snaps]),
        }

col_w = 14
hdr_w = 26
sep   = '-' * (hdr_w + col_w * len(CHECKPOINTS) * 3)

print('\n' + '=' * len(sep))
print(f'FINAL RESULTS SUMMARY  (mean over {N_RUNS} runs, noise={NOISE_STD})')
print('=' * len(sep))
header1 = f"{'':>{hdr_w}}" + ''.join(f"{'Iter ' + str(t):^{col_w*3}}" for t in CHECKPOINTS)
header2 = f"{'Simulation':>{hdr_w}}" + ''.join(f"{'JobA%':^{col_w}}{'JobB%':^{col_w}}{'FPay%M':^{col_w}}" for _ in CHECKPOINTS)
print(header1)
print(header2)
print(sep)
for name, _ in key_configs:
    row = f"{name:>{hdr_w}}"
    for t in CHECKPOINTS:
        s = averaged[name][t]
        row += f"{s['job_a']:^{col_w}.1f}{s['job_b']:^{col_w}.1f}{s['pay']:^{col_w}.1f}"
    print(row)
print('=' * len(sep))

# ─── Re-run summary at 50% job influence ──────────────────────────────────────
print('\n' + '=' * len(sep))
print(f'50% JOB INFLUENCE RESULTS  (mean over {N_RUNS} runs, noise={NOISE_STD})')
print('=' * len(sep))
print(header1)
print(header2)
print(sep)

averaged_50 = {}
for name, kwargs in key_configs:
    runs = [run_simulation(f'{name} run {s}', run_seed=1000+s, job_weight=0.5, **kwargs)
            for s in range(N_RUNS)]
    averaged_50[name] = {}
    for t in CHECKPOINTS:
        snaps = [snap(r, t) for r in runs]
        averaged_50[name][t] = {
            'job_a': np.mean([s['job_a'] for s in snaps]),
            'job_b': np.mean([s['job_b'] for s in snaps]),
            'pay':   np.mean([s['pay']   for s in snaps]),
        }

for name, _ in key_configs:
    row = f"{name:>{hdr_w}}"
    for t in CHECKPOINTS:
        s = averaged_50[name][t]
        row += f"{s['job_a']:^{col_w}.1f}{s['job_b']:^{col_w}.1f}{s['pay']:^{col_w}.1f}"
    print(row)
print('=' * len(sep))
