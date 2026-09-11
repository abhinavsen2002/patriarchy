# -*- coding: utf-8 -*-
"""
Scene 2 — Friendships of interest
=================================
No gender. Each of 1000 people has two interest attributes (x, y), drawn
Gaussian and clipped to [-100, +100]. Friendships are the 20 nearest
neighbours on that plane.

Council of 100. Each year:
  1. Every sitting member nominates 4 of their friends.
  2. Nominees campaign: chance of winning ∝ how many of their friends
     currently sit on the council ("friends in power").

The only seed of inequality: 60% of the initial council is drawn from
people with y > 0.

Run:
  python scene2_interests.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ─── Setup ────────────────────────────────────────────────────────────────────
N                    = 1000
COUNCIL_SIZE         = 100
INIT_POS_Y           = 60          # 60 of 100 initial seats are y > 0
K_FRIENDS            = 20
NOMINEES_PER_MEMBER  = 4
XY_SIGMA             = 40.0        # N(0, σ), then clipped to ±100
XY_CLIP              = 100.0
N_ROUNDS             = 25
N_RUNS               = 80
N_PATHS              = 40
EQUIL_FROM           = 8           # rounds used for "settled" stats
SEED                 = 42

OUT_DIR = Path(__file__).resolve().parent
C_POS = "#4A90D9"   # y > 0
C_NEG = "#E8705A"   # y < 0


def build_city(rng: np.random.Generator):
    xy = rng.normal(0.0, XY_SIGMA, size=(N, 2))
    xy = np.clip(xy, -XY_CLIP, XY_CLIP)
    pos = xy[:, 1] > 0

    delta = xy[:, None, :] - xy[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", delta, delta)
    np.fill_diagonal(d2, np.inf)
    friends = np.argpartition(d2, K_FRIENDS, axis=1)[:, :K_FRIENDS]
    return xy, pos, friends


def homophily(pos, friends):
    """Friend-sign stats from the 20 nearest neighbours."""
    friend_pos = pos[friends]
    n_pos_friends = friend_pos.sum(axis=1)
    plus, minus = pos, ~pos
    return {
        "pos_friends_if_pos": float(n_pos_friends[plus].mean()),
        "pos_friends_if_neg": float(n_pos_friends[minus].mean()),
        "neg_friends_if_neg": float((K_FRIENDS - n_pos_friends[minus]).mean()),
        "same_sign_if_pos":   float(n_pos_friends[plus].mean()),
        "same_sign_if_neg":   float((K_FRIENDS - n_pos_friends[minus]).mean()),
        "n_pos":              int(plus.sum()),
        "n_neg":              int(minus.sum()),
        "random_pos_friends": float(K_FRIENDS * plus.mean()),
    }


def draw_initial_council(pos, rng: np.random.Generator) -> np.ndarray:
    plus = np.flatnonzero(pos)
    minus = np.flatnonzero(~pos)
    n_plus = min(INIT_POS_Y, len(plus))
    n_minus = COUNCIL_SIZE - n_plus
    chosen_plus = rng.choice(plus, size=n_plus, replace=False)
    chosen_minus = rng.choice(minus, size=n_minus, replace=False)
    return np.concatenate([chosen_plus, chosen_minus])


def friends_in_power(council, friends) -> np.ndarray:
    on = np.zeros(N, dtype=bool)
    on[council] = True
    return on[friends].sum(axis=1)


def weighted_sample(items, weights, k, rng: np.random.Generator) -> np.ndarray:
    """Efraimidis–Spirakis weighted sample without replacement."""
    items = np.asarray(items)
    w = np.clip(np.asarray(weights, dtype=float), 1e-12, None)
    k = min(k, len(items))
    keys = rng.random(len(w)) ** (1.0 / w)
    pick = np.argpartition(-keys, kth=k - 1)[:k]
    return items[pick]


def elect(council, friends, rng: np.random.Generator) -> np.ndarray:
    noms = np.empty(COUNCIL_SIZE * NOMINEES_PER_MEMBER, dtype=int)
    for i, c in enumerate(council):
        noms[i * NOMINEES_PER_MEMBER:(i + 1) * NOMINEES_PER_MEMBER] = rng.choice(
            friends[c], size=NOMINEES_PER_MEMBER, replace=False
        )
    unique = np.unique(noms)
    power = friends_in_power(council, friends)
    weights = np.maximum(power[unique], 1.0)
    chosen = weighted_sample(unique, weights, COUNCIL_SIZE, rng)
    if len(chosen) < COUNCIL_SIZE:
        rest = np.setdiff1d(np.arange(N), chosen, assume_unique=False)
        extra = weighted_sample(rest, np.maximum(power[rest], 1e-6),
                                COUNCIL_SIZE - len(chosen), rng)
        chosen = np.concatenate([chosen, extra])
    return chosen


def likelihood_ratio(council, pos) -> float:
    """P(on council | y>0) / P(on council | y<0)."""
    n_pos, n_neg = pos.sum(), (~pos).sum()
    c_pos = pos[council].sum()
    c_neg = COUNCIL_SIZE - c_pos
    p_pos = c_pos / n_pos
    p_neg = c_neg / n_neg
    if p_neg == 0:
        return np.inf
    return float(p_pos / p_neg)


def simulate(xy, pos, friends, rng: np.random.Generator, n_rounds: int):
    council = draw_initial_council(pos, rng)
    history = [council.copy()]
    ratios = [likelihood_ratio(council, pos)]
    power0 = friends_in_power(council, friends)
    for _ in range(n_rounds):
        council = elect(council, friends, rng)
        history.append(council.copy())
        ratios.append(likelihood_ratio(council, pos))
    return history, np.array(ratios), power0


def pos_share(history, pos):
    return np.array([pos[c].mean() * 100 for c in history])


def fmt_ratio(x: float) -> str:
    if not np.isfinite(x):
        return "∞"
    return f"{x:.1f}×"


def main():
    rng_city = np.random.default_rng(SEED)
    xy, pos, friends = build_city(rng_city)
    h = homophily(pos, friends)

    paths, ratio_paths, last_histories = [], [], []
    power_plus, power_minus = [], []
    nom_plus_share = []

    for run in range(N_RUNS):
        rng = np.random.default_rng(SEED + 1 + run)
        history, ratios, power0 = simulate(xy, pos, friends, rng, N_ROUNDS)
        paths.append(pos_share(history, pos))
        ratio_paths.append(ratios)
        last_histories.append(history)
        plus, minus = pos, ~pos
        power_plus.append(power0[plus].mean())
        power_minus.append(power0[minus].mean())

        # share of nominations that are +y, averaged over rounds
        rng_n = np.random.default_rng(SEED + 1 + run)
        council = history[0]
        shares = []
        for t in range(N_ROUNDS):
            picks = []
            for c in council:
                picks.extend(rng_n.choice(friends[c], size=NOMINEES_PER_MEMBER,
                                          replace=False).tolist())
            shares.append(pos[np.array(picks)].mean())
            council = history[t + 1]
        nom_plus_share.append(np.mean(shares))

    paths = np.array(paths)
    ratio_paths = np.array(ratio_paths)
    equil = slice(EQUIL_FROM, None)
    equil_share = float(paths[:, equil].mean())
    n_pos, n_neg = h["n_pos"], h["n_neg"]
    # Ratio of average probabilities — not the average of per-run ratios,
    # which explodes when a run goes to ~100% +y.
    seats_pos = equil_share / 100.0 * COUNCIL_SIZE
    p_seat_pos = seats_pos / n_pos
    p_seat_neg = (COUNCIL_SIZE - seats_pos) / n_neg
    equil_ratio = p_seat_pos / p_seat_neg
    start_ratio = (INIT_POS_Y / n_pos) / ((COUNCIL_SIZE - INIT_POS_Y) / n_neg)
    n_all_pos = int((paths[:, -1] >= 99.0).sum())

    power_end_plus, power_end_minus = [], []
    for hist in last_histories:
        pwr = friends_in_power(hist[-1], friends)
        power_end_plus.append(pwr[pos].mean())
        power_end_minus.append(pwr[~pos].mean())
    mean_power_pos = float(np.mean(power_end_plus))
    mean_power_neg = float(np.mean(power_end_minus))

    # Representative run: closest to median final share
    med_final = np.median(paths[:, -1])
    rep = int(np.argmin(np.abs(paths[:, -1] - med_final)))
    hist_rep = last_histories[rep]
    council0, council_f = hist_rep[0], hist_rep[-1]

    # Friends-in-power at the start of the representative run
    rng_rep = np.random.default_rng(SEED + 1 + rep)
    # already have history; recompute power at t=0 and at equilibrium
    power_start = friends_in_power(council0, friends)
    power_end = friends_in_power(council_f, friends)

    print("Scene 2 — friendships of interest")
    print(f"  {N} people, {K_FRIENDS} nearest-neighbour friends, "
          f"x,y ~ N(0,{XY_SIGMA:.0f}) clipped to ±{XY_CLIP:.0f}")
    print(f"  population y>0: {h['n_pos']}   y<0: {h['n_neg']}")
    print(f"  +y person: {h['pos_friends_if_pos']:.2f} of {K_FRIENDS} friends are +y"
          f"  (random would be {h['random_pos_friends']:.2f})")
    print(f"  −y person: {h['pos_friends_if_neg']:.2f} of {K_FRIENDS} friends are +y")
    print(f"  start likelihood ratio P(seat|+y)/P(seat|−y): {start_ratio:.2f}")
    print(f"  settled (rounds {EQUIL_FROM}–{N_ROUNDS}) share +y on council: {equil_share:.1f}%")
    print(f"  settled P(seat | +y)={p_seat_pos*100:.2f}%  P(seat | −y)={p_seat_neg*100:.2f}%  "
          f"ratio {equil_ratio:.2f}")
    print(f"  settled friends-in-power: +y {mean_power_pos:.2f}   −y {mean_power_neg:.2f}")
    print(f"  mean nomination share +y: {np.mean(nom_plus_share)*100:.1f}%")
    print(f"  runs ending ≥99% +y: {n_all_pos}/{N_RUNS}")

    # ─── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.5, 9.6), facecolor="white")
    fig.suptitle(
        "Scene 2 — When friendships follow interests, a 60% +y council still concentrates\n"
        "nominate 4 friends  ·  P(win) ∝ friends currently in power",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28,
                  left=0.07, right=0.97, top=0.86, bottom=0.07)

    def scatter_council(ax, council, title):
        ax.scatter(xy[~pos, 0], xy[~pos, 1], s=12, c=C_NEG, alpha=0.45,
                   linewidths=0, rasterized=True)
        ax.scatter(xy[pos, 0], xy[pos, 1], s=12, c=C_POS, alpha=0.45,
                   linewidths=0, rasterized=True)
        on = np.zeros(N, dtype=bool)
        on[council] = True
        ax.scatter(xy[on & ~pos, 0], xy[on & ~pos, 1], s=42, facecolors=C_NEG,
                   edgecolors="black", linewidths=0.7, zorder=3)
        ax.scatter(xy[on & pos, 0], xy[on & pos, 1], s=42, facecolors=C_POS,
                   edgecolors="black", linewidths=0.7, zorder=3)
        ax.axhline(0, color="0.55", lw=0.8, ls="--")
        ax.set_xlim(-XY_CLIP - 5, XY_CLIP + 5)
        ax.set_ylim(-XY_CLIP - 5, XY_CLIP + 5)
        ax.set_aspect("equal")
        ax.set_xlabel("interest x")
        ax.set_ylabel("interest y")
        ax.set_title(title)
        n_plus = int(pos[council].sum())
        ax.text(0.02, 0.98, f"{n_plus} +y   ·   {COUNCIL_SIZE - n_plus} −y",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9))

    scatter_council(fig.add_subplot(gs[0, 0]), council0,
                    "Interest plane — council at start (60% +y)")
    scatter_council(fig.add_subplot(gs[0, 1]), council_f,
                    f"Same city — council after {N_ROUNDS} elections")

    ax = fig.add_subplot(gs[1, 0])
    t = np.arange(N_ROUNDS + 1)
    for p in paths[:N_PATHS]:
        ax.plot(t, p, color=C_POS, lw=1.0, alpha=0.18)
    ax.plot(t, paths[rep], color=C_POS, lw=2.4, label="typical run")
    ax.plot(t, paths.mean(axis=0), color="black", lw=1.4, ls="--", label="mean of runs")
    ax.axhline(60, color="0.45", ls=":", lw=1, label="start (60%)")
    ax.axhline(50, color=C_NEG, ls=":", lw=1, alpha=0.8, label="population (~50%)")
    ax.set_xlabel("Election round")
    ax.set_ylabel("% of council with y > 0")
    ax.set_title("+y share of the 100-seat council")
    ax.set_ylim(40, 105)
    ax.set_xlim(0, N_ROUNDS)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Observations", pad=8)

    # After round 0, a +y person is start_ratio times as likely...
    # Friends: out of 20
    obs = [
        ("A +y person is this much more likely to sit on the council",
         f"{fmt_ratio(equil_ratio)}  as likely as a −y person",
         f"seat chance {p_seat_pos*100:.1f}% vs {p_seat_neg*100:.1f}%  ·  "
         f"was {fmt_ratio(start_ratio)} at the start (60 / 40 seats)"),
        ("+y friends of a +y person",
         f"{h['pos_friends_if_pos']:.1f} of {K_FRIENDS}",
         f"random mixing would give {h['random_pos_friends']:.1f}  ·  "
         f"the 20 nearest neighbours almost always share your sign of y"),
        ("+y friends of a −y person",
         f"{h['pos_friends_if_neg']:.1f} of {K_FRIENDS}",
         "only people sitting near y = 0 have a mixed circle"),
        ("Friends currently in power (settled)",
         f"{mean_power_pos:.1f} for +y   vs   {mean_power_neg:.1f} for −y",
         "campaign weight is how many of your 20 friends already sit on the council"),
        ("Council +y share",
         f"{equil_share:.0f}% settled   (from 60%)",
         f"mean of rounds {EQUIL_FROM}–{N_ROUNDS}  ·  "
         f"{n_all_pos}/{N_RUNS} runs reach ≥99% +y"),
    ]

    y = 0.96
    for title, headline, detail in obs:
        ax.text(0.0, y, title.upper(), fontsize=7.5, color="0.35",
                va="top", fontweight="bold", transform=ax.transAxes)
        y -= 0.055
        ax.text(0.0, y, headline, fontsize=12.5, color="0.1",
                va="top", fontweight="bold", transform=ax.transAxes)
        y -= 0.055
        ax.text(0.0, y, detail, fontsize=8.2, color="0.25",
                va="top", transform=ax.transAxes, wrap=True)
        y -= 0.085

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_POS,
               markersize=8, label="y > 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NEG,
               markersize=8, label="y < 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.7",
               markeredgecolor="black", markersize=9, label="on the council"),
    ]
    fig.legend(handles=legend, loc="upper right", frameon=False,
               ncol=3, bbox_to_anchor=(0.97, 0.91), fontsize=9)

    out = OUT_DIR / "scene2_interests.png"
    fig.savefig(out, dpi=160)
    plt.close()
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
