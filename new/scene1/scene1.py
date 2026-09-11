# -*- coding: utf-8 -*-
"""
Scene 1 — Meritopolis as a patriarchy
=====================================
The city of Meritopolis has 50% men and 50% women.

Like all cities, it has a city council. Each year sitting members name
friends they already know, and then that same council chooses among those
names. Friends in office matter twice: once to get on the slate, once to
win.

But this city has two problems:

1. It is incredibly sexist. Men and women only form friendships with people
   of the same gender.
2. The existing city council already has slightly more men than women:
   60% of council members are men.

What happens? Because men only befriend men, a male-heavy council both
nominates more men and then prefers those men. The 60% majority does not
wash out — it compounds.

This file simulates that election. You cannot stand unless a sitting
member names you. Among nominees, chance of winning is proportional to
how many friends you have on the current council.

Run:
  python scene1.py
  python scene1.py --size small
  python scene1.py --size large --no-animate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
N_ROUNDS = 30
N_RUNS = 80
N_PATHS = 40
EQUIL_FROM = 8
INIT_MALE_FRAC = 0.60


def layout_city(n: int, male: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Place men on the left, women on the right, jittered."""
    xy = np.zeros((n, 2))
    m = np.flatnonzero(male)
    f = np.flatnonzero(~male)
    xy[m, 0] = rng.normal(-1.2, 0.28, size=len(m))
    xy[m, 1] = rng.uniform(-1.4, 1.4, size=len(m))
    xy[f, 0] = rng.normal(1.2, 0.28, size=len(f))
    xy[f, 1] = rng.uniform(-1.4, 1.4, size=len(f))
    return xy


def seed_council(male: np.ndarray, seats: int, rng: np.random.Generator) -> np.ndarray:
    n_male_seats = int(round(INIT_MALE_FRAC * seats))
    men = np.flatnonzero(male)
    women = np.flatnonzero(~male)
    return np.concatenate([
        rng.choice(men, size=n_male_seats, replace=False),
        rng.choice(women, size=seats - n_male_seats, replace=False),
    ])


def likelihood_ratio(council: np.ndarray, male: np.ndarray) -> float:
    n_m, n_w = int(male.sum()), int((~male).sum())
    c_m = int(male[council].sum())
    c_w = len(council) - c_m
    p_m, p_w = c_m / n_m, c_w / n_w
    if p_w == 0:
        return np.inf
    return float(p_m / p_w)


def simulate_once(n, seats, k, rng) -> tuple:
    male = np.arange(n) < (n // 2)
    friends = C.random_same_group_friends(male.astype(int), k, rng)
    C.check_invariants(n, seats, friends, np.arange(seats))
    council = seed_council(male, seats, rng)
    history = [council.copy()]
    for _ in range(N_ROUNDS):
        council = C.elect(council, friends, rng)
        history.append(council.copy())
    return male, friends, history


def male_share(history, male) -> np.ndarray:
    return np.array([male[c].mean() * 100 for c in history])


def run_ensemble(n, seats, k, seed) -> dict:
    paths, finals, all_male, all_female = [], [], 0, 0
    last_hist = None
    male = None
    friends = None
    xy = None
    for run in range(N_RUNS):
        rng = np.random.default_rng(seed + run)
        male, friends, history = simulate_once(n, seats, k, rng)
        s = male_share(history, male)
        paths.append(s)
        finals.append(s[-1])
        if s[-1] >= 99:
            all_male += 1
        if s[-1] <= 1:
            all_female += 1
        last_hist = history
    paths = np.array(paths)
    rng_xy = np.random.default_rng(seed)
    xy = layout_city(n, male, rng_xy)
    # representative: closest to median final
    med = np.median(paths[:, -1])
    rep = int(np.argmin(np.abs(paths[:, -1] - med)))
    rng_rep = np.random.default_rng(seed + rep)
    male, friends, hist_rep = simulate_once(n, seats, k, rng_rep)

    equil = slice(EQUIL_FROM, None)
    mean_share = float(paths[:, equil].mean())
    n_m, n_w = int(male.sum()), int((~male).sum())
    seats_m = mean_share / 100.0 * seats
    p_m = seats_m / n_m
    p_w = (seats - seats_m) / n_w
    ratio = p_m / p_w if p_w > 0 else np.inf
    start_seats_m = int(round(INIT_MALE_FRAC * seats))
    start_ratio = (start_seats_m / n_m) / ((seats - start_seats_m) / n_w)

    return {
        "male": male, "friends": friends, "xy": xy, "paths": paths,
        "hist_rep": hist_rep, "mean_share": mean_share, "ratio": ratio,
        "start_ratio": start_ratio, "p_m": p_m, "p_w": p_w,
        "all_male": all_male, "all_female": all_female, "n": n, "seats": seats,
        "n_m": n_m, "n_w": n_w,
    }


def plot_stats(st: dict, out: Path) -> None:
    fig = plt.figure(figsize=(13.5, 8.6), facecolor="white")
    fig.suptitle(
        f"Scene 1 — Same-gender friendships, 60% male council  ·  N={st['n']}, {st['seats']} seats\n"
        "named by sitting friends, then chosen by sitting friends",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.28,
                  left=0.07, right=0.97, top=0.86, bottom=0.08)

    def scatter(ax, council, title):
        male, xy = st["male"], st["xy"]
        ax.scatter(xy[~male, 0], xy[~male, 1], s=18 if st["n"] <= 100 else 8,
                   c=C.C_FEMALE, alpha=0.55, linewidths=0, zorder=1)
        ax.scatter(xy[male, 0], xy[male, 1], s=18 if st["n"] <= 100 else 8,
                   c=C.C_MALE, alpha=0.55, linewidths=0, zorder=1)
        on = np.zeros(st["n"], dtype=bool)
        on[council] = True
        s = 48 if st["n"] <= 100 else 22
        ax.scatter(xy[on & ~male, 0], xy[on & ~male, 1], s=s, facecolors=C.C_FEMALE,
                   edgecolors="black", linewidths=0.7, zorder=3)
        ax.scatter(xy[on & male, 0], xy[on & male, 1], s=s, facecolors=C.C_MALE,
                   edgecolors="black", linewidths=0.7, zorder=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title)
        n_m = int(male[council].sum())
        ax.text(0.02, 0.98, f"{n_m} men  ·  {len(council)-n_m} women",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8"))

    scatter(fig.add_subplot(gs[0, 0]), st["hist_rep"][0], "Council at start (60% men)")
    scatter(fig.add_subplot(gs[0, 1]), st["hist_rep"][-1], f"Council after {N_ROUNDS} elections")

    ax = fig.add_subplot(gs[1, 0])
    t = np.arange(N_ROUNDS + 1)
    for p in st["paths"][:N_PATHS]:
        ax.plot(t, p, color=C.C_MALE, lw=1.0, alpha=0.18)
    ax.plot(t, st["paths"].mean(0), color="black", lw=1.5, ls="--", label="mean of runs")
    ax.axhline(60, color="0.45", ls=":", label="start (60%)")
    ax.axhline(50, color=C.C_FEMALE, ls=":", label="parity (50%)")
    ax.set_xlabel("Election round")
    ax.set_ylabel("% of council who are men")
    ax.set_title("Male share of the council")
    ax.set_ylim(35, 105)
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = fig.add_subplot(gs[1, 1])
    ax.set_axis_off()
    ax.set_title("Observations", pad=8)
    ratio = st["ratio"]
    ratio_s = "∞" if not np.isfinite(ratio) else f"{ratio:.1f}×"
    start_s = f"{st['start_ratio']:.1f}×"
    lines = [
        ("A man is this much more likely to sit on the council",
         f"{ratio_s}  as likely as a woman",
         f"seat chance {st['p_m']*100:.1f}% vs {st['p_w']*100:.1f}%  ·  "
         f"was {start_s} at the start"),
        ("Settled male share of the council",
         f"{st['mean_share']:.0f}%   (from 60%)",
         f"mean of rounds {EQUIL_FROM}–{N_ROUNDS}  ·  "
         f"{st['all_male']}/{N_RUNS} runs ≥99% men, "
         f"{st['all_female']}/{N_RUNS} runs ≥99% women"),
        ("Why it does not wash out",
         "named, then chosen, among same-gender friends",
         "a male-heavy council names more men and then prefers those men. "
         "The 60% majority compounds instead of washing out."),
    ]
    y = 0.92
    for title, head, detail in lines:
        ax.text(0.0, y, title.upper(), fontsize=7.5, color="0.35", fontweight="bold", va="top")
        y -= 0.08
        ax.text(0.0, y, head, fontsize=13, color="0.1", fontweight="bold", va="top")
        y -= 0.08
        ax.text(0.0, y, detail, fontsize=8.5, color="0.25", va="top")
        y -= 0.14
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C.C_MALE, markersize=8, label="men"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C.C_FEMALE, markersize=8, label="women"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.7",
               markeredgecolor="black", markersize=9, label="on the council"),
    ], loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(0.97, 0.91), fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  plot → {out}")


def animate(st: dict, out: Path) -> None:
    male, xy, friends, hist = st["male"], st["xy"], st["friends"], st["hist_rep"]
    n = st["n"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), facecolor=C.DARK_BG,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax, axp = axes
    C.style_dark_figure(fig, axes)
    s = 28 if n <= 100 else 10
    ax.scatter(xy[~male, 0], xy[~male, 1], s=s, c=C.C_PINK_BRIGHT,
               alpha=0.82, linewidths=0, zorder=2)
    ax.scatter(xy[male, 0], xy[male, 1], s=s, c=C.C_BLUE_BRIGHT,
               alpha=0.82, linewidths=0, zorder=2)
    on0 = np.zeros(n, dtype=bool); on0[hist[0]] = True
    sc_c = ax.scatter(xy[on0, 0], xy[on0, 1], s=s + 18, facecolors="none",
                      edgecolors=C.C_COUNCIL, linewidths=1.35, zorder=4)
    # a light sample of friendship edges
    rng = np.random.default_rng(0)
    sample_i = rng.choice(n, size=min(n, 80), replace=False)
    segs = []
    for i in sample_i:
        for j in friends[i, : min(6, friends.shape[1])]:
            segs.append([(xy[i, 0], xy[i, 1]), (xy[j, 0], xy[j, 1])])
    from matplotlib.collections import LineCollection
    lc = LineCollection(segs, colors=C.DARK_GRID, linewidths=0.4,
                        alpha=0.55, zorder=1)
    ax.add_collection(lc)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(xy[:, 0].min() - 0.3, xy[:, 0].max() + 0.3)
    ax.set_ylim(xy[:, 1].min() - 0.3, xy[:, 1].max() + 0.3)
    title = ax.set_title("")
    shares = male_share(hist, male)
    line, = axp.plot([], [], color=C.C_BLUE_BRIGHT, lw=2.6)
    axp.axhline(50, color=C.C_PINK_BRIGHT, ls=":", lw=1.2, alpha=0.8)
    axp.axhline(60, color=C.C_COUNCIL, ls="--", lw=1.0, alpha=0.7)
    axp.set_xlim(0, N_ROUNDS)
    axp.set_ylim(35, 105)
    axp.set_xlabel("Election round")
    axp.set_ylabel("% men on council")
    fig.suptitle("Scene 1  ·  a small majority compounds into total control",
                 color=C.DARK_TEXT, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(t):
        on = np.zeros(n, dtype=bool); on[hist[t]] = True
        sc_c.set_offsets(xy[on])
        nm = int(male[hist[t]].sum())
        title.set_text(f"Round {t}  ·  {nm} men, {len(hist[t])-nm} women")
        line.set_data(np.arange(t + 1), shares[: t + 1])
        return sc_c, title, line

    anim = FuncAnimation(fig, update, frames=len(hist), interval=350, blit=False)
    C.save_mp4(anim, out, fps=4)
    plt.close()


def main():
    args = C.parse_cli("Scene 1 — Meritopolis patriarchy")
    print("Scene 1 — same-gender friendships, 60% male council")
    print("  election: named by sitting friends, then chosen by them")
    for size in C.sizes_to_run(args.size):
        cfg = C.SIZES[size]
        n, seats, k = cfg["n"], cfg["council"], cfg["k_friends"]
        print(f"\n[{size}] N={n}  seats={seats}  friends={k}  runs={N_RUNS}")
        st = run_ensemble(n, seats, k, args.seed)
        print(f"  start ratio P(seat|man)/P(seat|woman) = {st['start_ratio']:.2f}")
        print(f"  settled male share {st['mean_share']:.1f}%")
        print(f"  settled P(seat|man)={st['p_m']*100:.2f}%  P(seat|woman)={st['p_w']*100:.2f}%  "
              f"ratio {st['ratio']:.2f}")
        print(f"  lock-in: {st['all_male']}/{N_RUNS} all-male, {st['all_female']}/{N_RUNS} all-female")
        plot_stats(st, HERE / f"scene1_n{n}.png")
        if args.animate:
            animate(st, HERE / f"scene1_n{n}.mp4")


if __name__ == "__main__":
    main()
