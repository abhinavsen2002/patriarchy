# -*- coding: utf-8 -*-
"""
Scene 1 — A sexist city council
================================
City of 1000: 500 men, 500 women.
Council of 100, starting 55% male (55 men, 45 women).

Each year the whole council is replaced:
  1. Every sitting member nominates 4 candidates of their own gender.
  2. The sitting council then elects 100 new members from that nominee pool.

Two problems:
  1. Extreme sexism — men only nominate and vote for men; women only for women.
  2. The sitting council already has a male majority (55%).

A candidate's chance of being voted in is proportional to how many sitting
members share their gender. Combined with same-gender nominations (more
majority-gender tickets), each new seat is male with probability:

    P(male) = n_men² / (n_men² + n_women²)

Run:
  python scene1_council.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ─── Setup ────────────────────────────────────────────────────────────────────
CITY_MEN, CITY_WOMEN = 500, 500
COUNCIL_SIZE         = 100
INIT_MEN             = 55          # 55% male at the start
NOMINEES_PER_MEMBER  = 4
N_PATHS              = 40          # faint trajectories on the plot
N_STATS              = 2000        # runs for time-to-all-men stats
MAX_ROUNDS           = 500
SEED                 = 42

OUT_DIR = Path(__file__).resolve().parent
C_MALE, C_FEMALE, C_BG = "#4A90D9", "#E8705A", "#0D0D0D"


def p_male(n_men: int) -> float:
    """P(next seat is male): nominations × same-gender voting."""
    n_women = COUNCIL_SIZE - n_men
    if n_men == 0:
        return 0.0
    if n_women == 0:
        return 1.0
    # n_men * NOMINEES_PER_MEMBER male tickets, each weighted by n_men voters
    # (the 4 cancels)
    return n_men ** 2 / (n_men ** 2 + n_women ** 2)


def one_election(n_men: int, rng: np.random.Generator) -> int:
    """Replace the whole council. Returns the new male count."""
    n_male_tickets   = n_men * NOMINEES_PER_MEMBER
    n_female_tickets = (COUNCIL_SIZE - n_men) * NOMINEES_PER_MEMBER
    # City is large enough that we never run out of same-gender nominees.
    assert n_male_tickets <= CITY_MEN or n_men == 0
    assert n_female_tickets <= CITY_WOMEN or n_men == COUNCIL_SIZE
    return int(rng.binomial(COUNCIL_SIZE, p_male(n_men)))


def simulate_until_all_men(rng: np.random.Generator, max_rounds: int = MAX_ROUNDS):
    """Return (men_each_round including t=0, rounds_to_all_men or None)."""
    men = [INIT_MEN]
    n = INIT_MEN
    for t in range(1, max_rounds + 1):
        n = one_election(n, rng)
        men.append(n)
        if n == COUNCIL_SIZE:
            return men, t
        if n == 0:
            return men, None  # absorbed at all women
    return men, None


def main():
    rng = np.random.default_rng(SEED)

    paths, times, n_all_women, n_timeout = [], [], 0, 0
    for _ in range(N_STATS):
        path, t = simulate_until_all_men(rng)
        paths.append(path)
        if t is None:
            if path[-1] == 0:
                n_all_women += 1
            else:
                n_timeout += 1
        else:
            times.append(t)

    times = np.array(times, dtype=int)
    n_all_men = len(times)

    # A representative path: median time-to-all-men among successful runs
    median_t = int(np.median(times))
    highlight = min(
        (p for p in paths if p[-1] == COUNCIL_SIZE),
        key=lambda p: abs((len(p) - 1) - median_t),
    )

    print("Scene 1 — sexist city council")
    print(f"  City {CITY_MEN} men / {CITY_WOMEN} women, council of {COUNCIL_SIZE}")
    print(f"  Start: {INIT_MEN} men, {COUNCIL_SIZE - INIT_MEN} women")
    print(f"  Each member nominates {NOMINEES_PER_MEMBER} same-gender candidates")
    print(f"  P(seat is male) = n_men² / (n_men² + n_women²)")
    print(f"    round 0: P(male) = {p_male(INIT_MEN):.3f}")
    print()
    print(f"  {N_STATS} simulations (max {MAX_ROUNDS} rounds):")
    print(f"    reached all men:   {n_all_men:5d}  ({100 * n_all_men / N_STATS:.1f}%)")
    print(f"    reached all women: {n_all_women:5d}  ({100 * n_all_women / N_STATS:.1f}%)")
    print(f"    still mixed:       {n_timeout:5d}")
    if n_all_men:
        print(f"    time to all men:   median {np.median(times):.0f}  "
              f"mean {times.mean():.1f}  "
              f"p5 {np.percentile(times, 5):.0f}  "
              f"p95 {np.percentile(times, 95):.0f}  "
              f"max {times.max()}")

    # ─── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor="white")
    fig.suptitle(
        "Scene 1 — How a 55% male council becomes all-male\n"
        "same-gender nomination + voting  ·  P(male seat) ∝ (men on council)²",
        fontsize=13, fontweight="bold", y=1.02,
    )

    ax = axes[0]
    plot_paths = paths[:N_PATHS]
    for p in plot_paths:
        ax.plot(range(len(p)), p, color=C_MALE, lw=1.0, alpha=0.18)
    ax.plot(range(len(highlight)), highlight, color=C_MALE, lw=2.4, label="typical run")
    ax.axhline(COUNCIL_SIZE, color=C_MALE, ls="--", lw=1, alpha=0.7)
    ax.axhline(INIT_MEN, color="0.5", ls=":", lw=1, label=f"start ({INIT_MEN} men)")
    ax.axhline(COUNCIL_SIZE / 2, color=C_FEMALE, ls=":", lw=1, alpha=0.7, label="parity (50)")
    ax.set_xlabel("Election round")
    ax.set_ylabel("Men on the 100-seat council")
    ax.set_title("Council composition over time")
    ax.set_ylim(-2, 105)
    ax.set_xlim(0, max(len(p) for p in plot_paths) * 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    bins = np.arange(times.min() - 0.5, times.max() + 1.5, 1) if n_all_men else 20
    ax.hist(times, bins=bins, color=C_MALE, edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(times), color="black", lw=1.6, label=f"median = {median_t} rounds")
    ax.set_xlabel("Rounds until the council is all men")
    ax.set_ylabel("Simulations")
    ax.set_title(f"Time to all-male  ({n_all_men}/{N_STATS} runs)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="y")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    out = OUT_DIR / "scene1_council.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
