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
2. The existing city council already has more men than women:
   80% of council members are men.

What happens? Because men only befriend men, a male-heavy council both
nominates more men and then prefers those men. The majority does not
wash out — it compounds.

This file simulates that election. You cannot stand unless a sitting
member names you. Among nominees, chance of winning grows with how many
friends you have on the current council.

Scene-1-only tweaks (for a clearer, slower-locking visual — no other scene
is affected): the council starts 80% male, and the win advantage is damped
from the shared square rule (~k^2) to a gentler ~k^1.5 via ADVANTAGE_EXPONENT
and the local elect_soft. The animation is small-city only (N=100).

Run:
  python scene1.py
  python scene1.py --size small
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
N_ROUNDS = 10
N_RUNS = 80
N_PATHS = 40
EQUIL_FROM = 4
INIT_MALE_FRAC = 0.80

# Scene-1-only softening of the election advantage.
# common.elect uses named × chosen, so a group that is k times larger on the
# council wins roughly k^2 as often per person — the "square rule". For this
# scene we damp that to about k^1.5 (a gentler 3/2 power) so the lock-in is
# slower and easier to watch. Raising a k^2 weight to this exponent yields
# k^(2 * 0.75) = k^1.5. This does NOT touch common.elect or any other scene.
ADVANTAGE_EXPONENT = 0.75


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


def elect_soft(council: np.ndarray, friends: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """Named × chosen election, damped to a ~k^1.5 per-person advantage.

    Identical to common.elect except the winning weight is raised to
    ADVANTAGE_EXPONENT. Scene 1 uses this softer rule for readability; every
    other scene keeps the unmodified common.elect.
    """
    n = friends.shape[0]
    seats = len(council)
    tickets = C.nominations(council, friends, n)
    power = C.friends_in_power(council, friends, n)
    eligible = np.flatnonzero(tickets > 0)
    if len(eligible) < seats:
        chosen = eligible.copy()
        rest = np.setdiff1d(np.arange(n), chosen, assume_unique=False)
        extra = rng.choice(rest, size=seats - len(chosen), replace=False)
        return np.concatenate([chosen, extra])
    weight = (tickets[eligible] * np.maximum(power[eligible], 1.0)) ** ADVANTAGE_EXPONENT
    return C.weighted_sample(eligible, weight, seats, rng)


def simulate_once(n, seats, k, rng) -> tuple:
    male = np.arange(n) < (n // 2)
    friends = C.random_same_group_friends(male.astype(int), k, rng)
    C.check_invariants(n, seats, friends, np.arange(seats))
    council = seed_council(male, seats, rng)
    history = [council.copy()]
    for _ in range(N_ROUNDS):
        council = elect_soft(council, friends, rng)
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
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C.C_GREY))

    scatter(fig.add_subplot(gs[0, 0]), st["hist_rep"][0], "Council at start (60% men)")
    scatter(fig.add_subplot(gs[0, 1]), st["hist_rep"][-1], f"Council after {N_ROUNDS} elections")

    ax = fig.add_subplot(gs[1, 0])
    t = np.arange(N_ROUNDS + 1)
    for p in st["paths"][:N_PATHS]:
        ax.plot(t, p, color=C.C_MALE, lw=1.0, alpha=0.18)
    ax.plot(t, st["paths"].mean(0), color=C.C_GREY, lw=1.5, ls="--", label="mean of runs")
    ax.axhline(60, color=C.C_GREY, ls=":", label="start (60%)")
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
        ax.text(0.0, y, title.upper(), fontsize=7.5, color=C.C_GREY, fontweight="bold", va="top")
        y -= 0.08
        ax.text(0.0, y, head, fontsize=13, color=C.C_GREY, fontweight="bold", va="top")
        y -= 0.08
        ax.text(0.0, y, detail, fontsize=8.5, color=C.C_GREY, va="top")
        y -= 0.14
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C.C_MALE, markersize=8, label="men"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C.C_FEMALE, markersize=8, label="women"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C.C_GREY,
               markeredgecolor="black", markersize=9, label="on the council"),
    ], loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(0.97, 0.91), fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  plot → {out}")


def animate(st: dict, out: Path) -> None:
    male, xy, friends, hist = st["male"], st["xy"], st["friends"], st["hist_rep"]
    n = st["n"]
    black = "#000000"
    fig = plt.figure(figsize=(16, 9), facecolor=black)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=black)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # The world is split down the middle: men on the left, women on the right,
    # with only a thin seam between them. The top of each half is reserved for
    # that group's office-holders, who cluster in as many rows as they need.
    MID, SEAM = 0.5, 0.015
    CITY_TOP = 0.60

    def scale(values, lo, hi):
        span = max(np.ptp(values), 1e-9)
        return lo + (hi - lo) * (values - values.min()) / span

    base = np.zeros((n, 2))
    base[male, 0] = scale(xy[male, 0], 0.06, MID - SEAM)
    base[~male, 0] = scale(xy[~male, 0], MID + SEAM, 0.94)
    base[male, 1] = scale(xy[male, 1], 0.08, CITY_TOP)
    base[~male, 1] = scale(xy[~male, 1], 0.08, CITY_TOP)

    def grid_slots(count, x0, x1, per_row):
        """Pack ``count`` leader seats into rows filling from the top down."""
        if count == 0:
            return np.zeros((0, 2))
        y_top, y_gap = 0.86, 0.055
        out = []
        rows = int(np.ceil(count / per_row))
        for row in range(rows):
            in_row = min(per_row, count - row * per_row)
            xs = (np.linspace(x0, x1, in_row) if in_row > 1
                  else np.array([(x0 + x1) / 2]))
            y = y_top - row * y_gap
            out.extend((x, y) for x in xs)
        return np.asarray(out)

    def positions_for(council):
        pos = base.copy()
        c = np.asarray(council)
        leaders_m = np.sort(c[male[c]])
        leaders_w = np.sort(c[~male[c]])
        pos[leaders_m] = grid_slots(len(leaders_m), 0.06, MID - SEAM, 6)
        pos[leaders_w] = grid_slots(len(leaders_w), MID + SEAM, 0.94, 4)
        return pos

    def nomination_pairs(council):
        # All nominations are visible in the small city. The large-city view
        # samples three per leader so 40,000 lines do not become a solid block.
        shown = friends[council] if n <= 100 else friends[council, :3]
        src = np.repeat(council, shown.shape[1])
        return src, shown.ravel()

    def segments(pos, src, dst, progress):
        start = pos[src]
        end = start + progress * (pos[dst] - start)
        return np.stack((start, end), axis=1)

    ax.plot([MID, MID], [0.03, 0.90], color=C.C_GREY, lw=0.8,
            alpha=0.35, zorder=0)

    dot_size = 54 if n <= 100 else 11
    pos0 = positions_for(hist[0])
    sc_m = ax.scatter(
        pos0[male, 0], pos0[male, 1], s=dot_size, marker="o",
        c=C.C_BLUE, linewidths=0, alpha=0.95, zorder=3,
    )
    sc_w = ax.scatter(
        pos0[~male, 0], pos0[~male, 1], s=dot_size * 1.15, marker="^",
        c=C.C_PINK, linewidths=0, alpha=0.95, zorder=3,
    )
    nomination_lines = LineCollection(
        [], colors=C.C_GREY, linewidths=0.65 if n <= 100 else 0.35,
        alpha=0.34, zorder=1,
    )
    ax.add_collection(nomination_lines)

    glow_size = dot_size + (260 if n <= 100 else 55)
    ring_size = dot_size + (75 if n <= 100 else 22)
    current_pos = pos0[hist[0]]
    glow_old = ax.scatter(
        current_pos[:, 0], current_pos[:, 1], s=glow_size,
        c=C.C_COUNCIL, alpha=0.16, linewidths=0, zorder=2,
    )
    ring_old = ax.scatter(
        current_pos[:, 0], current_pos[:, 1], s=ring_size,
        facecolors="none", edgecolors=C.C_COUNCIL,
        linewidths=1.8 if n <= 100 else 0.8, zorder=4,
    )
    glow_new = ax.scatter(
        [], [], s=glow_size, c=C.C_COUNCIL, alpha=0.0,
        linewidths=0, zorder=2,
    )
    ring_new = ax.scatter(
        [], [], s=ring_size, facecolors="none", edgecolors=C.C_COUNCIL,
        linewidths=1.8 if n <= 100 else 0.8, alpha=0.0, zorder=4,
    )

    round_text = fig.text(
        0.05, 0.94, "Round 0", color=C.DARK_TEXT,
        fontsize=25, fontweight="bold", ha="left", va="center",
    )
    phase_text = fig.text(
        0.05, 0.895, "Current leaders", color=C.C_GREY,
        fontsize=14, ha="left", va="center",
    )
    count_text = fig.text(
        0.95, 0.94, "", color=C.DARK_TEXT,
        fontsize=16, fontweight="bold", ha="right", va="center",
    )
    fig.text(
        0.50, 0.965, "LEADERS", color=C.C_COUNCIL,
        fontsize=13, fontweight="bold", ha="center", va="center",
    )
    fig.text(
        0.05, 0.045, "●  MEN", color=C.C_BLUE,
        fontsize=15, fontweight="bold", ha="left",
    )
    fig.text(
        0.15, 0.045, "▲  WOMEN", color=C.C_PINK,
        fontsize=15, fontweight="bold", ha="left",
    )

    initial_hold = 18
    nomination_frames = 20
    nomination_hold = 8
    transition_frames = 24
    cycle = nomination_frames + nomination_hold + transition_frames

    def smoothstep(value):
        return value * value * (3.0 - 2.0 * value)

    def set_leader_artists(old_council, new_council, old_alpha, new_alpha, pos):
        old_xy = pos[old_council]
        new_xy = pos[new_council]
        glow_old.set_offsets(old_xy)
        ring_old.set_offsets(old_xy)
        glow_new.set_offsets(new_xy)
        ring_new.set_offsets(new_xy)
        glow_old.set_alpha(0.16 * old_alpha)
        ring_old.set_alpha(old_alpha)
        glow_new.set_alpha(0.16 * new_alpha)
        ring_new.set_alpha(new_alpha)

    def update(frame):
        if frame < initial_hold:
            round_index = 0
            local = -1
        else:
            elapsed = frame - initial_hold
            round_index = min(elapsed // cycle, N_ROUNDS - 1)
            local = elapsed % cycle

        current = hist[round_index]
        nxt = hist[min(round_index + 1, N_ROUNDS)]
        pos_current = positions_for(current)
        pos_next = positions_for(nxt)
        src, dst = nomination_pairs(current)

        if local < 0:
            pos = pos_current
            nomination_lines.set_segments([])
            set_leader_artists(current, nxt, 1.0, 0.0, pos)
            phase = "Current leaders"
            shown = current
            displayed_round = 0
        elif local < nomination_frames:
            progress = smoothstep(local / max(nomination_frames - 1, 1))
            pos = pos_current
            nomination_lines.set_segments(segments(pos, src, dst, progress))
            set_leader_artists(current, nxt, 1.0, 0.0, pos)
            phase = "Leaders nominate people they know"
            shown = current
            displayed_round = round_index + 1
        elif local < nomination_frames + nomination_hold:
            pos = pos_current
            nomination_lines.set_segments(segments(pos, src, dst, 1.0))
            set_leader_artists(current, nxt, 1.0, 0.0, pos)
            phase = "Nominees are considered"
            shown = current
            displayed_round = round_index + 1
        else:
            raw = (
                local - nomination_frames - nomination_hold
            ) / max(transition_frames - 1, 1)
            progress = smoothstep(raw)
            pos = pos_current * (1.0 - progress) + pos_next * progress
            nomination_lines.set_segments(
                segments(pos, src, dst, 1.0 - progress)
            )
            set_leader_artists(current, nxt, 1.0 - progress, progress, pos)
            phase = "Winners become the next leaders"
            shown = nxt if progress >= 0.5 else current
            displayed_round = round_index + 1

        sc_m.set_offsets(pos[male])
        sc_w.set_offsets(pos[~male])
        men = int(male[shown].sum())
        round_text.set_text(f"Round {displayed_round}")
        phase_text.set_text(phase)
        count_text.set_text(f"{men} men  ·  {len(shown) - men} women")
        return (
            sc_m, sc_w, nomination_lines, glow_old, ring_old,
            glow_new, ring_new, round_text, phase_text, count_text,
        )

    total_frames = initial_hold + N_ROUNDS * cycle
    anim = FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / 24, blit=False,
    )
    C.save_mp4(anim, out, fps=24, bg=black)
    plt.close()


def main():
    args = C.parse_cli("Scene 1 — Meritopolis patriarchy")
    if args.size == "large":
        raise SystemExit("Scene 1 is configured for the small city only (N=100).")
    print("Scene 1 — same-gender friendships, 80% male council (softened k^1.5 rule)")
    print("  election: named by sitting friends, then chosen by them")
    for size in ("small",):
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
