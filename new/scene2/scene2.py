# -*- coding: utf-8 -*-
"""
Scene 2 — Gender is gone; stereotypes are not
=============================================
You are the god of this world, and you do not like the patriarchy, so you
erase gender from Meritopolis. There are no more men or women. There are
only people.

The election rule does not change: sitting members name friends, then
choose among those names. With no gender left, people form friendships
from shared personality instead.

Each person has 10 personality traits, Gaussian and clipped to ±100.
Only two are easy to draw:

  colour  — liking blue (positive) vs liking pink (negative)
  animal  — liking dogs (positive) vs liking cats (negative)

The other eight have no leftover gender bias. Friendships are the k
nearest people in that full 10-D space (20 of 100, or 200 of 1,000).
Plots show colour vs animal. Reported homophily is former gender:
what share of an ex-man's friends are also former men.

Meritopolis used to have a gender stereotype. Colour preference was heavily
influenced by gender: most people who were previously male happened to like
blue. So most of the (still 80%-male-origin) council are blueys. Because
blueys befriend blueys, they keep an election advantage even though nobody
is a "man" anymore.

The animal axis is different if men and women used to like cats and dogs
equally — then dog/cat is roughly balanced on the council and does not
help anyone. But if mostly women liked cats and men liked dogs, blue and
dog line up. Blueys then sit even closer together, so a bluey is even more
likely to befriend another bluey.

It does not matter which group got which animal. Invert the stereotype
(men like cats, women like dogs) and the geometry is the same: any extra
axis that used to track gender still packs similar people together.

This file compares three matched worlds:
  - colour stereotyped, animal neutral
  - colour and animal stereotyped the same way
  - colour stereotyped, animal inverted

Run:
  python scene2.py
  python scene2.py --size small
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
N_ROUNDS = 25
N_RUNS = 60
N_PATHS = 30
EQUIL_FROM = 8
INIT_MALE_FRAC = 0.80   # carried over from Scene 1: the council was 80% ex-men
COLOUR_SHIFT = 50.0
ANIMAL_SHIFT = 50.0
N_TRAITS = 10           # dim 0 colour, dim 1 animal, 2..9 unbiased


def make_traits(was_male: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    n = len(was_male)
    xy = rng.normal(0.0, C.XY_SIGMA, size=(n, N_TRAITS))
    xy[was_male, 0] += COLOUR_SHIFT
    xy[~was_male, 0] -= COLOUR_SHIFT
    if mode == "aligned":
        xy[was_male, 1] += ANIMAL_SHIFT
        xy[~was_male, 1] -= ANIMAL_SHIFT
    elif mode == "inverted":
        xy[was_male, 1] -= ANIMAL_SHIFT
        xy[~was_male, 1] += ANIMAL_SHIFT
    elif mode != "neutral":
        raise ValueError(mode)
    return C.clip_traits(xy)


def seed_council(was_male: np.ndarray, seats: int, rng: np.random.Generator) -> np.ndarray:
    n_m = int(round(INIT_MALE_FRAC * seats))
    men = np.flatnonzero(was_male)
    women = np.flatnonzero(~was_male)
    return np.concatenate([
        rng.choice(men, size=n_m, replace=False),
        rng.choice(women, size=seats - n_m, replace=False),
    ])


def gender_stats(was_male: np.ndarray, friends: np.ndarray, council: np.ndarray) -> dict:
    """Former-gender homophily and council share. Not colour labels."""
    k = friends.shape[1]
    n_m = int(was_male.sum())
    n_w = int((~was_male).sum())
    c_m = int(was_male[council].sum())
    p_m = c_m / max(n_m, 1)
    p_w = (len(council) - c_m) / max(n_w, 1)
    friend_male = was_male[friends]
    return {
        "k": k,
        "n_m": n_m, "n_w": n_w, "c_m": c_m,
        "male_share": 100.0 * c_m / len(council),
        "p_m": p_m, "p_w": p_w,
        "ratio": (p_m / p_w) if p_w > 0 else np.inf,
        "homophily_m": float(friend_male[was_male].mean()),
        "homophily_w": float((~friend_male)[~was_male].mean()),
        "random": float(was_male.mean()),
    }


def simulate_mode(n, seats, k, mode, seed, n_runs=N_RUNS):
    paths = []
    snap = None
    hom_m, hom_w = [], []
    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)
        was_male = np.arange(n) < (n // 2)
        xy = make_traits(was_male, rng, mode)
        friends = C.knn_friends(xy, k)
        council = seed_council(was_male, seats, rng)
        C.check_invariants(n, seats, friends, council)
        hist = [council.copy()]
        for _ in range(N_ROUNDS):
            council = C.elect(council, friends, rng)
            hist.append(council.copy())
        g0 = gender_stats(was_male, friends, hist[0])
        paths.append([gender_stats(was_male, friends, c)["male_share"] for c in hist])
        hom_m.append(g0["homophily_m"])
        hom_w.append(g0["homophily_w"])
        if run == 0:
            snap = {
                "xy": xy, "friends": friends, "hist": hist, "was_male": was_male,
                "g0": g0, "g1": gender_stats(was_male, friends, hist[-1]),
            }
    paths = np.array(paths)
    mean_share = float(paths[:, EQUIL_FROM:].mean())
    n_m = n // 2
    n_w = n - n_m
    seats_m = mean_share / 100.0 * seats
    p_m, p_w = seats_m / n_m, (seats - seats_m) / n_w
    pooled = p_m / p_w if p_w > 0 else np.inf
    return {
        "paths": paths, "snap": snap, "mean_share": mean_share,
        "homophily_m": float(np.mean(hom_m)),
        "homophily_w": float(np.mean(hom_w)),
        "homophily_m_sd": float(np.std(hom_m)),
        "pooled_ratio": float(pooled),
        "p_m": p_m, "p_w": p_w,
        "k": k, "random": 0.5,
    }


def plot_stats(results: dict, n: int, seats: int, k: int, out: Path) -> None:
    fig = plt.figure(figsize=(14.2, 10.2), facecolor="white")
    fig.suptitle(
        f"Scene 2 — leftover stereotypes, former-gender homophily  ·  N={n}, {seats} seats, {k} friends\n"
        f"friends = {k} nearest in {N_TRAITS}-D  ·  named by sitting friends, then chosen by them",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.28,
                  left=0.07, right=0.98, top=0.88, bottom=0.06)
    modes = [("neutral", "Colour stereotype only\n(animal unrelated)"),
             ("aligned", "Colour + animal aligned\n(ex-men: blue & dog)"),
             ("inverted", "Animal inverted\n(ex-men: blue & cat)")]
    for col, (mode, title) in enumerate(modes):
        r = results[mode]
        xy = r["snap"]["xy"]
        was = r["snap"]["was_male"]
        council = r["snap"]["hist"][-1]
        ax = fig.add_subplot(gs[0, col])
        ax.axvline(0, color="0.6", lw=0.7, ls="--")
        ax.axhline(0, color="0.6", lw=0.7, ls="--")
        s = 10 if n <= 100 else 4
        ax.scatter(xy[~was, 0], xy[~was, 1], s=s, c=C.C_FEMALE, alpha=0.5, linewidths=0)
        ax.scatter(xy[was, 0], xy[was, 1], s=s, c=C.C_MALE, alpha=0.5, linewidths=0)
        on = np.zeros(n, dtype=bool); on[council] = True
        ax.scatter(xy[on, 0], xy[on, 1], s=28 if n <= 100 else 12,
                   facecolors="none", edgecolors="black", linewidths=0.7, zorder=3)
        ax.set_xlim(-105, 105); ax.set_ylim(-105, 105); ax.set_aspect("equal")
        ax.set_xlabel("colour  (pink ← → blue)")
        ax.set_ylabel("animal  (cat ← → dog)" if col == 0 else "")
        ax.set_title(title, fontsize=10)
        n_m = int(was[council].sum())
        ax.text(0.02, 0.98, f"{n_m} ex-men  ·  {seats - n_m} ex-women",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8"))

        ax = fig.add_subplot(gs[1, col])
        t = np.arange(N_ROUNDS + 1)
        for p in r["paths"][:N_PATHS]:
            ax.plot(t, p, color=C.C_MALE, lw=0.9, alpha=0.2)
        ax.plot(t, r["paths"].mean(0), color="black", lw=1.4, ls="--")
        ax.axhline(50, color=C.C_FEMALE, ls=":", lw=1)
        ax.axhline(100 * INIT_MALE_FRAC, color="0.5", ls=":", lw=1)
        ax.set_ylim(40, 105)
        ax.set_xlabel("Election round")
        ax.set_ylabel("% council ex-men" if col == 0 else "")
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = fig.add_subplot(gs[2, :])
    ax.set_axis_off()
    neu, ali, inv = results["neutral"], results["aligned"], results["inverted"]
    def pct(x): return 100 * x
    text = (
        f"Gender homophily = share of k-NN friends with the same former gender "
        f"(not colour). Random mixing would be 50%.\n\n"
        f"Colour stereotype only: an ex-man has {pct(neu['homophily_m']):.0f}% ex-men friends "
        f"(ex-woman {pct(neu['homophily_w']):.0f}% ex-women). "
        f"An ex-man is {neu['pooled_ratio']:.1f}× as likely to hold a seat "
        f"({neu['p_m']*100:.1f}% vs {neu['p_w']*100:.1f}%); council stays "
        f"{neu['mean_share']:.0f}% ex-male (start {100*INIT_MALE_FRAC:.0f}%).\n\n"
        f"Colour + animal aligned: homophily rises to {pct(ali['homophily_m']):.0f}% / "
        f"{pct(ali['homophily_w']):.0f}%, seat ratio {ali['pooled_ratio']:.1f}×, "
        f"council {ali['mean_share']:.0f}% ex-male.\n"
        f"Animal inverted: {pct(inv['homophily_m']):.0f}% / {pct(inv['homophily_w']):.0f}%, "
        f"ratio {inv['pooled_ratio']:.1f}×, council {inv['mean_share']:.0f}% ex-male "
        f"— same packing, opposite animal corner."
    )
    ax.text(0.0, 0.95, "Observations  (former-gender homophily)", fontsize=12,
            fontweight="bold", va="top")
    ax.text(0.0, 0.72, text, fontsize=9.2, va="top", wrap=True)
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C.C_MALE, markersize=8, label="former men"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C.C_FEMALE, markersize=8, label="former women"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="black", markersize=9, label="on the council"),
    ], loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(0.98, 0.935), fontsize=9)
    fig.savefig(out, dpi=140)
    plt.close()
    print(f"  plot → {out}")


def plot_homophily_compare(n: int, k: int, seed: int, out: Path) -> None:
    was = np.arange(n) < (n // 2)

    def pack(mode, s):
        r = np.random.default_rng(s)
        xy = make_traits(was, r, mode)
        fr = C.knn_friends(xy, k)
        hm = float(was[fr][was].mean())
        hw = float((~was)[fr][~was].mean())
        return xy, fr, hm, hw

    def hull_xy(pts2):
        c = pts2.mean(0)
        ang = np.arctan2(pts2[:, 1] - c[1], pts2[:, 0] - c[0])
        bins = np.linspace(-np.pi, np.pi, 17)
        verts = []
        for a0, a1 in zip(bins[:-1], bins[1:]):
            m = (ang >= a0) & (ang < a1)
            if not np.any(m):
                continue
            chunk = pts2[m]
            d = np.linalg.norm(chunk - c, axis=1)
            verts.append(chunk[np.argmax(d)])
        if len(verts) < 3:
            return None
        v = np.array(verts)
        return np.vstack([v, v[0]])

    def typical(xy, mask):
        idx = np.flatnonzero(mask)
        centre = np.array([np.median(xy[idx, 0]), np.median(xy[idx, 1])])
        return idx[np.argmin(np.linalg.norm(xy[idx, :2] - centre, axis=1))]

    def means(mode, n_seeds=16):
        ms, ws = [], []
        for s in range(n_seeds):
            _, _, hm, hw = pack(mode, s)
            ms.append(hm); ws.append(hw)
        return np.mean(ms), np.std(ms), np.mean(ws), np.std(ws)

    n_m, n_s, n_w, n_ws = means("neutral")
    a_m, a_s, a_w, a_ws = means("aligned")
    print(f"  gender homophily  no-animal  ex-men {100*n_m:.1f}±{100*n_s:.1f}%  "
          f"ex-women {100*n_w:.1f}±{100*n_ws:.1f}%")
    print(f"  gender homophily  +animal    ex-men {100*a_m:.1f}±{100*a_s:.1f}%  "
          f"ex-women {100*a_w:.1f}±{100*a_ws:.1f}%")

    left = pack("neutral", 3)
    right = pack("aligned", 3)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.5), facecolor="white")
    fig.suptitle(
        f"{k} friends out of {n}, chosen in {N_TRAITS}-D  ·  colour & animal shift {COLOUR_SHIFT:.0f}\n"
        "dots coloured by former gender  ·  homophily = same former gender among k-NN",
        fontsize=12.5, fontweight="bold", y=0.99,
    )
    titles = (
        "Without animal correlation\nformer men → blue, former women → pink",
        "With animal correlation (opposite directions)\nformer men → blue & dog, former women → pink & cat",
    )
    for ax, (xy, fr, hm, hw), title in zip(axes, (left, right), titles):
        ax.axvline(0, color="0.65", ls="--", lw=0.8)
        ax.axhline(0, color="0.65", ls="--", lw=0.8)
        sdot = 28 if n <= 100 else 8
        ax.scatter(xy[~was, 0], xy[~was, 1], s=sdot, c=C.C_FEMALE, alpha=0.7, linewidths=0, zorder=2)
        ax.scatter(xy[was, 0], xy[was, 1], s=sdot, c=C.C_MALE, alpha=0.7, linewidths=0, zorder=2)
        own = {}
        for mask, color, key in ((was, C.C_MALE, "m"), (~was, C.C_FEMALE, "w")):
            i = typical(xy, mask)
            friends = fr[i]
            own[key] = float(was[friends].mean() if key == "m" else (~was[friends]).mean())
            ax.scatter(xy[friends, 0], xy[friends, 1], s=sdot + 8, facecolors="none",
                       edgecolors=color, linewidths=0.85, zorder=3)
            ring = hull_xy(xy[friends, :2])
            if ring is not None:
                ax.fill(ring[:, 0], ring[:, 1], color=color, alpha=0.12, zorder=1)
                ax.plot(ring[:, 0], ring[:, 1], color=color, lw=1.3, alpha=0.9, zorder=3)
            ax.scatter([xy[i, 0]], [xy[i, 1]], s=130 if n <= 100 else 50, c=color,
                       edgecolors="black", linewidths=1.15, zorder=6)
        ax.set_xlim(-105, 105); ax.set_ylim(-105, 105); ax.set_aspect("equal")
        ax.set_xlabel("colour  (pink ← → blue)")
        ax.set_title(title, fontsize=10.5)
        ax.text(
            0.03, 0.97,
            f"Avg over all ex-men:  {100*hm:.0f}% of friends ex-men\n"
            f"Avg over all ex-women: {100*hw:.0f}% of friends ex-women\n"
            f"This typical ex-man: {100*own['m']:.0f}%   typical ex-woman: {100*own['w']:.0f}%",
            transform=ax.transAxes, va="top", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.96),
        )
    axes[0].set_ylabel("animal  (cat ← → dog)")
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C.C_MALE, markersize=8, label="former men"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C.C_FEMALE, markersize=8, label="former women"),
    ], loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(0.99, 0.86), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  homophily plot → {out}")


def animate(results: dict, n: int, out: Path) -> None:
    r = results["neutral"]
    xy, friends, hist = r["snap"]["xy"], r["snap"]["friends"], r["snap"]["hist"]
    was = r["snap"]["was_male"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.5), facecolor=C.DARK_BG)
    ax, axp = axes
    C.style_dark_figure(fig, axes)
    s = 22 if n <= 100 else 8
    ax.axvline(0, color=C.DARK_GRID, ls="--", lw=0.8)
    ax.axhline(0, color=C.DARK_GRID, ls="--", lw=0.8)
    ax.scatter(xy[~was, 0], xy[~was, 1], s=s, c=C.C_PINK_BRIGHT,
               alpha=0.72, linewidths=0, zorder=2)
    ax.scatter(xy[was, 0], xy[was, 1], s=s, c=C.C_BLUE_BRIGHT,
               alpha=0.72, linewidths=0, zorder=2)
    on = np.zeros(n, dtype=bool); on[hist[0]] = True
    sc_c = ax.scatter(xy[on, 0], xy[on, 1], s=s + 18, facecolors="none",
                      edgecolors=C.C_COUNCIL, linewidths=1.25, zorder=4)
    from matplotlib.collections import LineCollection
    rng = np.random.default_rng(1)
    sample = rng.choice(n, size=min(n, 70), replace=False)
    segs = [[xy[i, :2], xy[j, :2]] for i in sample for j in friends[i, :4]]
    ax.add_collection(LineCollection(
        segs, colors=C.DARK_GRID, linewidths=0.35, alpha=0.48, zorder=1
    ))
    ax.set_xlim(-105, 105); ax.set_ylim(-105, 105); ax.set_aspect("equal")
    ax.set_xlabel("colour (pink ← → blue)")
    ax.set_ylabel("animal (cat ← → dog)")
    title = ax.set_title("")
    shares = r["paths"][0]
    line, = axp.plot([], [], color=C.C_BLUE_BRIGHT, lw=2.6)
    axp.axhline(50, color=C.C_PINK_BRIGHT, ls=":", alpha=0.8)
    axp.axhline(100 * INIT_MALE_FRAC, color=C.C_COUNCIL, ls="--", alpha=0.7)
    axp.set_xlim(0, N_ROUNDS); axp.set_ylim(40, 105)
    axp.set_xlabel("Election round"); axp.set_ylabel("% council ex-men")
    fig.suptitle(f"Scene 2 — colour stereotype only  ·  N={n}  ·  former-gender view",
                 color=C.DARK_TEXT, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(t):
        on = np.zeros(n, dtype=bool); on[hist[t]] = True
        sc_c.set_offsets(xy[on, :2])
        nm = int(was[hist[t]].sum())
        title.set_text(f"Round {t}  ·  {nm} ex-men, {len(hist[t])-nm} ex-women")
        line.set_data(np.arange(t + 1), shares[: t + 1])
        return sc_c, title, line

    anim = FuncAnimation(fig, update, frames=len(hist), interval=350, blit=False)
    C.save_mp4(anim, out, fps=4)
    plt.close()


def main():
    args = C.parse_cli("Scene 2 — stereotypes, former-gender homophily")
    print("Scene 2 — 10-D friendships, former-gender homophily (traits frozen)")
    print(f"  colour/animal shift {COLOUR_SHIFT:.0f}, {N_TRAITS-2} unbiased dims, "
          f"start council {100*INIT_MALE_FRAC:.0f}% ex-men")
    print("  election: named by sitting friends, then chosen by them")
    for size in C.sizes_to_run(args.size):
        cfg = C.SIZES[size]
        n, seats, k = cfg["n"], cfg["council"], cfg["k_friends"]
        print(f"\n[{size}] N={n}  seats={seats}  friends={k}  runs={N_RUNS}")
        plot_homophily_compare(n, k, args.seed, HERE / f"scene2_homophily_n{n}.png")
        results = {}
        for i, mode in enumerate(("neutral", "aligned", "inverted")):
            r = simulate_mode(n, seats, k, mode, args.seed + 1000 * i)
            results[mode] = r
            print(f"  {mode}: homophily {100*r['homophily_m']:.0f}%/{100*r['homophily_w']:.0f}%  "
                  f"council {r['mean_share']:.1f}% ex-men  "
                  f"seat ratio {r['pooled_ratio']:.2f}×")
        plot_stats(results, n, seats, k, HERE / f"scene2_n{n}.png")
        if args.animate:
            animate(results, n, HERE / f"scene2_n{n}.mp4")


if __name__ == "__main__":
    main()
