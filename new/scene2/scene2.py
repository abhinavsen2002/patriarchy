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
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
N_ROUNDS = 25
N_RUNS = 60
N_PATHS = 30
EQUIL_FROM = 8
INIT_MALE_FRAC = 0.60   # same inherited start as Scene 1
COLOUR_SHIFT = 50.0
ANIMAL_SHIFT = 50.0
N_TRAITS = 10           # dim 0 colour, dim 1 animal, 2..9 unbiased
FINAL_ANIMATION_SEED = 110


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
        ax.axvline(0, color=C.C_GREY, lw=0.7, ls="--")
        ax.axhline(0, color=C.C_GREY, lw=0.7, ls="--")
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
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C.C_GREY))

        ax = fig.add_subplot(gs[1, col])
        t = np.arange(N_ROUNDS + 1)
        for p in r["paths"][:N_PATHS]:
            ax.plot(t, p, color=C.C_MALE, lw=0.9, alpha=0.2)
        ax.plot(t, r["paths"].mean(0), color=C.C_GREY, lw=1.4, ls="--")
        ax.axhline(50, color=C.C_FEMALE, ls=":", lw=1)
        ax.axhline(100 * INIT_MALE_FRAC, color=C.C_GREY, ls=":", lw=1)
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
        ax.axvline(0, color=C.C_GREY, ls="--", lw=0.8)
        ax.axhline(0, color=C.C_GREY, ls="--", lw=0.8)
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
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C.C_GREY, alpha=0.96),
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


def animate(results: dict, n: int, out: Path, mode: str = "neutral") -> None:
    r = results[mode]
    xy, friends, hist = r["snap"]["xy"], r["snap"]["friends"], r["snap"]["hist"]
    black = "#000000"
    fig = plt.figure(figsize=(16, 9), facecolor=black)
    ax = fig.add_axes([0.255, 0.14, 0.49, 0.72])
    C.style_dark_axis(ax, grid=False)
    fig.patch.set_facecolor(black)
    ax.set_facecolor(black)
    s = 36 if n <= 100 else 10
    blue = xy[:, 0] > 0
    ax.axvline(0, color=C.C_GREY, lw=0.8)
    ax.axhline(0, color=C.C_GREY, lw=0.8)

    rng = np.random.default_rng(1)
    src = np.repeat(np.arange(n), friends.shape[1])
    dst = friends.ravel()
    packed = np.minimum(src, dst).astype(np.int64) * n + np.maximum(src, dst)
    packed = np.unique(packed)
    src, dst = packed // n, packed % n
    keep = rng.random(len(src)) < 0.40
    src, dst = src[keep], dst[keep]
    segs = np.stack((xy[src, :2], xy[dst, :2]), axis=1)
    dist = np.linalg.norm(xy[src, :2] - xy[dst, :2], axis=1)
    scale = max(float(np.percentile(dist, 85)), 1.0)
    closeness = np.clip(1.0 - dist / scale, 0.0, 1.0)
    brightness = np.clip(
        0.04 + 0.42 * closeness * rng.uniform(0.55, 1.45, size=len(src)),
        0.03,
        0.58,
    )
    from matplotlib.colors import to_rgba
    colours = np.tile(np.asarray(to_rgba(C.C_GREY)), (len(src), 1))
    colours[:, 3] = brightness
    friend_lines = LineCollection(
        segs,
        colors=colours,
        linewidths=0.30 + 1.15 * closeness,
        zorder=1,
    )
    ax.add_collection(friend_lines)
    sc_p = ax.scatter(
        xy[~blue, 0], xy[~blue, 1], s=s, c=C.C_PINK_BRIGHT,
        alpha=0.82, linewidths=0, zorder=2,
    )
    sc_b = ax.scatter(
        xy[blue, 0], xy[blue, 1], s=s, c=C.C_BLUE_BRIGHT,
        alpha=0.82, linewidths=0, zorder=2,
    )
    on = np.zeros(n, dtype=bool)
    on[hist[0]] = True
    glow = ax.scatter(
        xy[on, 0], xy[on, 1], s=s + (260 if n <= 100 else 55),
        c=C.C_COUNCIL, alpha=0.16, linewidths=0, zorder=3,
    )
    sc_c = ax.scatter(
        xy[on, 0], xy[on, 1], s=s + (75 if n <= 100 else 22),
        facecolors="none", edgecolors=C.C_COUNCIL,
        linewidths=1.8 if n <= 100 else 0.9, zorder=4,
    )
    ax.set_xlim(-105, 105)
    ax.set_ylim(-105, 105)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    label = dict(
        color=C.DARK_TEXT, fontsize=22, fontweight="bold",
        transform=ax.transAxes, clip_on=False,
    )
    ax.text(-0.03, 0.50, "Pink", ha="right", va="center", **label)
    ax.text(1.03, 0.50, "Blue", ha="left", va="center", **label)
    ax.text(0.50, -0.03, "Cat", ha="center", va="top", **label)
    ax.text(0.50, 1.03, "Dog", ha="center", va="bottom", **label)

    year_text = fig.text(
        0.05, 0.90, "Year 0", color=C.DARK_TEXT, fontsize=26,
        fontweight="bold", ha="left", va="center",
    )

    n_rounds = len(hist) - 1
    sub = 8
    ambient = C.make_ambient(n, amp=0.22, seed=23, period=168.0)

    def update(f):
        seg = min(f / sub, n_rounds)
        t0 = int(np.floor(seg))
        t1 = min(t0 + 1, n_rounds)
        u = seg - t0
        ease = u * u * (3.0 - 2.0 * u)
        council = hist[t1 if ease >= 0.5 else t0]
        moved = xy[:, :2] + ambient(f)
        sc_p.set_offsets(moved[~blue])
        sc_b.set_offsets(moved[blue])
        friend_lines.set_segments(
            np.stack((moved[src], moved[dst]), axis=1)
        )
        on = np.zeros(n, dtype=bool)
        on[council] = True
        glow.set_offsets(moved[on])
        sc_c.set_offsets(moved[on])
        year_text.set_text(f"Year {int(round(seg))}")
        return sc_p, sc_b, friend_lines, glow, sc_c, year_text

    anim = FuncAnimation(
        fig, update, frames=n_rounds * sub + 1, interval=1000 / 24, blit=False,
    )
    C.save_mp4(anim, out, fps=24, bg=black)
    plt.close()


def animate_final_sequence(
    neutral: dict,
    biased: dict,
    n: int,
    out: Path,
) -> None:
    """One narrated sequence: inherited colour bias, then colour + animal."""
    black = "#000000"
    fps = 24
    xy0 = neutral["snap"]["xy"]
    xy1 = biased["snap"]["xy"]
    hist0 = neutral["snap"]["hist"]
    council0 = hist0[0]
    year5 = min(5, len(hist0) - 1)
    council_year5 = hist0[year5]
    friends0 = neutral["snap"]["friends"]
    friends1 = biased["snap"]["friends"]
    was_male = np.asarray(neutral["snap"]["was_male"], dtype=bool)
    blue = xy0[:, 0] > 0
    pink_rgba = np.asarray(to_rgba(C.C_PINK_BRIGHT))
    blue_rgba = np.asarray(to_rgba(C.C_BLUE_BRIGHT))
    colour_cols = np.where(blue[:, None], blue_rgba, pink_rgba)

    fig = plt.figure(figsize=(16, 9), facecolor=black)
    ax = fig.add_axes([0.255, 0.14, 0.49, 0.72])
    C.style_dark_axis(ax, grid=False)
    fig.patch.set_facecolor(black)
    ax.set_facecolor(black)
    ax.axvline(0, color=C.C_GREY, lw=0.8, alpha=0.7)
    ax.axhline(0, color=C.C_GREY, lw=0.8, alpha=0.7)
    ax.set_xlim(-105, 105)
    ax.set_ylim(-105, 105)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    label = dict(
        color=C.DARK_TEXT, fontsize=22, fontweight="bold",
        transform=ax.transAxes, clip_on=False,
    )
    ax.text(-0.03, 0.50, "Pink", ha="right", va="center", **label)
    ax.text(1.03, 0.50, "Blue", ha="left", va="center", **label)
    ax.text(0.50, -0.03, "Cat", ha="center", va="top", **label)
    ax.text(0.50, 1.03, "Dog", ha="center", va="bottom", **label)

    def packed_edges(friends):
        src = np.repeat(np.arange(n), friends.shape[1])
        dst = friends.ravel()
        lo = np.minimum(src, dst).astype(np.int64)
        hi = np.maximum(src, dst).astype(np.int64)
        packed = np.unique(lo * n + hi)
        keep = (
            ((packed * 2654435761) % (2 ** 32)) / (2 ** 32)
            < 0.40
        )
        packed = packed[keep]
        return packed // n, packed % n, packed

    def edge_style(xy, src, dst, packed):
        dist = np.linalg.norm(xy[src, :2] - xy[dst, :2], axis=1)
        scale = max(
            float(np.percentile(dist, 85)) if len(dist) else 1.0, 1.0
        )
        closeness = np.clip(1.0 - dist / scale, 0.0, 1.0)
        noise = 0.55 + 0.90 * (
            ((packed * 1103515245 + 12345) % (2 ** 31)) / (2 ** 31)
        )
        brightness = np.clip(
            0.04 + 0.42 * closeness * noise, 0.03, 0.58
        )
        colours = np.tile(
            np.asarray(to_rgba(C.C_GREY)), (len(src), 1)
        )
        colours[:, 3] = brightness
        return colours, 0.30 + 1.15 * closeness

    src0, dst0, packed0 = packed_edges(friends0)
    src1, dst1, packed1 = packed_edges(friends1)
    colours0, widths0 = edge_style(xy0, src0, dst0, packed0)
    colours1, widths1 = edge_style(xy1, src1, dst1, packed1)
    delays = np.random.default_rng(19).uniform(0.0, 0.55, len(src0))

    friend_lines = LineCollection([], zorder=1)
    ax.add_collection(friend_lines)
    dot_size = 36 if n <= 100 else 10
    people = ax.scatter(
        xy0[:, 0], xy0[:, 1], s=dot_size,
        c=colour_cols, linewidths=0, zorder=2,
    )
    men_ring = ax.scatter(
        [], [], s=dot_size + 48, facecolors="none",
        edgecolors=C.DARK_TEXT, linewidths=1.3, alpha=0.0, zorder=3.5,
    )
    glow = ax.scatter(
        [], [], s=dot_size + 260, c=C.C_COUNCIL,
        alpha=0.0, linewidths=0, zorder=3,
    )
    council_rings = ax.scatter(
        [], [], s=dot_size + 75, facecolors="none",
        edgecolors=C.C_COUNCIL, linewidths=1.8, alpha=0.0, zorder=4,
    )
    region_p = Circle(
        (0, 0), 40, facecolor=C.C_COUNCIL, edgecolor=C.C_COUNCIL,
        lw=1.2, alpha=0.0, zorder=0.6,
    )
    region_b = Circle(
        (0, 0), 40, facecolor=C.C_COUNCIL, edgecolor=C.C_COUNCIL,
        lw=1.2, alpha=0.0, zorder=0.6,
    )
    ax.add_patch(region_p)
    ax.add_patch(region_b)

    def group_centre(moved, mask):
        return np.median(moved[mask], axis=0)

    ambient = C.make_ambient(n, amp=0.22, seed=53, period=168.0)

    def smooth(value):
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    durations = {
        "origin_in": 2 * fps,
        "origin_hold": 6 * fps,
        "origin_out": int(2.5 * fps),
        "dots": 3 * fps,
        "leaders": 2 * fps,
        "friends": 4 * fps,
        "simulation": 12 * fps,
        "reset": 2 * fps,
        "start_pause": 4 * fps,
        "focus": int(1.5 * fps),
        "focus_pause": 5 * fps,
        "focus_out": int(1.5 * fps),
        "bias_map_in": 2 * fps,
        "bias_shift": 3 * fps,
        "bias_map_hold": 5 * fps,
        "bias_map_out": int(2.5 * fps),
        "biased_pause": 2 * fps,
        "biased_focus": int(1.5 * fps),
        "ending": 5 * fps,
    }
    order = tuple(durations)
    starts = {}
    cursor = 0
    for name in order:
        starts[name] = cursor
        cursor += durations[name]
    total_frames = cursor

    def stage(frame):
        for name in order:
            start = starts[name]
            if frame < start + durations[name]:
                return name, frame - start
        return order[-1], durations[order[-1]] - 1

    def set_edges(moved, which, alpha=1.0, progress=1.0):
        if which == 0:
            src, dst = src0, dst0
            colours, widths = colours0.copy(), widths0
        else:
            src, dst = src1, dst1
            colours, widths = colours1.copy(), widths1
        if np.isscalar(progress):
            progress = np.full(len(src), float(progress))
        progress = np.clip(np.asarray(progress), 0.0, 1.0)
        alive = progress > 1e-4
        start = moved[src]
        end = start + progress[:, None] * (moved[dst] - start)
        colours[:, 3] *= alpha
        friend_lines.set_segments(
            np.stack((start[alive], end[alive]), axis=1)
        )
        friend_lines.set_colors(colours[alive])
        friend_lines.set_linewidths(widths[alive])

    def set_council(moved, council, alpha):
        council_xy = moved[council]
        glow.set_offsets(council_xy)
        council_rings.set_offsets(council_xy)
        glow.set_alpha(0.16 * alpha)
        council_rings.set_alpha(alpha)

    def clear_focus():
        region_p.set_alpha(0.0)
        region_b.set_alpha(0.0)

    def set_focus(moved, amount, both=True):
        amount = smooth(amount)
        fill = 0.18 * amount
        pink_c = group_centre(moved, ~blue)
        blue_c = group_centre(moved, blue)
        region_p.center = (float(pink_c[0]), float(pink_c[1]))
        region_b.center = (float(blue_c[0]), float(blue_c[1]))
        region_p.set_alpha(fill if both else 0.0)
        region_b.set_alpha(fill)

    def update(frame):
        name, local = stage(frame)
        fraction = local / max(durations[name] - 1, 1)
        amb = ambient(frame)
        base = xy0[:, :2]
        moved = base + amb
        council = council0
        council_alpha = 1.0
        edge_alpha = 1.0
        edge_world = 0
        edge_progress = 1.0
        men_alpha = 0.0
        people_alpha = 0.88
        clear_focus()

        if name == "origin_in":
            council_alpha = 0.0
            edge_alpha = 0.0
            people_alpha = 0.88 * smooth(fraction)
        elif name == "origin_hold":
            council_alpha = 0.0
            edge_alpha = 0.0
            men_alpha = float(min(fraction * 3.0, 1.0))
        elif name == "origin_out":
            council_alpha = 0.0
            edge_alpha = 0.0
            men_alpha = 1.0 - smooth(fraction)
        elif name == "dots":
            council_alpha = 0.0
            edge_alpha = 0.0
        elif name == "leaders":
            council_alpha = smooth(fraction)
            edge_alpha = 0.0
        elif name == "friends":
            local_progress = (fraction - delays) / 0.10
            edge_progress = np.clip(local_progress, 0.0, 1.0)
        elif name == "simulation":
            sim = fraction * (len(hist0) - 1)
            council = hist0[min(int(round(sim)), len(hist0) - 1)]
        elif name == "reset":
            if fraction < 0.5:
                council = hist0[-1]
                council_alpha = 1.0 - smooth(fraction * 2.0)
            else:
                council = council_year5
                council_alpha = smooth((fraction - 0.5) * 2.0)
        elif name == "start_pause":
            council = council_year5
        elif name == "focus":
            council = council_year5
            set_focus(moved, fraction, both=True)
        elif name == "focus_pause":
            council = council_year5
            set_focus(moved, 1.0, both=True)
        elif name == "focus_out":
            council = council_year5
            set_focus(moved, 1.0 - smooth(fraction), both=True)
        elif name == "bias_map_in":
            council = council_year5
            men_alpha = smooth(fraction)
        elif name == "bias_shift":
            shift = smooth(fraction)
            base = xy0[:, :2] * (1.0 - shift) + xy1[:, :2] * shift
            moved = base + amb
            council = council_year5
            edge_world = 0 if fraction < 0.5 else 1
            edge_alpha = abs(2.0 * fraction - 1.0)
            men_alpha = 1.0
        elif name == "bias_map_hold":
            base = xy1[:, :2]
            moved = base + amb
            council = council_year5
            edge_world = 1
            men_alpha = 1.0
        elif name == "bias_map_out":
            base = xy1[:, :2]
            moved = base + amb
            council = council_year5
            edge_world = 1
            men_alpha = 1.0 - smooth(fraction)
        elif name == "biased_pause":
            base = xy1[:, :2]
            moved = base + amb
            council = council_year5
            edge_world = 1
        elif name == "biased_focus":
            base = xy1[:, :2]
            moved = base + amb
            council = council_year5
            edge_world = 1
            set_focus(moved, fraction, both=True)
        else:
            base = xy1[:, :2]
            moved = base + amb
            council = council_year5
            edge_world = 1
            set_focus(moved, 1.0, both=True)

        cols = colour_cols.copy()
        cols[:, 3] = people_alpha
        people.set_offsets(moved)
        people.set_facecolors(cols)
        men_ring.set_offsets(moved[was_male])
        men_ring.set_alpha(men_alpha)
        if edge_alpha <= 0:
            friend_lines.set_segments([])
        else:
            set_edges(
                moved, edge_world, alpha=edge_alpha,
                progress=edge_progress,
            )
        set_council(moved, council, council_alpha)
        return (
            people, men_ring, friend_lines, glow, council_rings,
            region_p, region_b,
        )

    anim = FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps, blit=False,
    )
    C.save_mp4(anim, out, fps=fps, bg=black)
    plt.close(fig)


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
            animate(results, n, HERE / f"scene2_n{n}.mp4", mode="neutral")
            animate(results, n, HERE / f"scene2_n{n}_v2.mp4", mode="inverted")
            matched_neutral = simulate_mode(
                n, seats, k, "neutral", FINAL_ANIMATION_SEED, n_runs=1
            )
            matched_biased = simulate_mode(
                n, seats, k, "inverted", FINAL_ANIMATION_SEED, n_runs=1
            )
            animate_final_sequence(
                matched_neutral,
                matched_biased,
                n,
                HERE / "scene4_final.mp4",
            )


if __name__ == "__main__":
    main()
