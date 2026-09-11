# -*- coding: utf-8 -*-
"""
Scene 3 — Burn the city down and start again
============================================
How do we fix the patriarchy? Start society from scratch. Create every
person equally. They still have two personality traits (colour and animal),
but there is no gender, no leftover stereotype, and no intent.

They operate on four rules:

  1. People form networks (20 nearest neighbours on the trait plane).
  2. People get influenced (copy friends, plus a little noise).
  3. People die.
  4. New people are born.

We compare two birth rules, because it matters:

  - random: a newborn's traits are drawn independently (fresh draw from
    the same Gaussian, clipped to ±100).
  - inherited: a newborn's traits sit near two random living "parents",
    plus mutation. Similarity can pile up across generations.

Each year: rebuild friendships, influence, replace some people, then elect
the same two-step way as scenes 1 and 2 (named, then chosen). Every year,
each trait is linearly rescaled to fill -100 to +100 so the drawing
remains legible without a continuous outward force. Learning starts here.
The city starts balanced.

If equality were a stable equilibrium, colour (and animal) would jitter
around 50/50 forever. If it is unstable, a small random bump compounds:
friends copy friends, clusters thicken, and whoever lucks into a bit of
the council keeps winning because their cluster holds more friends in
power.

This file runs many worlds from scratch. Some end blue-heavy, some pink-
heavy. That is the point — the side is an accident; the lock-in is not.
Random births fight clustering; inherited births help it. The plots show
which rule actually keeps a status quo of power.

Run:
  python scene3.py
  python scene3.py --size small
  python scene3.py --replay --size large
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS_PATH = HERE / "replay_seeds.json"
N_YEARS = 120
N_RUNS_SMALL = 40
N_RUNS_LARGE = 16
INFLUENCE = 0.08
NOISE = 3.0
RESCALE_EVERY = 1
DEATH_FRAC = 0.05
MUTATION = 12.0
INIT_MALE_FRAC = 0.50  # start equal — no gender leftover


def random_traits(n: int, rng: np.random.Generator) -> np.ndarray:
    return C.clip_traits(rng.normal(0.0, C.XY_SIGMA, size=(n, 2)))


def influence(xy: np.ndarray, friends: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return C.learn_from_friends(xy, friends, rng, INFLUENCE, NOISE)


def replace(xy: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    n = len(xy)
    n_die = max(1, int(round(DEATH_FRAC * n)))
    die = rng.choice(n, size=n_die, replace=False)
    if mode == "random":
        xy[die] = random_traits(n_die, rng)
    elif mode == "inherited":
        parents = rng.integers(0, n, size=(n_die, 2))
        mid = xy[parents].mean(axis=1)
        xy[die] = C.clip_traits(mid + rng.normal(0.0, MUTATION, size=mid.shape))
    else:
        raise ValueError(mode)
    return xy


def seed_council(n: int, seats: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n, size=seats, replace=False)


def step(xy, council, mode, k, rng, rescale=False):
    friends = C.knn_friends(xy, k)
    xy = influence(xy, friends, rng)
    xy = replace(xy, mode, rng)
    if rescale:
        xy = C.rescale_traits(xy)
    friends = C.knn_friends(xy, k)
    council = C.elect(council, friends, rng)
    return xy, friends, council


def blue_share(xy, council) -> float:
    return 100.0 * (xy[council, 0] > 0).mean()


def polarization(xy) -> float:
    """Mean |colour| — higher means the city has left the middle."""
    return float(np.abs(xy[:, 0]).mean())


def simulate_run(n, seats, k, mode, years, rng, record_xy=False):
    xy = random_traits(n, rng)
    friends = C.knn_friends(xy, k)
    council = seed_council(n, seats, rng)
    shares = [blue_share(xy, council)]
    pol = [polarization(xy)]
    frames = [(xy.copy(), council.copy(), friends.copy())] if record_xy else None
    for year in range(1, years + 1):
        xy, friends, council = step(
            xy, council, mode, k, rng, rescale=year % RESCALE_EVERY == 0
        )
        shares.append(blue_share(xy, council))
        pol.append(polarization(xy))
        if record_xy:
            frames.append((xy.copy(), council.copy(), friends.copy()))
    return np.array(shares), np.array(pol), frames


def summarize(paths: np.ndarray) -> dict:
    end = paths[:, -1]
    late = paths[:, N_YEARS // 2 :]
    # time spent away from 45–55
    imbalance = np.mean(np.abs(late - 50) > 15, axis=1)
    flips = 0
    for p in paths:
        s = np.sign(p - 50)
        s[s == 0] = 1
        if np.any(np.diff(s) != 0) and (p[0] - 50) * (p[-1] - 50) < 0:
            flips += 1
    return {
        "end_mean": float(end.mean()),
        "end_abs": float(np.abs(end - 50).mean()),
        "frac_blue_win": float((end > 60).mean()),
        "frac_pink_win": float((end < 40).mean()),
        "frac_imbalanced": float(imbalance.mean()),
        "flips": flips,
        "n_runs": len(paths),
    }


def plot_stats(bundle: dict, n: int, seats: int, out: Path) -> None:
    fig = plt.figure(figsize=(14.0, 9.4), facecolor="white")
    fig.suptitle(
        f"Scene 3 — Equality from scratch  ·  N={n}, {seats} seats, {N_YEARS} years\n"
        "influence + death/birth + named-then-chosen elections",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26,
                  left=0.07, right=0.97, top=0.86, bottom=0.08)
    for col, mode in enumerate(("random", "inherited")):
        b = bundle[mode]
        ax = fig.add_subplot(gs[0, col])
        t = np.arange(len(b["paths"][0]))
        for p in b["paths"]:
            ax.plot(t, p, color=C.C_BLUE if p[-1] >= 50 else C.C_PINK, lw=1.0, alpha=0.35)
        ax.axhline(50, color="0.4", ls="--", lw=1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Year")
        ax.set_ylabel("% council blue")
        ax.set_title("Random births" if mode == "random" else "Inherited births")
        ax.grid(alpha=0.25)
        s = b["sum"]
        ax.text(0.02, 0.04,
                f"end |blue−50|: {s['end_abs']:.1f} pp\n"
                f"blue-heavy: {s['frac_blue_win']*100:.0f}%  pink-heavy: {s['frac_pink_win']*100:.0f}%",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))

        ax = fig.add_subplot(gs[1, col])
        for p in b["pol"]:
            ax.plot(t, p, color="0.45", lw=0.9, alpha=0.3)
        ax.plot(t, np.mean(b["pol"], 0), color="black", lw=1.6)
        ax.set_xlabel("Year")
        ax.set_ylabel("mean |colour| in the city")
        ax.set_title("Trait polarisation")
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(out, dpi=140)
    plt.close()
    print(f"  plot → {out}")


def packed_friend_edges(friends: np.ndarray, people: np.ndarray, viz_k: int, n: int) -> np.ndarray:
    """Unique undirected (i, j) pairs among a displayed subset of friendships."""
    k = min(viz_k, friends.shape[1])
    src = np.repeat(people, k)
    dst = friends[people, :k].ravel()
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    packed = lo.astype(np.int64) * n + hi
    return np.unique(packed)


def unpack_edges(packed: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    return packed // n, packed % n


def growing_segments(xy: np.ndarray, packed: np.ndarray, n: int, progress: float) -> np.ndarray:
    """Draw each edge from the lower-index person toward the other, length = progress."""
    if len(packed) == 0 or progress <= 0:
        return np.empty((0, 2, 2))
    a, b = unpack_edges(packed, n)
    start = xy[a]
    end = start + progress * (xy[b] - start)
    return np.stack([start, end], axis=1)


def animate_pink_run(frames, blue_shares: np.ndarray, n: int, out: Path) -> None:
    """A slow, uncluttered replay of the inherited world that went pink."""
    BLACK = "#000000"
    fig = plt.figure(figsize=(16, 9), facecolor=BLACK)
    ax = fig.add_axes([0.255, 0.14, 0.49, 0.72])
    axp = fig.add_axes([0.77, 0.72, 0.20, 0.20])
    C.style_dark_figure(fig, (ax, axp))
    fig.patch.set_facecolor(BLACK)
    ax.set_facecolor(BLACK)
    axp.set_facecolor(BLACK)
    s = 36 if n <= 100 else 10
    viz_k = 8 if n <= 100 else 4
    people = np.arange(n) if n <= 100 else np.linspace(0, n - 1, 90, dtype=int)

    xy, council, friends0 = frames[0]
    blue = xy[:, 0] > 0
    ax.axvline(0, color=C.DARK_GRID, lw=0.8)
    ax.axhline(0, color=C.DARK_GRID, lw=0.8)
    friend_lines = LineCollection(
        growing_segments(xy, packed_friend_edges(friends0, people, viz_k, n), n, 1.0),
        colors="#9AA8B8", linewidths=0.55, alpha=0.28, zorder=1,
    )
    ax.add_collection(friend_lines)
    sc_p = ax.scatter(
        xy[~blue, 0], xy[~blue, 1], s=s,
        c=C.C_PINK_BRIGHT, alpha=0.82, linewidths=0, zorder=2,
    )
    sc_b = ax.scatter(
        xy[blue, 0], xy[blue, 1], s=s,
        c=C.C_BLUE_BRIGHT, alpha=0.82, linewidths=0, zorder=2,
    )
    on = np.zeros(len(xy), dtype=bool)
    on[council] = True
    sc_c = ax.scatter(
        xy[on, 0], xy[on, 1], s=s + 24, facecolors="none",
        edgecolors=C.C_COUNCIL, linewidths=1.5, zorder=3,
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

    pink_shares = 100.0 - np.asarray(blue_shares)
    line, = axp.plot([], [], color=C.C_PINK_BRIGHT, lw=3.0)
    dot, = axp.plot([], [], "o", color=C.C_PINK_BRIGHT, ms=6)
    axp.axhline(50, color=C.DARK_MUTED, ls="--", lw=0.9, alpha=0.6)
    axp.set_xlim(0, N_YEARS)
    axp.set_ylim(0, 100)
    axp.set_xticks([0, N_YEARS])
    axp.set_yticks([0, 50, 100])
    axp.tick_params(labelsize=9)
    axp.set_title("% pink", color=C.C_PINK_BRIGHT, fontsize=15, fontweight="bold")

    # Interpolate positions between yearly snapshots for smooth motion.
    positions = np.stack([f[0] for f in frames])  # (years+1, n, 2)
    councils = [f[1] for f in frames]
    edge_years = [
        packed_friend_edges(f[2], people, viz_k, n) for f in frames
    ]
    n_years = len(frames) - 1
    sub = 6  # interpolated frames per year

    def update(f):
        seg = min(f / sub, n_years)
        t0 = int(np.floor(seg))
        t1 = min(t0 + 1, n_years)
        u = seg - t0
        ease = u * u * (3.0 - 2.0 * u)  # smoothstep
        xy = positions[t0] * (1.0 - ease) + positions[t1] * ease
        blue = xy[:, 0] > 0
        sc_p.set_offsets(xy[~blue])
        sc_b.set_offsets(xy[blue])
        council = councils[t1 if ease >= 0.5 else t0]
        on = np.zeros(len(xy), dtype=bool)
        on[council] = True
        sc_c.set_offsets(xy[on])
        old = edge_years[t0]
        new = edge_years[t1]
        stay = np.intersect1d(old, new, assume_unique=True)
        appear = np.setdiff1d(new, old, assume_unique=True)
        vanish = np.setdiff1d(old, new, assume_unique=True)
        segs = np.concatenate([
            growing_segments(xy, stay, n, 1.0),
            growing_segments(xy, appear, n, ease),
            growing_segments(xy, vanish, n, 1.0 - ease),
        ], axis=0)
        friend_lines.set_segments(segs)
        whole = np.arange(t0 + 1)
        share_now = pink_shares[t0] * (1.0 - ease) + pink_shares[t1] * ease
        line.set_data(
            np.append(whole, seg), np.append(pink_shares[: t0 + 1], share_now)
        )
        dot.set_data([seg], [share_now])
        year_text.set_text(f"Year {int(round(seg))}")
        return sc_p, sc_b, sc_c, friend_lines, line, dot, year_text

    anim = FuncAnimation(
        fig, update, frames=n_years * sub + 1, interval=40, blit=False,
    )
    C.save_mp4(anim, out, fps=24, bg=BLACK)
    plt.close()


def load_replay_seeds() -> dict:
    if not SEEDS_PATH.exists():
        return {}
    return json.loads(SEEDS_PATH.read_text())


def save_replay_seeds(n: int, blue_seed: int, pink_seed: int,
                      blue_end: float, pink_end: float, cli_seed: int) -> None:
    data = load_replay_seeds()
    data.setdefault("mode", "inherited")
    data.setdefault("years", N_YEARS)
    data["cli_seed"] = cli_seed
    worlds = data.setdefault("worlds", {})
    worlds[str(n)] = {
        "blue": int(blue_seed),
        "pink": int(pink_seed),
        "blue_end_pct": round(float(blue_end), 2),
        "pink_end_pct": round(float(pink_end), 2),
    }
    SEEDS_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  stored replay seeds → {SEEDS_PATH}  (n={n}  blue={blue_seed}  pink={pink_seed})")


def run_from_seed(n, seats, k, mode, seed, years, record_xy=True):
    rng = np.random.default_rng(seed)
    return simulate_run(n, seats, k, mode, years, rng, record_xy=record_xy)


def pick_pink_frames(n, seats, k, mode, seed_base, n_runs, years, stored=None):
    """Return a reproducible inherited run whose council ends pink-majority."""
    if stored is not None:
        pink_seed = int(stored["pink"])
        pink_path, _, pink_frames = run_from_seed(
            n, seats, k, mode, pink_seed, years
        )
        if pink_path[-1] < 50:
            return pink_frames, pink_path, pink_seed

    for run in range(n_runs * 3):
        pink_seed = seed_base + 9000 + run
        pink_path, _, pink_frames = run_from_seed(
            n, seats, k, mode, pink_seed, years
        )
        if pink_path[-1] < 40:
            return pink_frames, pink_path, pink_seed
    raise RuntimeError("could not find a pink-majority replay seed")


def add_cli(p):
    p.add_argument(
        "--replay", action="store_true",
        help="recreate stored pink/blue inherited worlds (skip ensemble stats)",
    )


def main():
    args = C.parse_cli("Scene 3 — start from scratch", extra_args=add_cli)
    print("Scene 3 — influence, death, birth, named-then-chosen elections")
    stored_all = load_replay_seeds().get("worlds", {})
    for size in C.sizes_to_run(args.size):
        cfg = C.SIZES[size]
        n, seats, k = cfg["n"], cfg["council"], cfg["k_friends"]
        n_runs = N_RUNS_SMALL if size == "small" else N_RUNS_LARGE
        print(f"\n[{size}] N={n}  seats={seats}  years={N_YEARS}  runs/mode={n_runs}")
        bundle = {}
        if not args.replay:
            for m_i, mode in enumerate(("random", "inherited")):
                paths, pols = [], []
                for run in range(n_runs):
                    rng = np.random.default_rng(args.seed + 2000 * m_i + run)
                    sh, pol, _ = simulate_run(n, seats, k, mode, N_YEARS, rng, record_xy=False)
                    paths.append(sh); pols.append(pol)
                paths, pols = np.array(paths), np.array(pols)
                s = summarize(paths)
                bundle[mode] = {"paths": paths, "pol": pols, "sum": s}
                print(f"  {mode}: |end−50|={s['end_abs']:.1f}  "
                      f"blue-heavy {s['frac_blue_win']*100:.0f}%  "
                      f"pink-heavy {s['frac_pink_win']*100:.0f}%  "
                      f"sign-flip runs {s['flips']}/{s['n_runs']}")
            plot_stats(bundle, n, seats, HERE / f"scene3_n{n}.png")
        if args.animate or args.replay:
            stored = stored_all.get(str(n)) if args.replay else None
            if args.replay and stored is None:
                raise SystemExit(f"no stored pink/blue seeds for N={n} in {SEEDS_PATH}")
            print("  recording inherited pink-majority world for animation…")
            if stored:
                print(f"    replay pink seed={stored['pink']}")
            pf, pp, pseed = pick_pink_frames(
                n, seats, k, "inherited", args.seed, n_runs, N_YEARS, stored=stored,
            )
            print(f"    inherited went pink  seed={pseed}  end {pp[-1]:.1f}% blue")
            animate_pink_run(pf, pp, n, HERE / f"scene3_n{n}.mp4")


if __name__ == "__main__":
    main()
