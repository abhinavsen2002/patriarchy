# -*- coding: utf-8 -*-
"""
Scene 4 — Merit wins
====================
Meritopolis tries one final reform. Networks no longer nominate or choose
the council. Everyone receives an eleventh attribute called merit, and the
people with the highest merit simply take the seats.

The rest of society is unchanged from Scene 3. Each person has ten social
traits (colour, animal, and eight traits that are not drawn) plus merit.
Friendships are the nearest people in all eleven dimensions. Every year:

  1. People learn from their friends across all eleven traits.
  2. Five percent of the city dies and is replaced.
  3. The top-merit people become the council.

Merit is not protected from society. It affects who becomes friends, moves
toward friends through learning, and is random or inherited at birth just
like every other trait. We compare those two birth rules.

Every year, each trait is linearly rescaled to fill -100 to +100.
This keeps the visual space legible without continuously pushing people
toward either the origin or the extremes.

The election itself contains no incumbency, nomination, network weighting,
or lottery. Ranking by merit is the entire rule.

Run:
  python scene4.py
  python scene4.py --size small
  python scene4.py --size large --no-animate
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
N_YEARS = 120
N_RUNS_SMALL = 40
N_RUNS_LARGE = 16
N_TRAITS = 11
COLOUR = 0
ANIMAL = 1
MERIT = 10
INFLUENCE = 0.08
NOISE = 3.0
RESCALE_EVERY = 1
DEATH_FRAC = 0.05
MUTATION = 12.0


def random_traits(n: int, rng: np.random.Generator) -> np.ndarray:
    """Ten social traits plus merit, all equal and independent at creation."""
    return C.clip_traits(rng.normal(0.0, C.XY_SIGMA, size=(n, N_TRAITS)))


def knn_friends(traits: np.ndarray, k: int) -> np.ndarray:
    """k-NN in all eleven dimensions without allocating an n×n×11 tensor."""
    norm = np.einsum("ij,ij->i", traits, traits)
    d2 = norm[:, None] + norm[None, :] - 2.0 * traits @ traits.T
    np.fill_diagonal(d2, np.inf)
    k = min(k, len(traits) - 1)
    return np.argpartition(d2, kth=k - 1, axis=1)[:, :k]


def influence(
    traits: np.ndarray,
    friends: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return C.learn_from_friends(traits, friends, rng, INFLUENCE, NOISE)


def replace(
    traits: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(traits)
    n_die = max(1, int(round(DEATH_FRAC * n)))
    die = rng.choice(n, size=n_die, replace=False)
    if mode == "random":
        traits[die] = random_traits(n_die, rng)
    elif mode == "inherited":
        parents = rng.integers(0, n, size=(n_die, 2))
        midpoint = traits[parents].mean(axis=1)
        newborns = midpoint + rng.normal(0.0, MUTATION, size=midpoint.shape)
        traits[die] = C.clip_traits(newborns)
    else:
        raise ValueError(mode)
    return traits


def elect_by_merit(traits: np.ndarray, seats: int) -> np.ndarray:
    """Return exactly the highest-merit people; ties use stable index order."""
    order = np.argsort(traits[:, MERIT], kind="stable")
    return order[-seats:]


def assert_merit_council(
    traits: np.ndarray,
    council: np.ndarray,
    seats: int,
) -> None:
    assert len(council) == seats
    assert len(np.unique(council)) == seats
    on = np.zeros(len(traits), dtype=bool)
    on[council] = True
    assert traits[council, MERIT].min() >= traits[~on, MERIT].max() - 1e-12


def measurements(traits: np.ndarray, council: np.ndarray) -> tuple[float, float, float]:
    blue_share = 100.0 * (traits[council, COLOUR] > 0).mean()
    merit_cutoff = float(traits[council, MERIT].min())
    merit_gap = float(
        traits[council, MERIT].mean() - traits[:, MERIT].mean()
    )
    return blue_share, merit_cutoff, merit_gap


def step(
    traits: np.ndarray,
    mode: str,
    seats: int,
    k: int,
    rng: np.random.Generator,
    rescale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    friends = knn_friends(traits, k)
    traits = influence(traits, friends, rng)
    traits = replace(traits, mode, rng)
    if rescale:
        traits = C.rescale_traits(traits)
    council = elect_by_merit(traits, seats)
    assert_merit_council(traits, council, seats)
    return traits, council


def simulate_run(
    n: int,
    seats: int,
    k: int,
    mode: str,
    years: int,
    rng: np.random.Generator,
    record_frames: bool = False,
) -> dict:
    traits = random_traits(n, rng)
    council = elect_by_merit(traits, seats)
    assert_merit_council(traits, council, seats)

    blue, cutoff, gap = measurements(traits, council)
    blue_path = [blue]
    cutoff_path = [cutoff]
    gap_path = [gap]
    frames = (
        [(traits.copy(), council.copy())]
        if record_frames
        else None
    )

    for year in range(1, years + 1):
        traits, council = step(
            traits,
            mode,
            seats,
            k,
            rng,
            rescale=year % RESCALE_EVERY == 0,
        )
        blue, cutoff, gap = measurements(traits, council)
        blue_path.append(blue)
        cutoff_path.append(cutoff)
        gap_path.append(gap)
        if record_frames:
            frames.append((traits.copy(), council.copy()))

    return {
        "blue": np.asarray(blue_path),
        "cutoff": np.asarray(cutoff_path),
        "gap": np.asarray(gap_path),
        "frames": frames,
        "final_traits": traits,
        "final_council": council,
    }


def summarize(runs: list[dict]) -> dict:
    blue = np.asarray([run["blue"] for run in runs])
    cutoff = np.asarray([run["cutoff"] for run in runs])
    gap = np.asarray([run["gap"] for run in runs])
    end = blue[:, -1]
    return {
        "blue": blue,
        "cutoff": cutoff,
        "gap": gap,
        "end_abs": float(np.abs(end - 50).mean()),
        "blue_heavy": float((end > 60).mean()),
        "pink_heavy": float((end < 40).mean()),
        "end_gap": float(gap[:, -1].mean()),
    }


def scatter_merit(
    ax,
    traits: np.ndarray,
    council: np.ndarray,
    n: int,
    dark: bool = False,
) -> None:
    blue = traits[:, COLOUR] > 0
    size = 18 if n <= 100 else 5
    pink = C.C_PINK_BRIGHT if dark else C.C_PINK
    blue_colour = C.C_BLUE_BRIGHT if dark else C.C_BLUE
    council_colour = C.C_COUNCIL if dark else "black"
    ax.scatter(
        traits[~blue, COLOUR],
        traits[~blue, ANIMAL],
        traits[~blue, MERIT],
        s=size,
        c=pink,
        alpha=0.72 if dark else 0.48,
        linewidths=0,
    )
    ax.scatter(
        traits[blue, COLOUR],
        traits[blue, ANIMAL],
        traits[blue, MERIT],
        s=size,
        c=blue_colour,
        alpha=0.72 if dark else 0.48,
        linewidths=0,
    )
    ax.scatter(
        traits[council, COLOUR],
        traits[council, ANIMAL],
        traits[council, MERIT],
        s=size + 24,
        marker="o",
        facecolors="none",
        edgecolors=council_colour,
        linewidths=1.15 if dark else 0.8,
        depthshade=False,
    )
    ax.set_xlim(-105, 105)
    ax.set_ylim(-105, 105)
    ax.set_zlim(-105, 105)
    ax.set_xlabel("colour", labelpad=2)
    ax.set_ylabel("animal", labelpad=2)
    ax.set_zlabel("merit", labelpad=3)
    ax.view_init(elev=22, azim=-58)
    if dark:
        C.style_dark_axis(ax)


def plot_stats(
    bundle: dict,
    snapshots: dict,
    n: int,
    seats: int,
    k: int,
    out: Path,
) -> None:
    fig = plt.figure(figsize=(14.2, 13.0), facecolor="white")
    fig.suptitle(
        f"Scene 4 — Merit wins  ·  N={n}, {seats} seats, {k} friends, "
        f"{N_TRAITS} traits\n"
        "friendship + learning + birth shape merit; elections take the top "
        "merit ranks only",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.36,
        wspace=0.24,
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.06,
    )
    t = np.arange(N_YEARS + 1)
    for col, mode in enumerate(("random", "inherited")):
        title = "Random births" if mode == "random" else "Inherited births"
        snap = snapshots[mode]
        ax3 = fig.add_subplot(gs[0, col], projection="3d")
        scatter_merit(ax3, snap["final_traits"], snap["final_council"], n)
        ax3.set_title(f"{title} — year {N_YEARS}", pad=2)

        stats = bundle[mode]
        ax = fig.add_subplot(gs[1, col])
        for path in stats["blue"]:
            ax.plot(
                t,
                path,
                color=C.C_BLUE if path[-1] >= 50 else C.C_PINK,
                lw=0.9,
                alpha=0.3,
            )
        ax.plot(t, stats["blue"].mean(axis=0), color=C.C_GREY, lw=1.8)
        ax.axhline(50, color=C.C_GREY, ls="--", lw=1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Year")
        ax.set_ylabel("% council blue")
        ax.set_title(
            f"Council colour: end |blue−50| {stats['end_abs']:.1f} pp"
        )
        ax.grid(alpha=0.25)

        ax = fig.add_subplot(gs[2, col])
        for path in stats["gap"]:
            ax.plot(t, path, color=C.C_GREY, lw=0.8, alpha=0.25)
        ax.plot(t, stats["gap"].mean(axis=0), color=C.C_GREY, lw=1.8)
        ax.set_xlabel("Year")
        ax.set_ylabel("Council mean merit − city mean merit")
        ax.set_title(
            f"Merit advantage: {stats['end_gap']:.1f} points at year "
            f"{N_YEARS}"
        )
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.savefig(out, dpi=145)
    plt.close(fig)
    print(f"  plot → {out}")


def animate_pair(frames: dict, n: int, out: Path) -> None:
    step_size = max(1, len(frames["random"]) // 60)
    frame_ids = list(range(0, len(frames["random"]), step_size))
    if frame_ids[-1] != len(frames["random"]) - 1:
        frame_ids.append(len(frames["random"]) - 1)

    fig = plt.figure(figsize=(13.0, 9.0), facecolor=C.DARK_BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.2)
    axes3d = [
        fig.add_subplot(gs[0, 0], projection="3d"),
        fig.add_subplot(gs[1, 0], projection="3d"),
    ]
    axes_path = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    C.style_dark_figure(fig, axes3d + axes_path)
    modes = ("random", "inherited")
    paths = {}
    fig.suptitle(
        f"Scene 4 — top merit wins  ·  N={n}  ·  all {N_TRAITS} traits evolve",
        color=C.DARK_TEXT,
        fontsize=14,
        fontweight="bold",
    )

    for row, mode in enumerate(modes):
        traits, council = frames[mode][0]
        scatter_merit(axes3d[row], traits, council, n, dark=True)
        axes3d[row].set_title(
            "Random births — year 0"
            if mode == "random"
            else "Inherited births — year 0"
        )
        paths[mode] = np.asarray(
            [
                100.0 * (traits_i[council_i, COLOUR] > 0).mean()
                for traits_i, council_i in frames[mode]
            ]
        )
        ax = axes_path[row]
        line, = ax.plot([], [], color=C.C_BLUE_BRIGHT, lw=2.6)
        ax._scene4_line = line
        ax.axhline(50, color=C.DARK_MUTED, ls="--", alpha=0.65)
        ax.set_xlim(0, N_YEARS)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Year")
        ax.set_ylabel("% council blue")

    def update(frame_index):
        year = frame_ids[frame_index]
        for row, mode in enumerate(modes):
            ax3 = axes3d[row]
            ax3.cla()
            traits, council = frames[mode][year]
            scatter_merit(ax3, traits, council, n, dark=True)
            label = "Random births" if mode == "random" else "Inherited births"
            ax3.set_title(f"{label} — year {year}")
            axes_path[row]._scene4_line.set_data(
                np.arange(year + 1), paths[mode][: year + 1]
            )
        return tuple(ax._scene4_line for ax in axes_path)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=120,
        blit=False,
    )
    C.save_mp4(anim, out, fps=10)
    plt.close(fig)


def main() -> None:
    args = C.parse_cli("Scene 4 — top merit wins")
    print("Scene 4 — 10 social traits + merit; top merit wins")
    print("  merit affects friendships, is learned, and is inherited/mutated")
    for size in C.sizes_to_run(args.size):
        cfg = C.SIZES[size]
        n, seats, k = cfg["n"], cfg["council"], cfg["k_friends"]
        n_runs = N_RUNS_SMALL if size == "small" else N_RUNS_LARGE
        print(
            f"\n[{size}] N={n}  seats={seats}  friends={k}  "
            f"years={N_YEARS}  runs/mode={n_runs}"
        )
        bundle = {}
        snapshots = {}
        animation_frames = {}
        for mode_index, mode in enumerate(("random", "inherited")):
            runs = []
            seed_base = args.seed + 2000 * mode_index
            for run_index in range(n_runs):
                result = simulate_run(
                    n,
                    seats,
                    k,
                    mode,
                    N_YEARS,
                    np.random.default_rng(seed_base + run_index),
                )
                runs.append(result)
            stats = summarize(runs)
            bundle[mode] = stats
            snapshots[mode] = runs[0]
            print(
                f"  {mode}: end |blue−50|={stats['end_abs']:.1f} pp  "
                f"blue-heavy={100*stats['blue_heavy']:.0f}%  "
                f"pink-heavy={100*stats['pink_heavy']:.0f}%  "
                f"merit advantage={stats['end_gap']:.1f}"
            )
            if args.animate:
                representative = simulate_run(
                    n,
                    seats,
                    k,
                    mode,
                    N_YEARS,
                    np.random.default_rng(seed_base),
                    record_frames=True,
                )
                animation_frames[mode] = representative["frames"]

        plot_stats(
            bundle,
            snapshots,
            n,
            seats,
            k,
            HERE / f"scene4_n{n}.png",
        )
        if args.animate:
            animate_pair(animation_frames, n, HERE / f"scene4_n{n}.mp4")


if __name__ == "__main__":
    main()
