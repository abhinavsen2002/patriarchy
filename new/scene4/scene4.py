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
toward friends through learning, and is inherited at birth with mutation,
just like every other trait.

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
from mpl_toolkits.mplot3d import proj3d

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
# Seed of the representative animated world, chosen so blue wins by the end.
ANIM_SEED = {100: 2059}


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


def replace(traits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(traits)
    n_die = max(1, int(round(DEATH_FRAC * n)))
    die = rng.choice(n, size=n_die, replace=False)
    parents = rng.integers(0, n, size=(n_die, 2))
    midpoint = traits[parents].mean(axis=1)
    newborns = midpoint + rng.normal(0.0, MUTATION, size=midpoint.shape)
    traits[die] = C.clip_traits(newborns)
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
    seats: int,
    k: int,
    rng: np.random.Generator,
    rescale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    friends = knn_friends(traits, k)
    traits = influence(traits, friends, rng)
    traits = replace(traits, rng)
    if rescale:
        traits = C.rescale_traits(traits)
    council = elect_by_merit(traits, seats)
    assert_merit_council(traits, council, seats)
    return traits, council


def simulate_run(
    n: int,
    seats: int,
    k: int,
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
    stats: dict,
    snapshot: dict,
    n: int,
    seats: int,
    k: int,
    out: Path,
) -> None:
    fig = plt.figure(figsize=(14.2, 6.2), facecolor="white")
    fig.suptitle(
        f"Scene 4 — Merit wins  ·  N={n}, {seats} seats, {k} friends, "
        f"{N_TRAITS} traits\n"
        "inherited births; friendship + learning shape merit; elections take "
        "the top merit ranks only",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    gs = GridSpec(
        1,
        2,
        figure=fig,
        wspace=0.28,
        left=0.07,
        right=0.97,
        top=0.86,
        bottom=0.12,
    )
    t = np.arange(N_YEARS + 1)

    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    scatter_merit(ax3, snapshot["final_traits"], snapshot["final_council"], n)
    ax3.set_title(f"Inherited births — year {N_YEARS}", pad=2)

    ax = fig.add_subplot(gs[0, 1])
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
    ax.set_title(f"Council colour: end |blue−50| {stats['end_abs']:.1f} pp")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.savefig(out, dpi=145)
    plt.close(fig)
    print(f"  plot → {out}")


def _style_axis3d_grid(ax) -> None:
    """Hide the matplotlib cube; we draw a three-face grid ourselves."""
    ax.set_facecolor("#000000")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0, 0, 0, 0.0))
        axis.pane.set_edgecolor((0, 0, 0, 0.0))
        axis.line.set_color((0, 0, 0, 0))
        axis.line.set_linewidth(0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


def _edge_label_angle(ax, p0, p1) -> float:
    """Screen-space angle of a 3-D edge, flipped so the word stays upright."""
    x0, y0, _ = proj3d.proj_transform(*p0, ax.get_proj())
    x1, y1, _ = proj3d.proj_transform(*p1, ax.get_proj())
    angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    return float(angle)


def _label_on_edge(ax, p0, p1, offset, text, **label) -> None:
    mid = np.asarray(p0, dtype=float) * 0.5 + np.asarray(p1, dtype=float) * 0.5
    mid = mid + np.asarray(offset, dtype=float)
    x2, y2, _ = proj3d.proj_transform(*mid, ax.get_proj())
    ax.text2D(
        x2, y2, text, transform=ax.transData, ha="center", va="center",
        rotation=_edge_label_angle(ax, p0, p1), rotation_mode="anchor",
        **label,
    )


def _draw_corner_grid(ax) -> None:
    """Grid on three meeting faces at the far corner of the view."""
    e = 100.0
    ticks = np.arange(-100.0, 100.01, 25.0)
    rgb = tuple(int(C.DARK_GRID[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    kw = dict(color=rgb, lw=0.45, alpha=0.32, zorder=0)
    # Floor (z = -e), pink wall (x = -e), dog wall (y = +e) — they meet at the back.
    for t in ticks:
        ax.plot([-e, e], [t, t], [-e, -e], **kw)
        ax.plot([t, t], [-e, e], [-e, -e], **kw)
        ax.plot([-e, e], [e, e], [t, t], **kw)
        ax.plot([t, t], [e, e], [-e, e], **kw)
        ax.plot([-e, -e], [-e, e], [t, t], **kw)
        ax.plot([-e, -e], [t, t], [-e, e], **kw)
    edge = dict(color=C.DARK_GRID, lw=1.15, alpha=0.55, zorder=1)
    ax.plot([-e, e], [e, e], [-e, -e], **edge)
    ax.plot([-e, -e], [-e, e], [-e, -e], **edge)
    ax.plot([-e, -e], [e, e], [-e, e], **edge)
    label = dict(
        color=C.DARK_MUTED, fontsize=12, fontweight="normal", alpha=0.72,
    )
    colour_a, colour_b = (-e, e, -e), (e, e, -e)
    animal_a, animal_b = (-e, -e, -e), (-e, e, -e)
    _label_on_edge(ax, colour_a, colour_b, (0.0, 10.0, 0.0), "colour", **label)
    _label_on_edge(ax, animal_a, animal_b, (-10.0, 0.0, 0.0), "animal", **label)
    ax.text(-e, e, e + 14, "merit", ha="center", va="bottom", **label)


def _draw_frame3d(ax, xyz: np.ndarray, council: np.ndarray, n: int,
                  azim: float, elev: float) -> None:
    ax.cla()
    lim = 118
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    _style_axis3d_grid(ax)
    _draw_corner_grid(ax)
    blue = xyz[:, 0] > 0
    size = 24 if n <= 100 else 6
    pink = C.C_PINK_BRIGHT
    blue_c = C.C_BLUE_BRIGHT
    ax.scatter(
        xyz[~blue, 0], xyz[~blue, 1], xyz[~blue, 2],
        s=size, c=pink, alpha=0.95, linewidths=0, depthshade=False,
    )
    ax.scatter(
        xyz[blue, 0], xyz[blue, 1], xyz[blue, 2],
        s=size, c=blue_c, alpha=0.95, linewidths=0, depthshade=False,
    )
    ax.scatter(
        xyz[council, 0], xyz[council, 1], xyz[council, 2],
        s=size + 30, marker="o", facecolors="none",
        edgecolors=C.C_COUNCIL, linewidths=1.4, depthshade=False,
    )


def _style_mini_chart(ax) -> None:
    ax.set_facecolor("#000000")
    for spine in ax.spines.values():
        spine.set_color(C.DARK_GRID)
    ax.tick_params(colors=C.DARK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(C.DARK_MUTED)
    ax.yaxis.label.set_color(C.DARK_MUTED)


def animate_run(
    frames: list, n: int, out: Path, view_mode: str = "orbit",
) -> None:
    """3-D world; optionally settle into Colour × Merit around year 65."""
    BLACK = "#000000"
    dims = [COLOUR, ANIMAL, MERIT]
    positions = np.stack([t[:, dims] for t, _ in frames])
    councils = [c for _, c in frames]
    seats = len(councils[0])
    blue_counts = np.asarray(
        [(traits[council, COLOUR] > 0).sum() for traits, council in frames],
        dtype=float,
    )
    n_years = len(frames) - 1
    sub = 5

    fig = plt.figure(figsize=(16, 9), facecolor=BLACK)
    ax = fig.add_axes([0.00, 0.08, 0.54, 0.84], projection="3d")
    axp = fig.add_axes([0.61, 0.34, 0.24, 0.32])
    ax.set_facecolor(BLACK)
    _style_mini_chart(axp)

    year_text = fig.text(
        0.04, 0.93, "Year 0", color=C.DARK_TEXT, fontsize=26,
        fontweight="bold", ha="left", va="center",
    )
    blue_c = C.C_BLUE_BRIGHT
    line_b, = axp.plot([], [], color=blue_c, lw=3.0)
    dot_b, = axp.plot([], [], "o", color=blue_c, ms=6)
    axp.axhline(seats / 2, color=C.DARK_MUTED, ls="--", lw=0.9, alpha=0.6)
    axp.set_xlim(0, N_YEARS)
    axp.set_ylim(0, seats)
    axp.set_xticks([0, N_YEARS])
    axp.set_yticks([0, seats // 2, seats])
    axp.set_title("blueys in power", color=blue_c, fontsize=15, fontweight="bold")

    def _grow(line, dot, series, t0, seg, ease):
        t1 = min(t0 + 1, n_years)
        now = series[t0] * (1.0 - ease) + series[t1] * ease
        whole = np.arange(t0 + 1)
        line.set_data(np.append(whole, seg), np.append(series[: t0 + 1], now))
        dot.set_data([seg], [now])

    def _lerp_angle(a, b, t):
        delta = (b - a + 180.0) % 360.0 - 180.0
        return a + delta * t

    def _smooth(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    year_frames = n_years * sub
    end_azim = -52.0 + 22.0
    end_elev = 22.0
    tour = []
    if view_mode == "orbit":
        plane_views = (
            (-90.0, 90.0, "Colour × Animal"),
            (-90.0, 0.0, "Colour × Merit"),
            (0.0, 0.0, "Animal × Merit"),
        )
        move_n, hold_n = 30, 42
        az0, el0 = end_azim, end_elev
        for az1, el1, caption in plane_views:
            for i in range(move_n):
                t = _smooth((i + 1) / move_n)
                tour.append(
                    (_lerp_angle(az0, az1, t), el0 + (el1 - el0) * t, caption)
                )
            tour.extend([(az1, el1, caption)] * hold_n)
            az0, el0 = az1, el1

    plane_text = fig.text(
        0.04, 0.87, "", color=C.DARK_MUTED, fontsize=16,
        fontweight="bold", ha="left", va="center",
    )
    ambient = C.make_ambient(
        n, amp=0.22, seed=37, period=168.0, dims=3
    )

    def update(f):
        n_frames = year_frames
        if f <= year_frames:
            seg = min(f / sub, n_years)
            t0 = int(np.floor(seg))
            t1 = min(t0 + 1, n_years)
            u = seg - t0
            ease = u * u * (3.0 - 2.0 * u)
            xyz = positions[t0] * (1.0 - ease) + positions[t1] * ease
            xyz = xyz + ambient(f)
            council = councils[t1 if ease >= 0.5 else t0]
            base_azim = -52.0 + 22.0 * (f / max(n_frames, 1))
            base_elev = 22.0 + 2.0 * np.sin(np.pi * f / max(n_frames, 1))
            if view_mode == "colour_merit" and seg >= 58:
                turn = _smooth((seg - 58.0) / 12.0)
                start_azim = -52.0 + 22.0 * (58.0 / n_years)
                start_elev = 22.0 + 2.0 * np.sin(np.pi * 58.0 / n_years)
                azim = _lerp_angle(start_azim, -90.0, turn)
                elev = start_elev * (1.0 - turn)
                plane_text.set_text("Colour × Merit" if seg >= 64 else "")
            else:
                azim, elev = base_azim, base_elev
                plane_text.set_text("")
            _grow(line_b, dot_b, blue_counts, t0, seg, ease)
            year_text.set_text(f"Year {int(round(seg))}")
        else:
            xyz = positions[-1] + ambient(f)
            council = councils[-1]
            azim, elev, caption = tour[f - year_frames - 1]
            _grow(line_b, dot_b, blue_counts, n_years, n_years, 1.0)
            year_text.set_text(f"Year {n_years}")
            plane_text.set_text(caption)
        _draw_frame3d(ax, xyz, council, n, azim, elev)
        return line_b, dot_b, year_text, plane_text

    anim = FuncAnimation(
        fig, update, frames=year_frames + 1 + len(tour), interval=40, blit=False,
    )
    C.save_mp4(anim, out, fps=24, bg=BLACK)
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
            f"years={N_YEARS}  runs={n_runs}"
        )
        seed_base = args.seed + 2000
        runs = []
        for run_index in range(n_runs):
            runs.append(
                simulate_run(
                    n,
                    seats,
                    k,
                    N_YEARS,
                    np.random.default_rng(seed_base + run_index),
                )
            )
        stats = summarize(runs)
        print(
            f"  inherited: end |blue−50|={stats['end_abs']:.1f} pp  "
            f"blue-heavy={100*stats['blue_heavy']:.0f}%  "
            f"pink-heavy={100*stats['pink_heavy']:.0f}%"
        )
        plot_stats(
            stats,
            runs[0],
            n,
            seats,
            k,
            HERE / f"scene4_n{n}.png",
        )
        if args.animate:
            anim_seed = ANIM_SEED.get(n, seed_base)
            representative = simulate_run(
                n,
                seats,
                k,
                N_YEARS,
                np.random.default_rng(anim_seed),
                record_frames=True,
            )
            print(
                f"  animating seed={anim_seed}  "
                f"end {representative['blue'][-1]:.0f}% blue"
            )
            animate_run(representative["frames"], n, HERE / f"scene4_n{n}.mp4")


if __name__ == "__main__":
    main()
