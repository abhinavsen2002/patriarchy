# -*- coding: utf-8 -*-
"""
Scene 5 — Break the echo chambers
=================================
Scene 4's objective test remains: every year the highest-merit people take
the council seats. Scene 5 changes only the social network that shapes the
traits, including merit.

Each person has an open-mindedness score. It determines how many of their
friends are sampled randomly instead of selected for similarity. Each also
has a stickiness score. When stickiness is enabled, an old friend can remain
for several years after falling outside the person's nearest neighbours.

The experiment sweeps random mixing and maximum friendship persistence
separately. This lets us ask whether random cross-cutting ties are enough,
and whether stickiness adds anything after those ties exist.

Run:
  python scene5.py
  python scene5.py --size small
  python scene5.py --size large --no-animate
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scene4"))
import common as C  # noqa: E402
import scene4 as S4  # noqa: E402

HERE = Path(__file__).resolve().parent
N_YEARS = 500
LATE_FROM = 250
N_RUNS_SMALL = 30
N_RUNS_LARGE = 8
RANDOM_MIX_LEVELS = (0.0, 0.1, 0.25, 0.5, 1.0)
STICKY_YEAR_LEVELS = (0, 2, 5, 10)
PRIMARY_MODE = "random"
ANIMATION_YEARS = S4.N_YEARS
ANIMATION_SEED = {100: 67}
ANIMATION_RANDOM_MIX = 1.0


def ranked_nearest(traits: np.ndarray, k: int) -> np.ndarray:
    """The k nearest people in all eleven trait dimensions, nearest first."""
    norm = np.einsum("ij,ij->i", traits, traits)
    d2 = norm[:, None] + norm[None, :] - 2.0 * traits @ traits.T
    np.fill_diagonal(d2, np.inf)
    k = min(k, len(traits) - 1)
    selected = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    row = np.arange(len(traits))[:, None]
    order = np.argsort(d2[row, selected], axis=1)
    return selected[row, order]


def desired_network(
    traits: np.ndarray,
    nearest: np.ndarray,
    openness: np.ndarray,
    max_random_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    n, k = nearest.shape
    n_random = np.rint(k * max_random_fraction * openness).astype(int)
    n_random = np.clip(n_random, 0, k)
    n_near = k - n_random

    # Random scores choose the open-minded slots. Guaranteed nearest friends
    # receive negative scores, so they always survive the k-smallest selection.
    priority = rng.random((n, n))
    np.fill_diagonal(priority, np.inf)
    rows = np.arange(n)[:, None]
    row_grid = np.broadcast_to(rows, (n, k))
    rank = np.arange(k)[None, :]
    guaranteed = rank < n_near[:, None]
    priority[row_grid[guaranteed], nearest[guaranteed]] = (
        -1.0 - (k - rank.repeat(n, axis=0)[guaranteed])
    )
    friends = np.argpartition(priority, kth=k - 1, axis=1)[:, :k]
    return friends, float(n_random.sum() / (n * k))


def apply_stickiness(
    desired: np.ndarray,
    old_friends: np.ndarray | None,
    old_ages: np.ndarray | None,
    stickiness: np.ndarray,
    max_sticky_years: int,
    invalid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep obsolete old ties until each person's persistence timer expires."""
    n, k = desired.shape
    if old_friends is None or max_sticky_years <= 0:
        return desired.copy(), np.zeros((n, k), dtype=np.int16), 0.0

    invalid = np.zeros(n, dtype=bool) if invalid is None else invalid
    rows = np.arange(n)[:, None]
    desired_mask = np.zeros((n, n), dtype=bool)
    desired_mask[rows, desired] = True
    still_desired = desired_mask[rows, old_friends]
    new_age = np.where(still_desired, 0, old_ages + 1).astype(np.int16)
    ttl = np.rint(max_sticky_years * stickiness).astype(np.int16)[:, None]
    valid = (
        ~invalid[:, None]
        & ~invalid[old_friends]
        & (old_friends != rows)
    )
    retain = valid & (still_desired | (new_age <= ttl))

    # Retained ties rank ahead of proposed ties. A dense priority matrix makes
    # the variable number of retained friends selectable without Python loops.
    priority = np.full((n, n), np.inf, dtype=np.float32)
    desired_rank = np.broadcast_to(
        (k + np.arange(k, dtype=np.float32))[None, :], (n, k)
    )
    np.minimum.at(priority, (rows, desired), desired_rank)
    old_rank = np.broadcast_to(
        np.arange(k, dtype=np.float32)[None, :], (n, k)
    )
    retain_rows = np.broadcast_to(rows, (n, k))[retain]
    np.minimum.at(
        priority,
        (retain_rows, old_friends[retain]),
        old_rank[retain],
    )
    friends = np.argpartition(priority, kth=k - 1, axis=1)[:, :k]

    age_lookup = np.zeros((n, n), dtype=np.int16)
    age_lookup[retain_rows, old_friends[retain]] = new_age[retain]
    ages = age_lookup[rows, friends]
    sticky_extra = float((retain & ~still_desired).sum() / (n * k))
    return friends, ages, sticky_extra


def build_network(
    traits: np.ndarray,
    openness: np.ndarray,
    stickiness: np.ndarray,
    k: int,
    max_random_fraction: float,
    max_sticky_years: int,
    rng: np.random.Generator,
    old_friends: np.ndarray | None = None,
    old_ages: np.ndarray | None = None,
    invalid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    nearest = ranked_nearest(traits, k)
    desired, proposed_random = desired_network(
        traits, nearest, openness, max_random_fraction, rng
    )
    friends, ages, sticky_extra = apply_stickiness(
        desired,
        old_friends,
        old_ages,
        stickiness,
        max_sticky_years,
        invalid,
    )
    return friends, ages, proposed_random, sticky_extra


def replace_people(
    traits: np.ndarray,
    openness: np.ndarray,
    stickiness: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(traits)
    n_die = max(1, int(round(S4.DEATH_FRAC * n)))
    die = rng.choice(n, size=n_die, replace=False)
    if mode == "random":
        traits[die] = S4.random_traits(n_die, rng)
        openness[die] = rng.random(n_die)
        stickiness[die] = rng.random(n_die)
    elif mode == "inherited":
        parents = rng.integers(0, n, size=(n_die, 2))
        midpoint = traits[parents].mean(axis=1)
        traits[die] = C.clip_traits(
            midpoint + rng.normal(0.0, S4.MUTATION, size=midpoint.shape)
        )
        openness[die] = np.clip(
            openness[parents].mean(axis=1) + rng.normal(0.0, 0.08, n_die),
            0,
            1,
        )
        stickiness[die] = np.clip(
            stickiness[parents].mean(axis=1) + rng.normal(0.0, 0.08, n_die),
            0,
            1,
        )
    else:
        raise ValueError(mode)
    invalid = np.zeros(n, dtype=bool)
    invalid[die] = True
    return invalid


def network_metrics(traits: np.ndarray, friends: np.ndarray) -> tuple[float, float]:
    blue = traits[:, S4.COLOUR] > 0
    same = 100 * (blue[friends] == blue[:, None]).mean()
    p = blue.mean()
    random_baseline = 100 * (p * p + (1 - p) ** 2)
    return float(same), float(same - random_baseline)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def simulate_run(
    n: int,
    seats: int,
    k: int,
    mode: str,
    max_random_fraction: float,
    max_sticky_years: int,
    seed: int,
    record_frames: bool = False,
    years: int | None = None,
    full_random_openness: bool = False,
) -> dict:
    years = N_YEARS if years is None else years
    rng = np.random.default_rng(seed)
    traits = S4.random_traits(n, rng)
    openness = np.ones(n) if full_random_openness else rng.random(n)
    stickiness = rng.random(n)
    council = S4.elect_by_merit(traits, seats)
    friends = ages = None
    invalid = None
    friends, ages, proposed_random, sticky_extra = build_network(
        traits,
        openness,
        stickiness,
        k,
        max_random_fraction,
        max_sticky_years,
        rng,
        friends,
        ages,
        invalid,
    )

    blue_path = [100 * (traits[council, S4.COLOUR] > 0).mean()]
    city_blue_path = [100 * (traits[:, S4.COLOUR] > 0).mean()]
    turnover_path = [0.0]
    random_path = [proposed_random]
    sticky_path = [sticky_extra]
    frames = (
        [(traits.copy(), council.copy(), friends.copy())]
        if record_frames else None
    )

    for year in range(1, years + 1):
        friends, ages, proposed_random, sticky_extra = build_network(
            traits,
            openness,
            stickiness,
            k,
            max_random_fraction,
            max_sticky_years,
            rng,
            friends,
            ages,
            invalid,
        )
        traits = C.learn_from_friends(
            traits,
            friends,
            rng,
            S4.INFLUENCE,
            S4.NOISE,
        )
        old_council = council
        invalid = replace_people(traits, openness, stickiness, mode, rng)
        if year % S4.RESCALE_EVERY == 0:
            traits = C.rescale_traits(traits)
        council = S4.elect_by_merit(traits, seats)
        S4.assert_merit_council(traits, council, seats)
        blue_path.append(100 * (traits[council, S4.COLOUR] > 0).mean())
        city_blue_path.append(100 * (traits[:, S4.COLOUR] > 0).mean())
        turnover_path.append(
            1.0 - len(np.intersect1d(old_council, council)) / seats
        )
        random_path.append(proposed_random)
        sticky_path.append(sticky_extra)
        if record_frames:
            frames.append((traits.copy(), council.copy(), friends.copy()))

    # Rebuild once so final network reflects newborn identities.
    friends, _, proposed_random, sticky_extra = build_network(
        traits,
        openness,
        stickiness,
        k,
        max_random_fraction,
        max_sticky_years,
        rng,
        friends,
        ages,
        invalid,
    )
    same_colour, echo_excess = network_metrics(traits, friends)
    blue_path_array = np.asarray(blue_path)
    city_blue_array = np.asarray(city_blue_path)
    late_from = min(LATE_FROM, years)
    return {
        "blue": blue_path_array,
        "city_blue": city_blue_array,
        "turnover": np.asarray(turnover_path),
        "actual_random_fraction": float(np.mean(random_path)),
        "actual_sticky_extra_fraction": float(np.mean(sticky_path)),
        "same_colour_friend_pct": same_colour,
        "echo_chamber_excess_pp": echo_excess,
        "abs_colour_merit_correlation": abs(
            correlation(traits[:, S4.COLOUR], traits[:, S4.MERIT])
        ),
        "late_abs_council_imbalance_pp": float(
            np.abs(blue_path_array[late_from:] - 50).mean()
        ),
        "late_abs_representation_gap_pp": float(
            np.abs(
                blue_path_array[late_from:] - city_blue_array[late_from:]
            ).mean()
        ),
        "late_abs_city_imbalance_pp": float(
            np.abs(city_blue_array[late_from:] - 50).mean()
        ),
        "late_council_turnover_pct": float(
            100 * np.mean(turnover_path[late_from:])
        ),
        "frames": frames,
    }


def mean_ci(values, seed=991) -> dict:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(3000, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean": round(float(x.mean()), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
    }


def summarize(runs: list[dict]) -> dict:
    keys = (
        "late_abs_representation_gap_pp",
        "late_abs_council_imbalance_pp",
        "late_abs_city_imbalance_pp",
        "same_colour_friend_pct",
        "echo_chamber_excess_pp",
        "abs_colour_merit_correlation",
        "late_council_turnover_pct",
        "actual_random_fraction",
        "actual_sticky_extra_fraction",
    )
    result = {key: mean_ci([run[key] for run in runs]) for key in keys}
    result["n_runs"] = len(runs)
    result["blue_paths"] = [run["blue"].round(2).tolist() for run in runs]
    result["city_blue_paths"] = [
        run["city_blue"].round(2).tolist() for run in runs
    ]
    return result


def run_task(args: tuple) -> dict:
    return simulate_run(*args)


def paired_stickiness_effects(cells: dict) -> dict:
    """Paired change in representation gap versus no stickiness.

    Runs use identical seeds within each random-mixing level. Negative values
    mean stickiness improved representation.
    """
    rng = np.random.default_rng(771)
    effects = {}
    for random_mix in RANDOM_MIX_LEVELS:
        baseline = cells[f"random={random_mix:g},sticky=0"]
        base_blue = np.asarray(baseline["blue_paths"])[:, LATE_FROM:]
        base_city = np.asarray(baseline["city_blue_paths"])[:, LATE_FROM:]
        base_gap = np.abs(base_blue - base_city).mean(axis=1)
        for sticky_years in STICKY_YEAR_LEVELS[1:]:
            cell = cells[
                f"random={random_mix:g},sticky={sticky_years}"
            ]
            blue = np.asarray(cell["blue_paths"])[:, LATE_FROM:]
            city = np.asarray(cell["city_blue_paths"])[:, LATE_FROM:]
            change = np.abs(blue - city).mean(axis=1) - base_gap
            boot = rng.choice(
                change, size=(4000, len(change)), replace=True
            ).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            effects[
                f"random={random_mix:g},sticky={sticky_years}"
            ] = {
                "mean_change_pp": round(float(change.mean()), 3),
                "ci95": [round(float(lo), 3), round(float(hi), 3)],
                "negative_means_stickiness_helped": True,
            }
    return effects


def run_sweep(size: str, seed: int) -> dict:
    cfg = C.SIZES[size]
    n_runs = N_RUNS_SMALL if size == "small" else N_RUNS_LARGE
    cell_tasks = []
    for sticky_years in STICKY_YEAR_LEVELS:
        for random_mix in RANDOM_MIX_LEVELS:
            key = f"random={random_mix:g},sticky={sticky_years}"
            print(f"  {key}")
            for run in range(n_runs):
                cell_tasks.append(
                    (
                        key,
                        (
                            cfg["n"],
                            cfg["council"],
                            cfg["k_friends"],
                            PRIMARY_MODE,
                            random_mix,
                            sticky_years,
                            seed + run,
                        ),
                    )
                )

    sensitivity_tasks = []
    for random_mix in (0.0, 0.5):
        key = f"inherited_births,random={random_mix:g},sticky=0"
        for run in range(n_runs):
            sensitivity_tasks.append(
                (
                    key,
                    (
                        cfg["n"],
                        cfg["council"],
                        cfg["k_friends"],
                        "inherited",
                        random_mix,
                        0,
                        seed + 5000 + run,
                    ),
                )
            )

    all_tasks = cell_tasks + sensitivity_tasks
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(run_task, [task for _, task in all_tasks]))

    grouped: dict[str, list[dict]] = {}
    for (key, _), output in zip(all_tasks, outputs):
        grouped.setdefault(key, []).append(output)

    cells = {}
    for sticky_years in STICKY_YEAR_LEVELS:
        for random_mix in RANDOM_MIX_LEVELS:
            key = f"random={random_mix:g},sticky={sticky_years}"
            cells[key] = summarize(grouped[key])

    sensitivity = {}
    for random_mix in (0.0, 0.5):
        key = f"inherited_births,random={random_mix:g},sticky=0"
        sensitivity[key] = summarize(grouped[key])
    return {
        "size": size,
        "n": cfg["n"],
        "seats": cfg["council"],
        "friends": cfg["k_friends"],
        "birth_mode": PRIMARY_MODE,
        "cells": cells,
        "paired_stickiness_effects": paired_stickiness_effects(cells),
        "inherited_birth_sensitivity": sensitivity,
    }


def matrix(results: dict, metric: str) -> np.ndarray:
    return np.asarray(
        [
            [
                results["cells"][
                    f"random={random_mix:g},sticky={sticky_years}"
                ][metric]["mean"]
                for random_mix in RANDOM_MIX_LEVELS
            ]
            for sticky_years in STICKY_YEAR_LEVELS
        ]
    )


def heatmap(ax, values, title, unit, cmap="viridis_r"):
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(RANDOM_MIX_LEVELS)))
    ax.set_xticklabels([f"{50*x:.0f}%" for x in RANDOM_MIX_LEVELS])
    ax.set_yticks(range(len(STICKY_YEAR_LEVELS)))
    ax.set_yticklabels(STICKY_YEAR_LEVELS)
    ax.set_xlabel("Mean random share of friends")
    ax.set_ylabel("Maximum sticky years")
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                f"{values[row, col]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if values[row, col] > np.nanmedian(values) else "black",
            )
    cb = plt.colorbar(image, ax=ax, shrink=0.82)
    cb.set_label(unit)


def plot_sweep(results: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), facecolor="white")
    fig.suptitle(
        f"Scene 5 — breaking echo chambers  ·  N={results['n']}, "
        f"{results['seats']} merit-ranked seats\n"
        "Random births · open-minded random ties × sticky friendship duration",
        fontsize=13,
        fontweight="bold",
    )
    heatmap(
        axes[0, 0],
        matrix(results, "late_abs_representation_gap_pp"),
        f"Representational inequality, years {LATE_FROM}–{N_YEARS}",
        "Mean |council blue − city blue| (pp)",
    )
    heatmap(
        axes[0, 1],
        matrix(results, "echo_chamber_excess_pp"),
        f"Echo-chamber strength at year {N_YEARS}",
        "Same-colour friendship excess (pp)",
    )
    heatmap(
        axes[1, 0],
        matrix(results, "abs_colour_merit_correlation"),
        "Emergent colour–merit association",
        "Mean absolute Pearson correlation",
    )
    heatmap(
        axes[1, 1],
        matrix(results, "late_abs_city_imbalance_pp"),
        f"Population diversity, years {LATE_FROM}–{N_YEARS}",
        "Mean |city blue share − 50| (pp)",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  plot → {out}")


def packed_friend_edges(
    friends: np.ndarray, n: int, keep_frac: float = 0.40
) -> np.ndarray:
    k = friends.shape[1]
    src = np.repeat(np.arange(n), k)
    dst = friends.ravel()
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    packed = np.unique(lo.astype(np.int64) * n + hi)
    keep = ((packed * 2654435761) % (2 ** 32)) / (2 ** 32) < keep_frac
    return packed[keep]


def friend_edge_look(xy: np.ndarray, packed: np.ndarray, n: int):
    if len(packed) == 0:
        return (
            np.empty((0, 2, 2)),
            np.empty((0, 4)),
            np.empty((0,)),
        )
    a = packed // n
    b = packed % n
    segs = np.stack([xy[a], xy[b]], axis=1)
    dist = np.linalg.norm(xy[a] - xy[b], axis=1)
    scale = max(float(np.percentile(dist, 85)) if len(dist) else 1.0, 1.0)
    closeness = np.clip(1.0 - dist / scale, 0.0, 1.0)
    noise = 0.55 + 0.90 * (((packed * 1103515245 + 12345) % (2 ** 31)) / (2 ** 31))
    brightness = np.clip(0.04 + 0.42 * closeness * noise, 0.03, 0.58)
    colours = np.tile(np.asarray(to_rgba(C.C_GREY)), (len(packed), 1))
    colours[:, 3] = brightness
    widths = 0.30 + 1.15 * closeness
    return segs, colours, widths


def _draw_world(
    ax, xy: np.ndarray, council: np.ndarray, friends: np.ndarray, n: int
) -> None:
    ax.cla()
    plane = xy[:, [S4.COLOUR, S4.ANIMAL]]
    packed = packed_friend_edges(friends, n)
    segs, colours, widths = friend_edge_look(plane, packed, n)
    if len(segs):
        ax.add_collection(
            LineCollection(segs, colors=colours, linewidths=widths, zorder=1)
        )
    blue = xy[:, S4.COLOUR] > 0
    size = 30 if n <= 100 else 7
    ax.scatter(
        xy[~blue, S4.COLOUR], xy[~blue, S4.ANIMAL],
        s=size, c=C.C_PINK_BRIGHT, alpha=0.95, linewidths=0, zorder=2,
    )
    ax.scatter(
        xy[blue, S4.COLOUR], xy[blue, S4.ANIMAL],
        s=size, c=C.C_BLUE_BRIGHT, alpha=0.95, linewidths=0, zorder=2,
    )
    ax.scatter(
        xy[council, S4.COLOUR], xy[council, S4.ANIMAL],
        s=size + 28, facecolors="none", edgecolors=C.C_COUNCIL,
        linewidths=1.4, zorder=3,
    )
    ax.axvline(0, color=C.DARK_GRID, lw=0.8, alpha=0.7)
    ax.axhline(0, color=C.DARK_GRID, lw=0.8, alpha=0.7)
    ax.set_xlim(-105, 105)
    ax.set_ylim(-105, 105)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    label = dict(color=C.DARK_TEXT, fontsize=12, fontweight="bold")
    ax.text(-0.02, 0.5, "Pink", transform=ax.transAxes, ha="right", va="center", **label)
    ax.text(1.02, 0.5, "Blue", transform=ax.transAxes, ha="left", va="center", **label)
    ax.text(0.5, -0.02, "Cat", transform=ax.transAxes, ha="center", va="top", **label)
    ax.text(0.5, 1.01, "Dog", transform=ax.transAxes, ha="center", va="bottom", **label)


def animate_random_friends(n: int, seats: int, k: int, seed: int, out: Path) -> None:
    run = simulate_run(
        n,
        seats,
        k,
        "inherited",
        ANIMATION_RANDOM_MIX,
        0,
        seed,
        record_frames=True,
        years=ANIMATION_YEARS,
        full_random_openness=True,
    )
    years = ANIMATION_YEARS
    sub = 6
    fig = plt.figure(figsize=(16, 9), facecolor="#000000")
    ax = fig.add_axes([0.255, 0.14, 0.49, 0.72])
    ax.set_facecolor("#000000")
    year_text = fig.text(
        0.04, 0.92, "Year 0", color=C.DARK_TEXT, fontsize=25,
        fontweight="bold", ha="left",
    )
    positions = np.stack([frame[0] for frame in run["frames"]])
    ambient = C.make_ambient(n, amp=0.22, seed=41, period=168.0)

    def update(f):
        seg = min(f / sub, years)
        t0 = int(np.floor(seg))
        t1 = min(t0 + 1, years)
        u = seg - t0
        ease = u * u * (3.0 - 2.0 * u)
        traits = positions[t0] * (1.0 - ease) + positions[t1] * ease
        traits = traits.copy()
        traits[:, [S4.COLOUR, S4.ANIMAL]] += ambient(f)
        frame = run["frames"][t1 if ease >= 0.5 else t0]
        _draw_world(ax, traits, frame[1], frame[2], n)
        year_text.set_text(f"Year {int(round(seg))}")
        return ()

    animation = FuncAnimation(
        fig, update, frames=years * sub + 1, interval=40, blit=False,
    )
    C.save_mp4(animation, out, fps=24, bg="#000000")
    plt.close(fig)


def main() -> None:
    args = C.parse_cli("Scene 5 — break echo chambers")
    all_results = {
        "metadata": {
            "seed": args.seed,
            "years": N_YEARS,
            "late_window": [LATE_FROM, N_YEARS],
            "random_mix_levels": RANDOM_MIX_LEVELS,
            "sticky_year_levels": STICKY_YEAR_LEVELS,
            "note": (
                "Mean random friend share is approximately half the configured "
                "maximum because individual openness is initially Uniform(0,1)."
            ),
        },
        "sizes": {},
    }
    for size in C.sizes_to_run(args.size):
        print(f"\n[{size}] Scene 5 sweep")
        results = run_sweep(size, args.seed)
        all_results["sizes"][size] = results
        plot_sweep(results, HERE / f"scene5_n{results['n']}.png")
        if args.animate:
            animation_seed = ANIMATION_SEED.get(results["n"], args.seed)
            print(f"  animation seed={animation_seed}")
            animate_random_friends(
                results["n"],
                results["seats"],
                C.SIZES[size]["k_friends"],
                animation_seed,
                HERE / f"scene5_n{results['n']}.mp4",
            )
    out = HERE / "scene5_results.json"
    out.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"  results → {out}")


if __name__ == "__main__":
    main()
