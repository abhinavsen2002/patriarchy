# -*- coding: utf-8 -*-
"""Shared election mechanics for Meritopolis scenes 1–3."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SIZES = {
    "large": {"n": 1000, "council": 200, "k_friends": 200},
    "small": {"n": 100, "council": 20, "k_friends": 20},
}

MALE, FEMALE = 0, 1
XY_CLIP = 100.0
XY_SIGMA = 40.0

C_GREY = "#787882"  # rgb(120, 120, 130)
C_BLUE = "#93B4D2"  # rgb(147, 180, 210)
C_PINK = "#D2AABE"  # rgb(210, 170, 190)
C_MALE = C_BLUE
C_FEMALE = C_PINK
C_DOG = "#2A9D8F"
C_CAT = "#E9C46A"

# Animation palette: designed for a near-black background.
DARK_BG = "#05070A"
DARK_PANEL = "#0B0F14"
DARK_TEXT = "#F4F7FA"
DARK_MUTED = C_GREY
DARK_GRID = C_GREY
C_BLUE_BRIGHT = "#5AA8E8"
C_PINK_BRIGHT = "#F07AAE"
C_COUNCIL = "#FFD166"


def parse_cli(description: str, extra_args=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--size", choices=("small", "large", "both"), default="both")
    p.add_argument("--animate", action="store_true", default=True)
    p.add_argument("--no-animate", action="store_false", dest="animate")
    p.add_argument("--seed", type=int, default=42)
    if extra_args is not None:
        extra_args(p)
    return p.parse_args()


def sizes_to_run(kind: str) -> list[str]:
    return ["small", "large"] if kind == "both" else [kind]


def weighted_sample(items: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Systematic PPS sample of k distinct items.

    Inclusion probability is (approximately) k * w_i / sum(w), so a group
    that holds 60% of the weight keeps ~60% of the seats. Sequential
    without-replacement lotteries pull large councils toward 50/50.
    """
    items = np.asarray(items)
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.0)
    k = min(int(k), len(items))
    if k <= 0:
        return items[:0]
    if w.sum() <= 0:
        return rng.choice(items, size=k, replace=False)
    lam = k * w / w.sum()
    certain = lam >= 1.0 - 1e-9
    chosen = np.flatnonzero(certain)
    k_left = k - len(chosen)
    if k_left <= 0:
        return items[chosen[:k]]
    rest = np.flatnonzero(~certain)
    lam_r = lam[rest]
    lam_r = lam_r * (k_left / lam_r.sum())
    cum = np.cumsum(lam_r)
    u = rng.random()
    targets = u + np.arange(k_left)
    pick = np.searchsorted(cum, targets, side="right")
    pick = np.clip(pick, 0, len(rest) - 1)
    extra = np.unique(rest[pick])
    if len(extra) < k_left:
        pool = np.setdiff1d(rest, extra, assume_unique=False)
        extra = np.concatenate([extra, rng.choice(pool, size=k_left - len(extra), replace=False)])
    out = np.concatenate([chosen, extra])
    return items[out]


def friends_in_power(council: np.ndarray, friends: np.ndarray, n: int) -> np.ndarray:
    on = np.zeros(n, dtype=bool)
    on[np.asarray(council)] = True
    return on[friends].sum(axis=1).astype(float)


def nominations(council: np.ndarray, friends: np.ndarray, n: int) -> np.ndarray:
    """How many sitting members named each person (put them on the slate)."""
    tickets = np.zeros(n, dtype=float)
    np.add.at(tickets, friends[np.asarray(council)].ravel(), 1)
    return tickets


def elect(council: np.ndarray, friends: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Two-step election.

    1. Sitting members name their friends (the slate).
    2. The sitting council chooses among that slate; a nominee's chance
       still scales with how many friends they have in office.

    Named × chosen, through the same network, is the square rule without
    extra machinery.
    """
    n = friends.shape[0]
    seats = len(council)
    tickets = nominations(council, friends, n)
    power = friends_in_power(council, friends, n)
    eligible = np.flatnonzero(tickets > 0)
    if len(eligible) < seats:
        chosen = eligible.copy()
        rest = np.setdiff1d(np.arange(n), chosen, assume_unique=False)
        extra = rng.choice(rest, size=seats - len(chosen), replace=False)
        return np.concatenate([chosen, extra])
    weight = tickets[eligible] * np.maximum(power[eligible], 1.0)
    return weighted_sample(eligible, weight, seats, rng)


def knn_friends(xy: np.ndarray, k: int) -> np.ndarray:
    n = len(xy)
    delta = xy[:, None, :] - xy[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", delta, delta)
    np.fill_diagonal(d2, np.inf)
    k = min(k, n - 1)
    return np.argpartition(d2, k, axis=1)[:, :k]


def random_same_group_friends(group: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(group)
    friends = np.empty((n, k), dtype=int)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        for i in idx:
            pool = idx[idx != i]
            friends[i] = rng.choice(pool, size=k, replace=False)
    return friends


def clip_traits(xy: np.ndarray) -> np.ndarray:
    return np.clip(xy, -XY_CLIP, XY_CLIP)


def learn_from_friends(
    traits: np.ndarray,
    friends: np.ndarray,
    rng: np.random.Generator,
    influence: float = 0.08,
    noise: float = 3.0,
) -> np.ndarray:
    """Copy friends and add noise; no inward or outward correction force."""
    pulled = traits[friends].mean(axis=1)
    learned = traits + influence * (pulled - traits)
    learned = learned + rng.normal(0.0, noise, size=traits.shape)
    return clip_traits(learned)


def rescale_traits(traits: np.ndarray) -> np.ndarray:
    """Linearly map every trait's current city range onto [-100, 100].

    This periodic display/scale correction preserves ordering and relative
    spacing within each trait. A collapsed trait is set to neutral.
    """
    low = traits.min(axis=0, keepdims=True)
    high = traits.max(axis=0, keepdims=True)
    span = high - low
    safe_span = np.where(span > 1e-12, span, 1.0)
    scaled = -XY_CLIP + 2.0 * XY_CLIP * (traits - low) / safe_span
    scaled[:, (span <= 1e-12).ravel()] = 0.0
    return scaled


def make_ambient(
    n: int,
    amp: float,
    seed: int = 99,
    period: float = 168.0,
    freq_jitter: float = 0.22,
    dims: int = 2,
):
    """Return a stable per-dot ambient offset generator.

    Each dot gets a fixed random phase and a slightly detuned frequency, so
    the whole field shimmers with small looping motion. The offsets are always
    zero-mean sinusoids, so dots wobble in place without drifting from their
    base location. ``amp`` is in the same units as the positions being drawn.
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, dims))
    freq = 1.0 + freq_jitter * (rng.random((n, dims)) * 2.0 - 1.0)
    # A mild per-dot amplitude variation keeps the motion from looking uniform.
    amp_scale = 0.7 + 0.6 * rng.random((n, dims))
    two_pi = 2.0 * np.pi

    def offset(frame: float) -> np.ndarray:
        t = frame / period
        return amp * amp_scale * np.sin(two_pi * freq * t + phase)

    return offset


def save_mp4(anim, path: Path, fps: int = 8, bg: str = DARK_BG) -> None:
    path = Path(path)
    writer = __import__("matplotlib.animation", fromlist=["FFMpegWriter"]).FFMpegWriter(
        fps=fps, metadata={"title": path.stem}, bitrate=1800
    )
    anim.save(str(path), writer=writer, savefig_kwargs={"facecolor": bg})
    print(f"  animation → {path}")


def style_dark_axis(ax, grid: bool = True) -> None:
    """Apply the shared animation theme to a 2-D or 3-D axis."""
    ax.set_facecolor(DARK_PANEL)
    ax.tick_params(colors=DARK_MUTED, labelsize=8)
    ax.xaxis.label.set_color(DARK_MUTED)
    ax.yaxis.label.set_color(DARK_MUTED)
    ax.title.set_color(DARK_TEXT)
    for spine in ax.spines.values():
        spine.set_color(DARK_GRID)
    if grid:
        ax.grid(True, color=DARK_GRID, alpha=0.45, linewidth=0.65)
    if hasattr(ax, "zaxis"):
        ax.zaxis.label.set_color(DARK_MUTED)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(DARK_PANEL)
            axis.pane.set_edgecolor(DARK_GRID)
            axis._axinfo["grid"]["color"] = DARK_GRID
            axis._axinfo["grid"]["linewidth"] = 0.5


def style_dark_figure(fig, axes) -> None:
    fig.patch.set_facecolor(DARK_BG)
    for ax in np.asarray(axes, dtype=object).ravel():
        style_dark_axis(ax)


def style_dark_legend(legend) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor(DARK_PANEL)
    legend.get_frame().set_edgecolor(DARK_GRID)
    for text in legend.get_texts():
        text.set_color(DARK_TEXT)


def check_invariants(n: int, seats: int, friends: np.ndarray, council: np.ndarray) -> None:
    assert friends.shape == (n, friends.shape[1])
    assert len(np.unique(council)) == len(council)
    assert len(council) == seats
    assert council.min() >= 0 and council.max() < n
    assert np.all(friends >= 0) and np.all(friends < n)
    for i in range(min(n, 8)):
        assert i not in friends[i]
