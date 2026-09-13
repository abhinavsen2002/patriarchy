# -*- coding: utf-8 -*-
"""
Scene 6 — Partnerships
======================
Scene 4 remains the baseline: inherited births, deterministic nearest-neighbour
friendships in eleven traits, social learning, and top-merit council selection.
There are no random friendship slots and no sticky friendships.

We compare three social rules:

  baseline:
      Scene 4 unchanged.

  partner_friend:
      Each blue person is paired with at most one pink person, and vice versa,
      using globally nearest remaining opposite-colour matches. Unmatched
      people (when the two sides have different counts) have no partner.
      Normal friendship learning remains, and the close partner adds a
      separate social influence.

  partner_complement:
      The same opposite-colour partner is chosen, but the additional social
      target is the negative of the partner's traits. Partners therefore learn
      complementary rather than similar roles.

Run:
  python scene6.py --size small
  python scene6.py --size large --no-animate
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scene4"))
import common as C  # noqa: E402
import scene4 as S4  # noqa: E402

HERE = Path(__file__).resolve().parent
N_YEARS = S4.N_YEARS
N_RUNS_SMALL = 40
N_RUNS_LARGE = 12
PARTNER_INFLUENCE = 0.20
ANIMATION_SEED = {100: 67}
MODES = ("baseline", "partner_friend", "partner_complement")
LABELS = {
    "baseline": "No forced partnership",
    "partner_friend": "Partners act like friends",
    "partner_complement": "Partners complement each other",
}


def choose_partners(traits: np.ndarray) -> np.ndarray:
    """Strict one-to-one nearest opposite-colour partnerships."""
    n = len(traits)
    blue = traits[:, S4.COLOUR] > 0
    partner = np.full(n, -1, dtype=int)
    pink_people = np.flatnonzero(~blue)
    blue_people = np.flatnonzero(blue)
    if len(pink_people) == 0 or len(blue_people) == 0:
        return partner

    comparison_traits = np.delete(traits, S4.COLOUR, axis=1)
    a = comparison_traits[pink_people]
    b = comparison_traits[blue_people]
    a2 = np.einsum("ij,ij->i", a, a)[:, None]
    b2 = np.einsum("ij,ij->i", b, b)[None, :]
    d2 = a2 + b2 - 2.0 * a @ b.T

    # Globally shortest available pair first. Each bluey is paired with exactly
    # one unmatched pinky until one side runs out.
    pink_used = np.zeros(len(pink_people), dtype=bool)
    blue_used = np.zeros(len(blue_people), dtype=bool)
    order = np.argsort(d2, axis=None, kind="stable")
    for flat in order:
        pi, bi = np.unravel_index(flat, d2.shape)
        if pink_used[pi] or blue_used[bi]:
            continue
        p, q = pink_people[pi], blue_people[bi]
        partner[p] = q
        partner[q] = p
        pink_used[pi] = True
        blue_used[bi] = True
        if pink_used.all() or blue_used.all():
            break
    return partner


def learn(
    traits: np.ndarray,
    friends: np.ndarray,
    partner: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Normal social learning plus a distinct, equally strong partner effect."""
    friend_target = traits[friends].mean(axis=1)
    learned = traits + S4.INFLUENCE * (friend_target - traits)
    if mode != "baseline":
        valid = partner >= 0
        partner_target = traits[np.maximum(partner, 0)].copy()
        if mode == "partner_complement":
            partner_target = -partner_target
        learned[valid] += PARTNER_INFLUENCE * (
            partner_target[valid] - traits[valid]
        )
    learned += rng.normal(0.0, S4.NOISE, size=traits.shape)
    return C.clip_traits(learned)


def unique_cross_colour_friendships(
    traits: np.ndarray,
    friends: np.ndarray,
) -> int:
    """Absolute number of unique undirected normal friendships across colour."""
    n = len(traits)
    src = np.repeat(np.arange(n), friends.shape[1])
    dst = friends.ravel()
    lo = np.minimum(src, dst).astype(np.int64)
    hi = np.maximum(src, dst).astype(np.int64)
    packed = np.unique(lo * n + hi)
    a, b = packed // n, packed % n
    blue = traits[:, S4.COLOUR] > 0
    return int((blue[a] != blue[b]).sum())


def measurements(
    traits: np.ndarray,
    friends: np.ndarray,
    council: np.ndarray,
) -> tuple[float, float, int]:
    blue_power = 100.0 * (traits[council, S4.COLOUR] > 0).mean()
    cat_power = 100.0 * (traits[council, S4.ANIMAL] < 0).mean()
    cross = unique_cross_colour_friendships(traits, friends)
    return blue_power, cat_power, cross


def simulate_run(
    n: int,
    seats: int,
    k: int,
    mode: str,
    years: int,
    seed: int,
    record_frames: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    traits = S4.random_traits(n, rng)
    friends = S4.knn_friends(traits, k)
    council = S4.elect_by_merit(traits, seats)
    partner = choose_partners(traits) if mode != "baseline" else np.full(n, -1)

    blue, cat, cross = measurements(traits, friends, council)
    blue_path = [blue]
    cat_path = [cat]
    cross_path = [cross]
    frames = (
        [(traits.copy(), council.copy(), partner.copy())]
        if record_frames
        else None
    )

    for year in range(1, years + 1):
        friends = S4.knn_friends(traits, k)
        partner = (
            choose_partners(traits)
            if mode != "baseline"
            else np.full(n, -1, dtype=int)
        )
        traits = learn(traits, friends, partner, mode, rng)
        traits = S4.replace(traits, rng)
        if year % S4.RESCALE_EVERY == 0:
            traits = C.rescale_traits(traits)
        friends = S4.knn_friends(traits, k)
        partner = (
            choose_partners(traits)
            if mode != "baseline"
            else np.full(n, -1, dtype=int)
        )
        council = S4.elect_by_merit(traits, seats)
        S4.assert_merit_council(traits, council, seats)
        blue, cat, cross = measurements(traits, friends, council)
        blue_path.append(blue)
        cat_path.append(cat)
        cross_path.append(cross)
        if record_frames:
            frames.append((traits.copy(), council.copy(), partner.copy()))

    return {
        "blue": np.asarray(blue_path),
        "cat": np.asarray(cat_path),
        "cross": np.asarray(cross_path),
        "frames": frames,
    }


def run_ensemble(n: int, seats: int, k: int, seed: int, n_runs: int) -> dict:
    bundle = {}
    for mode in MODES:
        runs = [
            simulate_run(n, seats, k, mode, N_YEARS, seed + run)
            for run in range(n_runs)
        ]
        bundle[mode] = {
            key: np.asarray([run[key] for run in runs])
            for key in ("blue", "cat", "cross")
        }
    return bundle


def plot_results(bundle: dict, n: int, seats: int, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 12.0), facecolor="white")
    fig.suptitle(
        f"Scene 6 — opposite-colour partnerships  ·  N={n}, {seats} seats\n"
        "Inherited births · deterministic nearest friendships · top merit wins",
        fontsize=14,
        fontweight="bold",
    )
    t = np.arange(N_YEARS + 1)
    colours = {
        "baseline": C.C_GREY,
        "partner_friend": C.C_BLUE,
        "partner_complement": C.C_PINK,
    }
    panels = (
        ("blue", "% council blue", (0, 100), 50),
        ("cat", "% council cat", (0, 100), 50),
        ("cross", "cross-colour friendships (count)", (0, None), None),
    )
    for ax, (key, ylabel, ylim, reference) in zip(axes, panels):
        for mode in MODES:
            values = bundle[mode][key]
            mean = values.mean(axis=0)
            lo, hi = np.percentile(values, [10, 90], axis=0)
            colour = colours[mode]
            ax.fill_between(t, lo, hi, color=colour, alpha=0.12)
            ax.plot(t, mean, color=colour, lw=2.3, label=LABELS[mode])
        if reference is not None:
            ax.axhline(reference, color=C.C_GREY, ls="--", lw=0.9)
        ax.set_xlim(0, N_YEARS)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  plot → {out}")


def _unique_partner_edges(partner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    people = np.flatnonzero(partner >= 0)
    other = partner[people]
    lo = np.minimum(people, other)
    hi = np.maximum(people, other)
    packed = np.unique(lo.astype(np.int64) * len(partner) + hi)
    return packed // len(partner), packed % len(partner)


def _draw_world(
    ax,
    xy: np.ndarray,
    council: np.ndarray,
    partner: np.ndarray,
    title: str,
    n: int,
) -> None:
    ax.cla()
    blue = xy[:, S4.COLOUR] > 0
    a, b = _unique_partner_edges(partner)
    if len(a):
        from matplotlib.collections import LineCollection

        segments = np.stack(
            [xy[a][:, [S4.COLOUR, S4.MERIT]],
             xy[b][:, [S4.COLOUR, S4.MERIT]]],
            axis=1,
        )
        ax.add_collection(
            LineCollection(
                segments,
                colors=[(0.62, 0.65, 0.72, 0.16)],
                linewidths=0.85,
                zorder=1,
            )
        )
    size = 30 if n <= 100 else 7
    ax.scatter(
        xy[~blue, S4.COLOUR], xy[~blue, S4.MERIT],
        s=size, c=C.C_PINK_BRIGHT, alpha=0.95, linewidths=0, zorder=2,
    )
    ax.scatter(
        xy[blue, S4.COLOUR], xy[blue, S4.MERIT],
        s=size, c=C.C_BLUE_BRIGHT, alpha=0.95, linewidths=0, zorder=2,
    )
    ax.scatter(
        xy[council, S4.COLOUR], xy[council, S4.MERIT],
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
    ax.set_title(
        title, color=C.DARK_TEXT, fontsize=15, fontweight="bold", y=1.10
    )
    label = dict(color=C.DARK_TEXT, fontsize=12, fontweight="bold")
    ax.text(-0.02, 0.5, "Pink", transform=ax.transAxes, ha="right", va="center", **label)
    ax.text(1.02, 0.5, "Blue", transform=ax.transAxes, ha="left", va="center", **label)
    ax.text(0.5, -0.02, "Low merit", transform=ax.transAxes, ha="center", va="top", **label)
    ax.text(0.5, 1.01, "High merit", transform=ax.transAxes, ha="center", va="bottom", **label)


def animate_mode(
    n: int,
    seats: int,
    k: int,
    mode: str,
    seed: int,
    out: Path,
) -> None:
    run = simulate_run(
        n, seats, k, mode, N_YEARS, seed, record_frames=True
    )
    sub = 6
    fig = plt.figure(figsize=(16, 9), facecolor="#000000")
    ax = fig.add_axes([0.255, 0.14, 0.49, 0.72])
    ax.set_facecolor("#000000")
    year_text = fig.text(
        0.04, 0.92, "Year 0", color=C.DARK_TEXT, fontsize=25,
        fontweight="bold", ha="left",
    )

    positions = np.stack([frame[0] for frame in run["frames"]])
    ambient = C.make_ambient(n, amp=0.22, seed=47, period=168.0)

    def update(f):
        seg = min(f / sub, N_YEARS)
        t0 = int(np.floor(seg))
        t1 = min(t0 + 1, N_YEARS)
        u = seg - t0
        ease = u * u * (3.0 - 2.0 * u)
        traits = positions[t0] * (1.0 - ease) + positions[t1] * ease
        traits = traits.copy()
        traits[:, [S4.COLOUR, S4.MERIT]] += ambient(f)
        frame = run["frames"][t1 if ease >= 0.5 else t0]
        _draw_world(ax, traits, frame[1], frame[2], LABELS[mode], n)
        year_text.set_text(f"Year {int(round(seg))}")
        return ()

    animation = FuncAnimation(
        fig, update, frames=N_YEARS * sub + 1, interval=40, blit=False,
    )
    C.save_mp4(animation, out, fps=24, bg="#000000")
    plt.close(fig)


def main() -> None:
    args = C.parse_cli("Scene 6 — opposite-colour partnerships")
    for size in C.sizes_to_run(args.size):
        cfg = C.SIZES[size]
        n, seats, k = cfg["n"], cfg["council"], cfg["k_friends"]
        n_runs = N_RUNS_SMALL if size == "small" else N_RUNS_LARGE
        print(f"\n[{size}] Scene 6  N={n}  seats={seats}  runs={n_runs}")
        bundle = run_ensemble(n, seats, k, args.seed, n_runs)
        for mode in MODES:
            blue = bundle[mode]["blue"]
            cat = bundle[mode]["cat"]
            cross = bundle[mode]["cross"]
            print(
                f"  {mode}: late |blue−50|={np.abs(blue[:, 60:] - 50).mean():.1f} pp  "
                f"|cat−50|={np.abs(cat[:, 60:] - 50).mean():.1f} pp  "
                f"cross ties={cross[:, -1].mean():.0f}"
            )
        plot_results(bundle, n, seats, HERE / f"scene6_n{n}.png")
        if args.animate:
            animation_seed = ANIMATION_SEED.get(n, args.seed)
            print(f"  animation seed={animation_seed}")
            animate_mode(
                n, seats, k, "partner_friend", animation_seed,
                HERE / f"scene6_n{n}_friends.mp4",
            )
            animate_mode(
                n, seats, k, "partner_complement", animation_seed,
                HERE / f"scene6_n{n}_complement.mp4",
            )


if __name__ == "__main__":
    main()
