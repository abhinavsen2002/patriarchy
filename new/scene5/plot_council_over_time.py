#!/usr/bin/env python3
"""Council colour over time at 5, 10, 25, 50% random friends (Scene 5)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scene4"))
sys.path.insert(0, str(ROOT / "scene5"))
import common as C  # noqa: E402
import scene5 as S5  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 42
# actual random share ≈ configured mix / 2 because openness ~ U(0,1)
LEVELS = [(5, 0.1), (10, 0.2), (25, 0.5), (50, 1.0)]


def make_figure(size: str, out: Path) -> None:
    cfg = C.SIZES[size]
    n, seats, k = cfg["n"], cfg["council"], cfg["k_friends"]
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 8.5), facecolor=C.DARK_BG, sharex=True, sharey=True
    )
    axes_flat = axes.ravel()
    C.style_dark_figure(fig, axes_flat)
    years = np.arange(S5.N_YEARS + 1)

    for ax, (pct, mix) in zip(axes_flat, LEVELS):
        run = S5.simulate_run(n, seats, k, "random", mix, 0, SEED)
        council = run["blue"]
        city = run["city_blue"]
        ax.axhline(50, color=C.DARK_MUTED, ls=":", lw=1.0, alpha=0.6)
        ax.fill_between(
            years, 50, council, where=council >= 50,
            color=C.C_BLUE_BRIGHT, alpha=0.12, linewidth=0,
        )
        ax.fill_between(
            years, 50, council, where=council < 50,
            color=C.C_PINK_BRIGHT, alpha=0.12, linewidth=0,
        )
        ax.plot(years, city, color=C.DARK_TEXT, lw=1.4, ls="--",
                alpha=0.85, label="whole city")
        ax.plot(years, council, color=C.C_BLUE_BRIGHT, lw=2.4,
                label="council")
        ax.set_title(
            f"{pct}% random friends  ({round(mix / 2 * k)} of {k})",
            color=C.DARK_TEXT, fontsize=12, fontweight="bold",
        )
        ax.set_xlim(0, S5.N_YEARS)
        ax.set_ylim(0, 100)

    for ax in axes[1]:
        ax.set_xlabel("Year")
    for ax in axes[:, 0]:
        ax.set_ylabel("% blue")
    legend = axes_flat[0].legend(frameon=False, loc="upper left", fontsize=9)
    C.style_dark_legend(legend)
    fig.suptitle(
        f"Scene 5 — council colour over 500 years  ·  N={n}  ·  one representative world (seed {SEED})",
        color=C.DARK_TEXT, fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, facecolor=C.DARK_BG)
    plt.close(fig)
    print(f"plot → {out}")


def main() -> None:
    for size in ("small", "large"):
        make_figure(size, HERE / f"scene5_council_over_time_n{C.SIZES[size]['n']}.png")


if __name__ == "__main__":
    main()
