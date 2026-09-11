#!/usr/bin/env python3
"""Plot Scene 5 metrics at 5, 10, 25, and 50% mean random friends."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
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
# actual random share ≈ configured mix / 2 because openness ~ U(0,1)
TARGET_PCT = (5, 10, 25, 50)
MIX_FOR_PCT = {5: 0.1, 10: 0.2, 25: 0.5, 50: 1.0}


def run_missing_mix(size: str, mix: float, seed: int) -> dict:
    cfg = C.SIZES[size]
    n_runs = S5.N_RUNS_SMALL if size == "small" else S5.N_RUNS_LARGE
    tasks = [
        (
            cfg["n"],
            cfg["council"],
            cfg["k_friends"],
            S5.PRIMARY_MODE,
            mix,
            0,
            seed + run,
        )
        for run in range(n_runs)
    ]
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(S5.run_task, tasks))
    return S5.summarize(runs)


def cell(results: dict, size: str, mix: float) -> dict:
    return results["sizes"][size]["cells"][f"random={mix:g},sticky=0"]


def main() -> None:
    saved = json.loads((HERE / "scene5_results.json").read_text())
    seed = int(saved["metadata"]["seed"])
    extra_path = HERE / "scene5_mix02.json"
    if extra_path.exists():
        extra = json.loads(extra_path.read_text())
    else:
        extra = {}
        for size in ("small", "large"):
            print(f"[{size}] running 10% random friends (mix=0.2, sticky=0)")
            extra[size] = run_missing_mix(size, 0.2, seed)
        extra_path.write_text(json.dumps(extra, indent=2) + "\n")

    series = {}
    for size in ("small", "large"):
        series[size] = {"echo": [], "echo_lo": [], "echo_hi": [],
                        "gap": [], "gap_lo": [], "gap_hi": [],
                        "same": []}
        for pct in TARGET_PCT:
            mix = MIX_FOR_PCT[pct]
            c = extra[size] if mix == 0.2 else cell(saved, size, mix)
            echo = c["echo_chamber_excess_pp"]
            gap = c["late_abs_representation_gap_pp"]
            series[size]["echo"].append(echo["mean"])
            series[size]["echo_lo"].append(echo["ci95"][0])
            series[size]["echo_hi"].append(echo["ci95"][1])
            series[size]["gap"].append(gap["mean"])
            series[size]["gap_lo"].append(gap["ci95"][0])
            series[size]["gap_hi"].append(gap["ci95"][1])
            series[size]["same"].append(c["same_colour_friend_pct"]["mean"])
            print(
                f"{size} {pct}%  echo={echo['mean']:.2f}  "
                f"gap={gap['mean']:.2f}  actual={c['actual_random_fraction']['mean']:.3f}"
            )

    x = np.array(TARGET_PCT, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), facecolor="white")
    fig.suptitle(
        "Scene 5 — 5%, 10%, 25%, 50% random friends\n"
        "Random births, no stickiness, years 250–500 (echo at year 500)",
        fontsize=13,
        fontweight="bold",
    )
    specs = (
        (
            axes[0],
            "echo",
            "Echo excess (pp)",
            "Same-colour friendship minus chance",
        ),
        (
            axes[1],
            "gap",
            "Representation gap (pp)",
            "Mean |council blue − city blue|",
        ),
    )
    colors = {"small": "#C45C26", "large": "#2F6FED"}
    labels = {"small": "N=100  (20 friends)", "large": "N=1,000  (200 friends)"}
    for ax, key, ylabel, subtitle in specs:
        for size in ("small", "large"):
            y = np.array(series[size][key])
            lo = np.array(series[size][f"{key}_lo"])
            hi = np.array(series[size][f"{key}_hi"])
            ax.errorbar(
                x,
                y,
                yerr=[y - lo, hi - y],
                color=colors[size],
                marker="o",
                lw=2.2,
                capsize=4,
                label=labels[size],
            )
        ax.set_xticks(TARGET_PCT)
        ax.set_xlabel("Mean random friends (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlim(0, 55)
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = HERE / "scene5_random_5_10_25_50.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"plot → {out}")

    (HERE / "scene5_random_5_10_25_50.json").write_text(
        json.dumps({"targets_pct": TARGET_PCT, "series": series}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
