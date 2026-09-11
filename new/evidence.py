#!/usr/bin/env python3
"""Reproduce narration-facing statistics for Meritopolis Scenes 1–4."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def load_scene(name: str):
    path = HERE / name / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


S1 = load_scene("scene1")
S2 = load_scene("scene2")
S3 = load_scene("scene3")
S4 = load_scene("scene4")
OUT = HERE / "evidence.json"
SEED = 42


def mean_ci(values, rng_seed=12345) -> dict:
    """Mean and deterministic 95% bootstrap confidence interval."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(rng_seed)
    draws = rng.choice(x, size=(4000, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": round(float(x.mean()), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "n": int(len(x)),
    }


def fraction_ci(successes: int, total: int) -> dict:
    """Wilson 95% interval for a proportion."""
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / den
    return {
        "count": int(successes),
        "n": int(total),
        "pct": round(100 * p, 2),
        "ci95_pct": [round(100 * (centre - half), 2), round(100 * (centre + half), 2)],
    }


def ratio_from_share(
    council_share_pct: float,
    group_share_pct: float,
    seats: int,
    n: int,
) -> tuple[float, float, float]:
    seats_a = seats * council_share_pct / 100
    n_a = n * group_share_pct / 100
    p_a = seats_a / n_a
    p_b = (seats - seats_a) / (n - n_a)
    return p_a, p_b, p_a / p_b if p_b > 0 else np.inf


def scene1_stats() -> dict:
    result = {}
    for size in ("small", "large"):
        cfg = C.SIZES[size]
        st = S1.run_ensemble(cfg["n"], cfg["council"], cfg["k_friends"], SEED)
        paths = st["paths"]
        lock_round = []
        for path in paths:
            hits = np.flatnonzero(path >= 90)
            lock_round.append(hits[0] if len(hits) else S1.N_ROUNDS + 1)
        result[size] = {
            "population": cfg["n"],
            "runs": S1.N_RUNS,
            "start_male_council_pct": 60.0,
            "start_seat_likelihood_ratio_male_vs_female": round(st["start_ratio"], 3),
            "late_male_council_pct": mean_ci(paths[:, S1.EQUIL_FROM:].mean(axis=1)),
            "final_all_male": fraction_ci(int((paths[:, -1] >= 99).sum()), len(paths)),
            "post_start_council_years_male_majority": fraction_ci(
                int((paths[:, 1:] > 50).sum()), int(paths[:, 1:].size)
            ),
            "round_first_reaches_90pct_male": mean_ci(lock_round),
        }
    return result


def scene2_run(n, seats, k, mode, seed) -> dict:
    rng = np.random.default_rng(seed)
    former_male = np.arange(n) < n // 2
    traits = S2.make_traits(former_male, rng, mode)
    friends = C.knn_friends(traits, k)
    council = S2.seed_council(former_male, seats, rng)
    history = [council.copy()]
    for _ in range(S2.N_ROUNDS):
        council = C.elect(council, friends, rng)
        history.append(council.copy())

    blue = traits[:, 0] > 0
    late = history[S2.EQUIL_FROM:]
    return {
        "city_blue_pct": 100 * blue.mean(),
        "late_blue_council_pct": np.mean([100 * blue[c].mean() for c in late]),
        "late_former_male_council_pct": np.mean(
            [100 * former_male[c].mean() for c in late]
        ),
        "former_group_pct": 50.0,
        "former_male_homophily_pct": 100 * former_male[friends][former_male].mean(),
        "former_female_homophily_pct": 100 * (~former_male[friends])[~former_male].mean(),
    }


def bootstrap_ratio(runs, share_key, group_key, seats, n, seed) -> dict:
    rng = np.random.default_rng(seed)
    shares = np.asarray([r[share_key] for r in runs])
    groups = np.asarray([r[group_key] for r in runs])

    def calculate(idx):
        p_a, p_b, ratio = ratio_from_share(
            shares[idx].mean(), groups[idx].mean(), seats, n
        )
        return 100 * p_a, 100 * p_b, ratio

    base = calculate(np.arange(len(runs)))
    ratios = []
    for _ in range(4000):
        idx = rng.integers(0, len(runs), len(runs))
        ratios.append(calculate(idx)[2])
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return {
        "p_seat_group_a_pct": round(base[0], 3),
        "p_seat_group_b_pct": round(base[1], 3),
        "likelihood_ratio": round(base[2], 3),
        "ratio_ci95": [round(float(lo), 3), round(float(hi), 3)],
    }


def scene2_stats() -> dict:
    result = {}
    modes = ("neutral", "aligned", "inverted")
    for size in ("small", "large"):
        cfg = C.SIZES[size]
        n, seats, k = cfg["n"], cfg["council"], cfg["k_friends"]
        size_result = {}
        for mode_index, mode in enumerate(modes):
            runs = [
                scene2_run(n, seats, k, mode, SEED + 1000 * mode_index + run)
                for run in range(S2.N_RUNS)
            ]
            blue_ratio = bootstrap_ratio(
                runs, "late_blue_council_pct", "city_blue_pct", seats, n,
                100 + mode_index,
            )
            former_ratio = bootstrap_ratio(
                runs,
                "late_former_male_council_pct",
                "former_group_pct",
                seats,
                n,
                200 + mode_index,
            )
            size_result[mode] = {
                "runs": S2.N_RUNS,
                "former_male_homophily_pct": mean_ci(
                    [r["former_male_homophily_pct"] for r in runs]
                ),
                "former_female_homophily_pct": mean_ci(
                    [r["former_female_homophily_pct"] for r in runs]
                ),
                "late_blue_council_pct": mean_ci(
                    [r["late_blue_council_pct"] for r in runs]
                ),
                "blue_vs_pink": blue_ratio,
                "former_male_vs_former_female": former_ratio,
                "late_former_male_council_pct": mean_ci(
                    [r["late_former_male_council_pct"] for r in runs]
                ),
            }
        result[size] = size_result
    return result


def same_colour_friendship_pct(traits, friends) -> float:
    blue = traits[:, 0] > 0
    return float(100 * (blue[friends] == blue[:, None]).mean())


def safe_correlation(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def scene3_single(n, seats, k, mode, seed) -> dict:
    rng = np.random.default_rng(seed)
    traits = S3.random_traits(n, rng)
    initial_corr = safe_correlation(traits[:, 0], traits[:, 1])
    council = S3.seed_council(n, seats, rng)
    shares = [S3.blue_share(traits, council)]
    city_shares = [100 * (traits[:, 0] > 0).mean()]
    for year in range(1, S3.N_YEARS + 1):
        traits, friends, council = S3.step(
            traits,
            council,
            mode,
            k,
            rng,
            rescale=year % S3.RESCALE_EVERY == 0,
        )
        shares.append(S3.blue_share(traits, council))
        city_shares.append(100 * (traits[:, 0] > 0).mean())
    friends = C.knn_friends(traits, k)
    blue = traits[:, 0] > 0
    animal_gap = (
        traits[blue, 1].mean() - traits[~blue, 1].mean()
        if blue.any() and (~blue).any()
        else 0.0
    )
    shares_array = np.asarray(shares)
    city_array = np.asarray(city_shares)
    late = shares_array[S3.N_YEARS // 2:]
    signs = np.sign(late - 50)
    nonzero_signs = signs[signs != 0]
    sign_changes = (
        int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))
        if len(nonzero_signs) > 1
        else 0
    )
    return {
        "end_blue_council_pct": shares[-1],
        "end_abs_council_imbalance_pp": abs(shares[-1] - 50),
        "end_city_blue_pct": 100 * blue.mean(),
        "end_abs_representation_gap_pp": abs(
            shares_array[-1] - city_array[-1]
        ),
        "late_extreme_council_time_pct": float(
            100 * (np.abs(late - 50) >= 35).mean()
        ),
        "late_parity_crossings": sign_changes,
        "initial_abs_colour_animal_correlation": abs(initial_corr),
        "end_colour_animal_correlation": safe_correlation(
            traits[:, 0], traits[:, 1]
        ),
        "end_abs_colour_animal_correlation": abs(
            safe_correlation(traits[:, 0], traits[:, 1])
        ),
        "end_animal_mean_gap_blue_minus_pink": float(animal_gap),
        "end_same_colour_friendship_pct": same_colour_friendship_pct(
            traits, friends
        ),
    }


def scene3_stats() -> dict:
    result = {}
    for size in ("small", "large"):
        cfg = C.SIZES[size]
        n_runs = S3.N_RUNS_SMALL if size == "small" else S3.N_RUNS_LARGE
        size_result = {}
        for mode_index, mode in enumerate(("random", "inherited")):
            runs = [
                scene3_single(
                    cfg["n"], cfg["council"], cfg["k_friends"], mode,
                    SEED + 2000 * mode_index + run,
                )
                for run in range(n_runs)
            ]
            end = np.asarray([r["end_blue_council_pct"] for r in runs])
            positive_corr = sum(r["end_colour_animal_correlation"] > 0 for r in runs)
            size_result[mode] = {
                "runs": n_runs,
                "end_abs_council_imbalance_pp": mean_ci(
                    [r["end_abs_council_imbalance_pp"] for r in runs]
                ),
                "blue_heavy": fraction_ci(int((end > 60).sum()), n_runs),
                "pink_heavy": fraction_ci(int((end < 40).sum()), n_runs),
                "end_city_blue_pct": mean_ci(
                    [r["end_city_blue_pct"] for r in runs]
                ),
                "end_abs_representation_gap_pp": mean_ci(
                    [r["end_abs_representation_gap_pp"] for r in runs]
                ),
                "late_extreme_council_time_pct": mean_ci(
                    [r["late_extreme_council_time_pct"] for r in runs]
                ),
                "runs_with_late_parity_crossing": fraction_ci(
                    sum(r["late_parity_crossings"] > 0 for r in runs), n_runs
                ),
                "initial_abs_colour_animal_correlation": mean_ci(
                    [r["initial_abs_colour_animal_correlation"] for r in runs]
                ),
                "end_abs_colour_animal_correlation": mean_ci(
                    [r["end_abs_colour_animal_correlation"] for r in runs]
                ),
                "correlation_sign_positive": fraction_ci(positive_corr, n_runs),
                "end_abs_animal_gap": mean_ci(
                    [abs(r["end_animal_mean_gap_blue_minus_pink"]) for r in runs]
                ),
                "end_same_colour_friendship_pct": mean_ci(
                    [r["end_same_colour_friendship_pct"] for r in runs]
                ),
            }
        result[size] = size_result
    return result


def scene4_stats() -> dict:
    result = {}
    for size in ("small", "large"):
        cfg = C.SIZES[size]
        n_runs = S4.N_RUNS_SMALL if size == "small" else S4.N_RUNS_LARGE
        size_result = {}
        for mode_index, mode in enumerate(("random", "inherited")):
            runs = [
                S4.simulate_run(
                    cfg["n"], cfg["council"], cfg["k_friends"], mode,
                    S4.N_YEARS,
                    np.random.default_rng(SEED + 2000 * mode_index + run),
                )
                for run in range(n_runs)
            ]
            end = np.asarray([r["blue"][-1] for r in runs])
            corr = [
                abs(safe_correlation(
                    r["final_traits"][:, S4.COLOUR],
                    r["final_traits"][:, S4.MERIT],
                ))
                for r in runs
            ]
            city_blue = [
                100 * (r["final_traits"][:, S4.COLOUR] > 0).mean()
                for r in runs
            ]
            for r in runs:
                S4.assert_merit_council(
                    r["final_traits"], r["final_council"], cfg["council"]
                )
            size_result[mode] = {
                "runs": n_runs,
                "end_abs_council_imbalance_pp": mean_ci(abs(end - 50)),
                "end_city_blue_pct": mean_ci(city_blue),
                "end_abs_representation_gap_pp": mean_ci(
                    [abs(a - b) for a, b in zip(end, city_blue)]
                ),
                "blue_heavy": fraction_ci(int((end > 60).sum()), n_runs),
                "pink_heavy": fraction_ci(int((end < 40).sum()), n_runs),
                "end_merit_advantage_points": mean_ci(
                    [r["gap"][-1] for r in runs]
                ),
                "end_abs_colour_merit_correlation": mean_ci(corr),
                "all_final_councils_are_exact_top_merit": True,
            }
        result[size] = size_result
    return result


def main() -> None:
    evidence = {
        "metadata": {
            "seed": SEED,
            "confidence_intervals": "95% bootstrap for means; Wilson for proportions",
            "generated_by": "new/evidence.py",
        },
        "scene1": scene1_stats(),
        "scene2": scene2_stats(),
        "scene3": scene3_stats(),
        "scene4": scene4_stats(),
    }
    OUT.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"evidence → {OUT}")


if __name__ == "__main__":
    main()
