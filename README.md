# Meritopolis

A city simulation of how inequality can persist after gender is removed from the rules — through friendships, influence, inheritance, and even an “objective” merit test.

The current work lives in `new/`. `old/` is earlier experiments.

---

## Scenes

| Scene | What it tests |
|---|---|
| 1 | Same-gender friendships and a 60% male council lock in male power |
| 2 | Gender is gone; colour/animal stereotypes still pack similar people together |
| 3 | Start equal. Influence, death, and birth make 50/50 unstable |
| 4 | Replace the network election with a pure merit ranking |
| 5 | Add random friendships (and optional stickiness) to break echo chambers |

Scene 5 is run to year 500. Random ties help in the large city; they do not bring representation to zero. Stickiness is not required once random ties exist.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

From `new/`:

```bash
python scene1/scene1.py --size small
python scene2/scene2.py --size small
python scene3/scene3.py --size small
python scene4/scene4.py --size small
python scene5/scene5.py --size small
```

`--size large` or `--size both` also work. Animations are written next to each scene script.

Reproduce narration statistics:

```bash
python evidence.py
```

---

## Files

| Path | Description |
|---|---|
| `new/common.py` | Shared elections, friendships, trait learning, dark animation theme |
| `new/sceneN/sceneN.py` | Scene N simulation, plots, and animations |
| `new/evidence.py` | Reproducible stats for scenes 1–4 |
| `new/scene5/plot_council_over_time.py` | Council colour over 500 years at 5/10/25/50% random friends |
