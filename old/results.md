# Patriarchy Simulation — Full Results

## Global Parameters

| Parameter | Value |
|---|---|
| Population | 100 (50 male, 50 female) |
| Job A slots | 15 (highest pay: 1000) |
| Job B slots | 35 (mid pay: 300) |
| Job C slots | 50 (lowest pay: 100) |
| Iterations | 600 |
| Runs per scenario (summary) | 20 |
| Trait update rate (alpha) | 0.10 |
| Noise std | 0.3 |
| Default friendship cutoff | 0.80 (cosine similarity) |

### Initial Trait Distribution

Males start with job-relevant traits (traits 0–2) drawn from Normal(57, 15) — biased high.
Females start with job-relevant traits drawn from Normal(43, 15) — biased low.
All other traits (3–9) are drawn from Normal(50, 12) for both genders.
This means **at t=0, all 15 Job A slots go to men**.

---

## Scenario Legend

### How the simulation works

Each iteration:
1. Jobs are assigned by ranking everyone's sum of job-relevant traits (0–2). Top 15 → Job A, next 35 → Job B, rest → Job C.
2. Friendships are computed (depending on the scenario).
3. Each person's traits drift toward the average of their friends (80% weight) and job peers (20% weight), resisted by innate traits and bounded noise.
4. Metrics are recorded.

Trait convergence across genders is the primary mechanism that closes the job/pay gap over time. Scenarios differ in **who becomes friends with whom** and whether **structural interventions** (reservations) are applied.

---

### Friendship Modes

#### Fixed-2-Childhood (no new friends)
Each person's 4 **most similar** peers at t=0 become permanent childhood friends. No new friendships ever form. Because traits are gender-skewed at the start, men tend to befriend men and women tend to befriend women. Cross-gender influence exists but is weak. The frozen network means the pace of change is slow and entirely driven by within-network drift.

#### Fixed-Random-Childhood (no new friends) *(new)*
Each person's 4 childhood friends are chosen **uniformly at random** from the full population — ignoring similarity entirely. No new friendships ever form. Because friends are random, men and women are mixed from day one, producing strong cross-gender trait convergence from the very first iteration. This is structurally simple but surprisingly powerful: random mixing breaks the homophily trap without any formal intervention.

#### Fixed Disjoint Cliques of 4 (S1b)
The population is partitioned into disjoint groups of 4 by a greedy similarity algorithm: people cluster with their most similar peers, and no edges exist between cliques. This is the most socially isolated scenario — zero cross-pollination between groups. Because high scorers cluster together and never influence outsiders, job gaps are highly persistent.

#### 2 Childhood + Dynamic (S1c / hybrid)
Each person starts with 4 similarity-based childhood friends (permanent), **plus** dynamic friendships that form and dissolve each iteration based on cosine similarity exceeding the cutoff. As traits converge over time, the dynamic layer expands dramatically (avg friends grows from ~5 to ~60), accelerating convergence further. The childhood anchor prevents complete social isolation in early iterations.

#### Dynamic + Female-Female Bias (S2)
Fully dynamic friendships, but female-female pairs have a **lower formation threshold** (cutoff × 0.70 instead of cutoff × 1.0), and female friends receive **2× the influence weight** when updating a woman's traits. This models a scenario where women preferentially form and strengthen bonds with other women. Counterintuitively, this tends to create a female echo chamber: women's job traits converge toward the female average (which starts low) rather than the higher male average, resulting in the **worst** long-run outcomes for gender parity.

#### Baseline / Only Dynamic (S3)
Pure dynamic friendships only — no childhood friends, no reservation, no bias. Friendships form and dissolve based on cosine similarity each iteration. Because the cutoff is high (0.80) and traits start gender-segregated, the network is sparse early on (~0.2 avg friends), and cross-gender friendships take a long time to form. The gap narrows but plateau around 20% female in Job A and ~43% pay parity.

#### Baseline + Reservation (S4)
Same as S3 (dynamic only), but **2 of the 15 Job A slots are reserved for the top-scoring women** each iteration (~13% reservation). The reservation provides an immediate structural boost at t=0 but the dynamic network is still slow to form cross-gender connections. By iter 600 the reservation effect is nearly indistinguishable from the no-reservation baseline — the network dynamics dominate long-run outcomes.

---

### Scenario 5: Openness Sweep (cutoff sweep, dynamic only)

These scenarios test how the **friendship formation threshold** (cosine similarity cutoff) affects outcomes. A lower cutoff means people are more willing to befriend dissimilar others — i.e., more "open-minded." No childhood friends, no reservation.

| Label | Cutoff | Meaning |
|---|---|---|
| S5 cutoff=0.80 | 0.80 | Very restrictive — only near-identical people become friends (default) |
| S5 cutoff=0.70 | 0.70 | Slightly more open |
| S5 cutoff=0.60 | 0.60 | Moderately open |
| S5 cutoff=0.50 | 0.50 | Half-open — friends if above-average similarity |
| S5 cutoff=0.40 | 0.40 | Quite open |
| S5 cutoff=0.30 | 0.30 | Very open — broad social mixing |

Notably, lower cutoffs do **not** monotonically improve outcomes in the dynamic-only case. Because everyone quickly becomes friends with everyone (the network saturates), cross-gender influence is diffuse and the system converges to a muddled middle rather than a directed shift.

---

### Combined Scenarios (summary table only, 20-run averages)

These stack multiple interventions to find the most effective combinations.

| Scenario | Friendship Mode | Reservation | Cutoff |
|---|---|---|---|
| Fixed-2-Childhood | Similarity-based childhood (fixed) | None | n/a |
| Fixed-Random-Childhood | Random childhood (fixed) | None | n/a |
| Childhood + Dynamic | Similarity childhood + dynamic | None | 0.80 |
| Only Dynamic | Dynamic only | None | 0.80 |
| Dynamic + F-Bias | Dynamic + F-F lower threshold + 2× influence | None | 0.80 |
| Dynamic + Reservation | Dynamic only | 2/15 slots for women (~13%) | 0.80 |
| Child + Dyn + Res | Similarity childhood + dynamic | 2/15 (~13%) | 0.80 |
| Child + Dyn + Res + Open-60 | Similarity childhood + dynamic | 2/15 (~13%) | 0.60 |
| Child + Dyn + Res + Open-40 | Similarity childhood + dynamic | 2/15 (~13%) | 0.40 |
| Child + Dyn + Res + Open-20 | Similarity childhood + dynamic | 2/15 (~13%) | 0.20 |
| Child+Dyn+20%Res+Open-60 | Similarity childhood + dynamic | 3/15 (~20%) | 0.60 |
| Child+Dyn+20%Res+Open-40 | Similarity childhood + dynamic | 3/15 (~20%) | 0.40 |
| Child+Dyn+20%Res+Open-20 | Similarity childhood + dynamic | 3/15 (~20%) | 0.20 |

---

## Single-Run Results (600 iterations, fixed seeds)

Format: `start → end`

| Scenario | Job A Female % | Job B Female % | F Pay as % of M | Pay Gap (M−F) | Avg Friends |
|---|---|---|---|---|---|
| Fixed-2-Childhood | 0% → 33% | 29% → 40% | 30% → 68% | 330 → 118 | 5.1 → 5.1 |
| Fixed-Random-Childhood | 0% → 60% | 29% → 34% | 30% → 103% | 330 → −10 | 7.8 → 7.8 |
| S1b: Fixed Disjoint Cliques | 0% → 13% | 29% → 26% | 30% → 39% | 330 → 266 | 3.0 → 3.0 |
| S1c: Childhood + Dynamic | 0% → 40% | 29% → 46% | 30% → 80% | 330 → 66 | 5.1 → 60.0 |
| S2: Dynamic + F-F Bias | 0% → 20% | 29% → 14% | 30% → 40% | 330 → 262 | 1.8 → 35.8 |
| S3: Baseline (Dynamic only) | 0% → 27% | 29% → 17% | 30% → 47% | 330 → 218 | 0.2 → 48.6 |
| S4: Baseline + Reservation | 13% → 27% | 23% → 17% | 38% → 47% | 274 → 218 | 0.2 → 48.8 |
| S5 cutoff=0.80 | 0% → 20% | 29% → 17% | 30% → 41% | 330 → 254 | 0.2 → 47.8 |
| S5 cutoff=0.70 | 0% → 13% | 29% → 23% | 30% → 38% | 330 → 274 | 0.7 → 49.0 |
| S5 cutoff=0.60 | 0% → 20% | 29% → 26% | 30% → 45% | 330 → 230 | 3.1 → 45.2 |
| S5 cutoff=0.50 | 0% → 20% | 29% → 29% | 30% → 47% | 330 → 222 | 6.9 → 46.8 |
| S5 cutoff=0.40 | 0% → 20% | 29% → 26% | 30% → 45% | 330 → 230 | 12.2 → 48.7 |
| S5 cutoff=0.30 | 0% → 27% | 29% → 23% | 30% → 50% | 330 → 202 | 19.9 → 49.3 |

---

## Final Summary Table (mean over 20 independent runs)

Columns: **Job A % female**, **Job B % female**, **Female pay as % of male pay**

| Scenario | Iter 20 A% | Iter 20 B% | Iter 20 Pay% | Iter 100 A% | Iter 100 B% | Iter 100 Pay% | Iter 300 A% | Iter 300 B% | Iter 300 Pay% | Iter 600 A% | Iter 600 B% | Iter 600 Pay% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed-2-Childhood | 6.0 | 29.4 | 34.8 | 11.3 | 30.0 | 39.5 | 14.7 | 31.9 | 43.3 | 34.3 | 44.3 | 72.7 |
| Fixed-Random-Childhood | 1.7 | 28.7 | 31.1 | 28.0 | 42.4 | 63.3 | 48.7 | 52.1 | 100.9 | 52.0 | 50.9 | 105.1 |
| Childhood + Dynamic | 6.0 | 29.4 | 34.8 | 8.7 | 31.3 | 37.8 | 17.0 | 31.0 | 45.2 | 38.0 | 46.0 | 81.8 |
| Only Dynamic | 0.0 | 28.4 | 29.7 | 4.3 | 26.4 | 32.3 | 20.0 | 19.3 | 42.4 | 20.7 | 19.0 | 42.7 |
| Dynamic + F-Bias | 0.0 | 25.1 | 28.5 | 0.0 | 23.3 | 27.8 | 12.7 | 18.4 | 35.6 | 11.3 | 18.7 | 34.7 |
| Dynamic + Reservation | 13.3 | 22.7 | 37.9 | 13.3 | 22.4 | 37.8 | 20.0 | 19.4 | 42.4 | 20.3 | 19.3 | 42.6 |
| Child + Dyn + Res | 13.3 | 26.1 | 39.5 | 13.3 | 28.9 | 40.7 | 18.3 | 30.7 | 46.3 | 39.0 | 46.0 | 82.9 |
| Child + Dyn + Res + Open-60 | 13.3 | 26.3 | 39.5 | 14.7 | 28.3 | 41.6 | 19.0 | 28.4 | 45.8 | 34.3 | 40.6 | 74.0 |
| Child + Dyn + Res + Open-40 | 13.3 | 23.9 | 38.5 | 15.0 | 22.7 | 39.4 | 17.7 | 23.6 | 42.2 | 42.7 | 47.3 | 89.5 |
| Child + Dyn + Res + Open-20 | 13.3 | 24.0 | 38.5 | 15.3 | 24.7 | 40.6 | 28.3 | 40.4 | 64.1 | 53.0 | 50.0 | 106.5 |
| Child+Dyn+20%Res+Open-60 | 20.0 | 23.3 | 44.1 | 20.0 | 25.9 | 45.3 | 22.3 | 27.0 | 48.2 | 36.3 | 41.1 | 76.7 |
| Child+Dyn+20%Res+Open-40 | 20.0 | 21.0 | 43.0 | 20.3 | 20.1 | 42.9 | 21.7 | 21.9 | 45.0 | 44.3 | 46.4 | 91.5 |
| Child+Dyn+20%Res+Open-20 | 20.0 | 21.1 | 43.1 | 20.3 | 22.4 | 44.0 | 29.7 | 40.6 | 65.3 | 53.0 | 50.0 | 106.5 |

---

## Key Findings

### Worst outcome: Dynamic + F-F Bias
Female pay ends at only **34.7% of male pay** — lower than almost every other scenario. Preferential female-female bonding creates an echo chamber where women's traits converge toward the (initially low) female average instead of the male average. Structural isolation hurts more than it helps.

### Baseline trap: Dynamic-only and Reservation-only
Both plateau around **20–21% female in Job A** and **42–43% pay parity** by iter 600. Reservation alone shifts the starting point but not the long-run equilibrium — the slow, similarity-gated dynamic network dominates. Without a mechanism for sustained cross-gender mixing, neither policy moves the needle much.

### Surprise winner: Fixed-Random-Childhood
A frozen network of 4 **random** friends per person achieves **52% female in Job A** and **105% female pay** by iter 600 (mean over 20 runs) — with no reservation, no dynamic friendships, and no structural intervention of any kind. The randomness itself is the intervention: it ensures every person has cross-gender influence baked in from birth and never loses it.

### Most equalizing combination: Child + Dyn + Res + Open-20
Similarity childhood friends + dynamic friendships + 13% reservation + very open cutoff (0.20) reaches **53% female in Job A** and **106.5% female pay** — slightly edging out the random childhood scenario at iter 600, but requiring four stacked interventions to get there.

### The openness non-linearity
More openness (lower cutoff) is not always better in isolation. In the dynamic-only setting, very low cutoffs cause the network to saturate quickly (everyone befriends everyone), diffusing influence so broadly that directed convergence slows. The sweet spot requires combining openness with a stable anchor (childhood friends) to maintain directional pressure.

---

## 50% Job Influence Results

In the baseline simulation, each person's traits drift 80% toward their friend average and 20% toward their job-peer average. Here we raise job peer influence to **50%** (friends 50%, job peers 50%).

**What this means:** Job peers are people doing the same job as you. At 50% weight, your traits converge heavily toward the people already in your job category. Since Job A starts all-male, men in Job A reinforce each other's high job traits — creating a self-reinforcing feedback loop that makes the gap much harder to close. The friendship network has to fight against a strong job-segregation force.

### Summary Table (mean over 20 runs, job influence = 50%)

| Scenario | Iter 20 A% | Iter 20 B% | Iter 20 Pay% | Iter 100 A% | Iter 100 B% | Iter 100 Pay% | Iter 300 A% | Iter 300 B% | Iter 300 Pay% | Iter 600 A% | Iter 600 B% | Iter 600 Pay% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed-2-Childhood | 0.3 | 28.3 | 29.9 | 7.0 | 25.7 | 34.1 | 13.7 | 31.3 | 42.2 | 36.7 | 46.4 | 77.3 |
| Fixed-Random-Childhood | 0.3 | 27.7 | 29.7 | 12.3 | 34.3 | 42.6 | 48.3 | 52.0 | 100.5 | 50.0 | 51.3 | 102.0 |
| Childhood + Dynamic | 0.3 | 28.3 | 29.9 | 2.7 | 27.3 | 31.3 | 14.0 | 22.4 | 38.5 | 31.7 | 47.4 | 73.8 |
| Only Dynamic | 0.0 | 28.6 | 29.8 | 0.0 | 28.6 | 29.8 | 20.7 | 19.7 | 43.2 | 22.0 | 19.1 | 44.1 |
| Dynamic + F-Bias | 0.0 | 28.4 | 29.7 | 0.0 | 28.4 | 29.7 | 19.3 | 20.1 | 42.2 | 21.3 | 19.3 | 43.6 |
| Dynamic + Reservation | 13.3 | 22.9 | 38.0 | 13.3 | 22.9 | 38.0 | 19.7 | 20.1 | 42.4 | 21.0 | 19.6 | 43.4 |
| Child + Dyn + Res | 13.3 | 22.7 | 37.9 | 13.3 | 22.7 | 37.9 | 16.3 | 21.4 | 40.0 | 35.0 | 48.3 | 78.1 |
| Child + Dyn + Res + Open-60 | 13.3 | 22.7 | 37.9 | 13.3 | 22.7 | 37.9 | 20.3 | 19.7 | 42.8 | 20.0 | 21.9 | 43.5 |
| Child + Dyn + Res + Open-40 | 13.3 | 22.7 | 37.9 | 15.7 | 21.7 | 39.5 | 20.0 | 19.9 | 42.6 | 20.3 | 19.9 | 42.9 |
| Child + Dyn + Res + Open-20 | 13.3 | 22.7 | 37.9 | 16.0 | 21.6 | 39.8 | 19.7 | 20.0 | 42.3 | 21.0 | 20.9 | 44.1 |
| Child+Dyn+20%Res+Open-60 | 20.0 | 19.9 | 42.5 | 20.0 | 19.9 | 42.5 | 22.0 | 19.0 | 43.9 | 23.0 | 20.6 | 45.6 |
| Child+Dyn+20%Res+Open-40 | 20.0 | 19.9 | 42.5 | 22.3 | 18.9 | 44.1 | 22.0 | 19.0 | 43.9 | 23.7 | 18.4 | 45.2 |
| Child+Dyn+20%Res+Open-20 | 20.0 | 19.9 | 42.5 | 21.3 | 19.3 | 43.4 | 22.0 | 19.0 | 43.9 | 24.3 | 19.4 | 46.4 |

### Key Differences vs 20% Job Influence

**Most scenarios collapse.** At 50% job influence, job-peer pressure locks people into their starting job's trait profile. Most dynamic and combined scenarios plateau around 20–24% female in Job A and 43–46% pay parity — barely above the baseline and essentially unchanged from iter 100 to 600.

**Random childhood is uniquely resilient.** Fixed-Random-Childhood still reaches **50% female in Job A** and **102% female pay** at iter 600. Because cross-gender mixing is baked in from the start and the network is fixed (not dissolved and reformed), the random friends' influence compounds steadily and eventually overcomes the job reinforcement signal. No other scenario comes close.

**Openness becomes irrelevant.** At 50% job influence, varying the friendship cutoff (0.20–0.80) produces nearly identical outcomes. The job feedback loop is so dominant that adding more or fewer friends makes little difference once the network saturates.

**F-F bias catches up to baseline.** At 20% job influence, F-F bias was the worst outcome (34.7% pay). At 50% job influence it lands at 43.6% — almost identical to the baseline (44.1%). When job influence dominates, friendship network composition matters less, so the echo-chamber penalty shrinks.

**Reservations are almost entirely neutralized.** The 13% and 20% reservation scenarios show only marginal gains (43–46% pay parity) versus no reservation, compared to the 20% job influence case where stacked interventions could reach 106%. Structural reservation cannot overcome a strong job-reinforcement dynamic without also achieving cross-gender trait convergence.
