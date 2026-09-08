# SCLUB-RP

Bandit clustering algorithms with combined dimensionality and clustering reductions.

## Layout

```
algorithms/
  base.py              # MultiUserLinearBandit base class
  linucb/              # LinUCB, LinUCB_RP, LinUCB_IND, LinUCB_IND_RP
  club/                # CLUB (Gentile et al. 2014)
  sclub/               # SCLUB + RP + CA + RP_CA
  bandit/              # Our proposals: Diag, Sketch, History, Window
algo_registry.py       # Single source of truth for algos, default params, plot styles
data/
  synthetic.py         # Synthetic data generator
  movielens.py         # MovieLens preprocessor (run as a script)
main.py                # Run experiments with checkpoints + incremental plotting
plot.py                # Re-plot from .npz files (no recomputation)
launch.sh              # Helper to launch experiments via nohup with named groups
get_movielens.sh       # Helper to download + preprocess MovieLens
results/               # Output: results/<exp>/<algo>.npz + plots
```

## Quick start (synthetic)

```bash
# Run all algorithms on the 'small' experiment (1 run each)
python main.py --experiment small

# Run a SINGLE algo by name (positional argument, case-insensitive)
python main.py linucb
python main.py sclub --experiment medium

# Run a GROUP of algorithms (defined in algo_registry.GROUPS)
python main.py slow                    # LinUCB + LinUCB_IND + CLUB + SCLUB
python main.py rp                      # all RP variants
python main.py bandits                 # all Bandit_* variants

# Mix anything: any combination of algo names and group names works
python main.py linucb sclub            # two algos
python main.py lin bandit_lite         # two groups → LinUCB+LinUCB_RP+Bandit_Diag+Bandit_Window
python main.py linucb_rp linucb_ind_rp # ad-hoc pair

# Run all (default if no positional arg given)
python main.py

# List available algos and groups
python main.py --list
```

For SSH + nohup workflow:

```bash
ssh server1
cd SCLUB-RP
nohup python main.py slow --experiment medium &      # logs to results/medium/nohup_medium_slow.log
exit

ssh server2
cd SCLUB-RP
nohup python main.py rp --experiment medium &        # logs to results/medium/nohup_medium_rp.log
exit

ssh server3
cd SCLUB-RP
nohup python main.py bandits --experiment medium &   # logs to results/medium/nohup_medium_bandits.log
exit
```

The log filename is auto-derived from the targets. Each server gets its own
log so they don't interleave on a shared filesystem. Algorithms checkpoint
independently per file, so two servers writing different algos don't collide.

```bash
# Other useful options
python main.py --experiment medium --nb_runs 5         # average over runs
python main.py linucb --T 100000 --k 25 --experiment custom
python main.py --experiment medium --plot-only         # replot only
```

## Plotting

After a run, `.npz` checkpoints sit in `results/<exp>/`. Plots are also
produced **incrementally during runs**, so a partial plot is on disk if the
script crashes.

```bash
# Replot all algorithms in an experiment without re-running
python main.py --experiment small --plot-only

# Plot directly via plot.py
python plot.py            # plots 'default' experiment
python plot.py small      # plots a specific experiment
python plot.py all        # plots every experiment under results/
```

Plots use:
- **Fixed colors and markers per algorithm** (defined in `algo_registry.STYLES`)
- **Staggered marker positions** so overlapping curves are visually distinguishable
- **Log Y-axes** by default

## Multi-server runs (your workflow)

Your usual workflow: SSH to each server, launch `nohup python main.py <something> &`,
disconnect. The new positional CLI makes this clean — no commenting/uncommenting code:

```bash
# server1: the slow group (CLUB is the bottleneck, runs alongside the simpler ones)
ssh server1
cd SCLUB-RP
nohup python main.py slow --experiment medium > /dev/null 2>&1 &
exit

# server2: the RP variants
ssh server2
cd SCLUB-RP
nohup python main.py rp --experiment medium > /dev/null 2>&1 &
exit

# server3: our Bandit variants
ssh server3
cd SCLUB-RP
nohup python main.py bandits --experiment medium > /dev/null 2>&1 &
exit
```

Each invocation writes its own log file at
`results/medium/nohup_medium_<group>.log`, so even on a shared filesystem
they don't overwrite each other.

You can also pass arbitrary combinations:

```bash
nohup python main.py linucb_rp linucb_ind_rp --experiment medium &  # ad-hoc pair
nohup python main.py lin bandit_lite &                              # two groups merged
```

The progress logging shows percentage complete (every 5%), elapsed time, and
ETA per algorithm, with no tqdm so it stays grep-friendly:

```bash
grep "++ "    results/medium/nohup_medium_*.log    # done lines
grep "FAILED" results/medium/nohup_medium_*.log
tail -f       results/medium/nohup_medium_slow.log # live monitoring
```

The optional helper `launch.sh` wraps this for convenience but isn't required:

```bash
./launch.sh medium slow      # equivalent to: nohup python main.py slow --experiment medium &
./launch.sh medium "linucb sclub"   # explicit list also works
```

## MovieLens

```bash
# 1) Download + preprocess in one step (small / 1m / 25m)
./get_movielens.sh small

# 2) Run experiments on it (use --T to limit horizon if needed)
python main.py --experiment medium --data-npz data/movielens_small.npz
```

The preprocessor:
- One-hot encodes 20 genres + 6 rating statistics (mean/std/min/max/count/frac≥4)
+ 14 release-year buckets + bias terms = ~46 base features
- Pads up to `--d-target` (default 100) using deterministic random Fourier-like
features built from movie title trigrams
- Drops users with fewer than `--min-user` ratings (default 20) and movies with
fewer than `--min-item` (default 20)
- Estimates each user's theta via ridge regression on (item_features, rating)
pairs.
- For `--D 2`: objective 1 = scaled rating (r/5), objective 2 = "loved it"
indicator (rating ≥ 4)
- Saves `user_stream` = the sequence of users in time-stamp order, used as the
arrival sequence

## Algorithm notes

- **LinUCB / LinUCB_RP**: standard, with optional random projection.
- **LinUCB_IND / LinUCB_IND_RP**: one model per user, no info sharing.
- **CLUB**: faithful Gentile et al. 2014, with the complete user graph.
- **SCLUB / SCLUB_RP / SCLUB_CA / SCLUB_RP_CA**: faithful Li et al. 2019, with
optional random projection on item features (RP) and our arm-clustering speedup
(CA). RP-only and CA-only variants do nothing else differently from SCLUB.
- **Bandit_Diag**: SCLUB_RP_CA where each user's covariance S_u is approximated
by its diagonal only (memory O(k) per user instead of O(k²)).
- **Bandit_Sketch**: per-user S_u via low-rank Frequent Directions sketch.
- **Bandit_History**: no S_u; keep a sliding window of (item, reward) pairs and
reconstruct stats on demand.
- **Bandit_Window**: SCLUB_RP_CA + garbage collection of inactive users.

## Hyperparameters from the original papers

- `alpha_theta` (SCLUB family + Bandit_*): split-merge sensitivity, default 1.0.
- `alpha_cb` (CLUB): edge deletion sensitivity, default 2.0.
- `lam`, `scale`: ridge λ and UCB exploration scale.

## Hyperparameters added by us (clearly marked as such)

- `k`: projected dimension (RP variants).
- `n_arm_groups`: arm-clustering size; defaults to ⌈√K⌉.
- `arm_recluster_freq`: re-clustering period; default 500.
- `sketch_rank` (Bandit_Sketch): default 5.
- `max_history` (Bandit_History): default 4·k.
- `window_size`, `gc_freq` (Bandit_Window): defaults 10·n_users and 1000.

No `max_clusters` cap is applied: algorithms split as the SCLUB tests dictate.
