# Running experiments and the parameter sweep

## 1. A single experiment (reproduces the thesis primary result)

```bash
python run_full_experiment.py
```

Runs the thesis primary configuration (Q=0.5, k=3.0, gamma=1.0, w=300, tau=0.08,
c=7, N=20000, seed=42) and writes the 3 plots + 6 CSVs + `experiment_config.json`
into a timestamped folder under `experiments/`.

Override any parameter from the command line:

```bash
python run_full_experiment.py --Q 0.4 --tau 0.085 --warmup 350 --seed 7
python run_full_experiment.py --no-plots          # metrics + CSVs only (faster)
```

The original script is preserved as `run_full_experiment_ORIGINAL_backup.py`.

## 2. The full grid + multi-seed sweep

```bash
python run_sweep.py --quick --yes     # ~8 runs, a fast sanity check first
python run_sweep.py                    # full grid x 5 seeds (asks to confirm)
python run_sweep.py --yes              # full grid, no prompt
```

Edit the grid at the top of `run_sweep.py`:

```python
GRID = {
    "Q":                  [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6],
    "tau":                [0.06, 0.07, 0.08, 0.085, 0.09, 0.10, 0.12],
    "warmup":             [300, 400],          # add 200, 250 for more
    "k":                  [3.0, 6.0],          # add 2.0, 4.0 for more
    "gamma":              [0.6, 0.7, 0.8, 1.0],
    "confirmation_steps": [5, 7, 10],          # confirmation window c
}
SEEDS = [42, 7, 123, 2024, 99]
```

The default grid is 9 x 7 x 2 x 2 x 4 x 3 = 3,024 combinations x 5 seeds =
**15,120 runs (~2.2 hours)** on this project's hardware. `tau` and `gamma` get
the finest resolution because they are the strongest levers for the
detection / false-alarm trade-off; `Q` has extra points near the useful region.
The time estimate printed before the run is calibrated to a real run here (it
matched the observed ~20 min for the earlier 2,520-run grid). Widen the `warmup`
or `k` lists to go bigger; trim any list to go faster.

`c` (confirmation_steps) is a false-alarm lever in principle (larger `c` requires
more consecutive exceedances), though the first sweep showed its effect here is
modest compared with `tau` and `gamma`.

### Outputs (in `sweeps/sweep_<timestamp>/`)

| File | Contents |
|------|----------|
| `sweep_results.csv` | one row per (combination x seed) run — all metrics |
| `sweep_aggregated.csv` | seed-averaged results: mean / std / count per combination |
| `sweep_best_configs.csv` | every combination ranked: **most drifts detected, then fewest false alarms**, then fastest. Top row = best detector with lowest false alarms |
| `sweep_fig1_detection_vs_Q.png` | drifts detected vs Q (mean ± std over seeds) |
| `sweep_fig2_delay_vs_Q.png` | mean detection delay vs Q |
| `sweep_fig3_alarms_vs_tau.png` | total / unmatched (false) alarms vs threshold tau |
| `sweep_fig4_detection_vs_warmup.png` | drifts detected vs warm-up length |
| `sweep_fig4b_detection_falsealarms_vs_c.png` | detection **and false alarms** vs confirmation window c |
| `sweep_fig5_heatmap_Q_tau.png` | detection heatmap over (Q, tau) |
| `sweep_fig6_seed_variability.png` | seed spread of delay at the primary config |
| `sweep_fig7_detection_vs_falsealarms.png` | **detection vs false-alarm trade-off** — every config; the green-ringed point detects the most drifts with the fewest false alarms |

### False alarms

Each run records `total_alarms`, `matched_alarms`, and `unmatched_alarms`.
**`unmatched_alarms` is the false-alarm count** (alarms that do not correspond to
any true drift); `unmatched_ratio` is its share of all alarms. To find the
parameter set that detects every drift with the fewest false alarms, open
`sweep_best_configs.csv` (top rows) or look at the green-ringed point in
`sweep_fig7_detection_vs_falsealarms.png`.

`sweep_results.csv` is saved incrementally, so a long sweep is crash-safe.

## 3. Why multiple seeds

Detection delays for the borderline variance drifts (events 10 and 11) sit very
close to the ISE threshold, so the exact delay shifts by a few samples with the
noise realisation. Averaging several seeds and reporting **mean ± std** gives
robust numbers for the thesis instead of one brittle run. `sweep_aggregated.csv`
already contains these mean/std values, ready to drop into Tables 5.4–5.5.

## Notes

- Metric column `delay_mean` is the **overall** mean delay across all detected
  drifts. Per-drift-type averages use the `delaytype_*` columns
  (`delaytype_mean`, `delaytype_variance`, ...).
- The density evaluation was vectorised (a precomputed Hermite basis matrix);
  results are bit-identical to the original per-point loop but ~20x faster.
