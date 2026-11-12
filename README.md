# pumpfun-top5-forecast-01 — momentum + newcomer-aware forecast

Forecasts the top 5 new pump.fun coins by market cap at each of the seven
answer reads, for the make.fun competition `pumpfun-top5-forecast-01`.

## How it works

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads `$MAKEFUN_INPUT_DIR/history.ndjson` + `task.json` and writes
`$MAKEFUN_OUTPUT_DIR/forecast.json` with exactly
`{probabilities, top5, sizes}` per the competition rules. All times (the
opening, the history cut, the seven answer reads) come from `task.json` —
nothing is hardcoded — so the same code runs the 2-hour practice rehearsal
and the 24-hour final.

The model, briefly:

- **Projection.** Each coin's log10 market cap (in SOL) is projected to
  each answer read from its own recent per-hour log-cap slope, shrunk
  toward a level-dependent drift prior estimated from the whole history
  (in this data the current leaders keep pumping while mid-pack coins
  bleed), with growth saturating over ~1.5 h and a slow decay once a coin
  is past ~2 h old, so a leader cannot be projected to hold the top
  forever.
- **Newcomers.** Coins born after the history ends took a third of the
  practice's top-5 slots, and no forecast can name them. Phantom
  newcomers compete inside a fixed-seed Monte Carlo around a per-place
  trend (log-cap of each place against log-age since the opening), so
  they steal place mass in proportion to the newcomer threat — which
  ramps up with the horizon (0.5 h time constant) and is fully saturated
  at the final's 7–13 h horizon — and lift the size distribution the
  same way.
- **Sizes.** Each place's size distribution mixes the Monte Carlo's
  k-th-largest distribution with the trend's wide normal, because the
  order statistic of known coins alone is far too narrow and too low at
  long horizons.
- Place probabilities are averaged over the seven reads (one forecast is
  scored at every read) and each place's named mass is capped at 0.85,
  so part of every place always stays unnamed for the coins not yet born.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json in ~1 s
```

## Local validation (practice data)

The kit ships no answer file, so the practice score cannot be computed
locally. Instead the entry was validated on three holdouts: truncate the
history at a cut, forecast the later marks as if they were answer reads,
and score order + size log scores against them, with the house baseline
(persistence) scored the same way.

| holdout | horizon | this entry | house baseline |
|---------|---------|-----------|----------------|
| A (cut 20:40) | 10 min | -25.34 | -29.87 |
| B (cut 20:30) | 20–30 min | -25.67 | -33.48 |
| C (cut 20:20) | 30–50 min | -37.71 | -49.96 |

(Even spread scores -45.06; the house baseline scores -40.41 on the full
practice.) The entry beats the baseline on every holdout, by +4.5 to
+12.2. Its full-practice top 5 also names MIHAO VI and Google, which
finished #2 and #3 at the practice's last answer read.
