#!/usr/bin/env python3
"""Predict the top 5 new pump.fun coins by market cap, 24 hours out.

Reads $MAKEFUN_INPUT_DIR/history.ndjson + task.json (defaults /input) and
writes $MAKEFUN_OUTPUT_DIR/forecast.json (default /output).

Model, in one paragraph: every coin's log10 market cap (in SOL) is projected
to each of the seven answer reads from task.json with a momentum / trend
model -- its own recent per-hour log-cap slope, shrunk toward a
level-dependent drift prior estimated from the whole history (in this data
the current leaders keep pumping while mid-pack coins bleed), with the
growth saturating over a ~1.5 h time constant so a hot streak cannot be
extrapolated into the stratosphere.  Each coin's projected cap carries
uncertainty that grows with the forecast horizon, and a fixed-seed Monte
Carlo over the projected caps gives calibrated place probabilities and the
distribution of each place's size; the place probabilities are averaged over
the seven reads (one forecast is scored at every read).  The phantom count
and the trend-size weight both ramp up with the horizon (time constants
0.5 h): at a 10-minute horizon known coins still hold the chart, while
30-70 minutes out newcomers routinely take top places, and at the final's
7-13 hour horizon the newcomer threat is fully saturated.

Two features cover the coins no forecast can name -- the ones born after
the history ends, which took a third of the practice's top-5 slots.  First,
phantom newcomers compete inside the Monte Carlo: their caps are drawn
around a per-place trend fitted to the history (log-cap of each place
against log-age since the opening, which captures the steady arrival of
new winners pushing every place's cap up), so they steal place mass from
known coins in proportion to how likely a newcomer is to take that place,
and they lift the size distribution the same way.  Second, each place's
size distribution is a mixture of the Monte Carlo's k-th-largest
distribution and the trend's own wide normal, because the k-th order
statistic of known coins alone is far too narrow and too low at long
horizons.  Named place mass is capped so at least 15% of every place
always stays unnamed.

Python 3.13, standard library only, no network, no model calls,
deterministic: the same input gives byte-identical output.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
N_SAMPLES = 6000          # Monte Carlo draws per answer read
TAU_HOURS = 1.5           # growth-saturation time constant (hours)
SIGMA0 = 0.12             # base log10-cap uncertainty
SIGMA_HOURLY = 0.22       # extra log10-cap uncertainty per sqrt(hour) of horizon
NAMED_SHARE = 1.0         # phantoms already leave mass unnamed; no extra scaling
MAX_NAMED_SUM = 0.85      # cap on a place's named probability (rest stays unnamed)
MIN_P = 0.0008            # smallest named place probability kept
MIN_SIZE_P = 0.0005       # smallest named size-bin probability kept
MAX_ROWS_PER_PLACE = 100  # RULES.txt cap
STALE_DECAY_PER_HOUR = 0.15  # extra log-cap decay for coins not seen recently
OWN_SLOPE_CLIP = 1.0      # own recent slope clipped to +-1 log10 per hour
AGE_DECAY_PER_HOUR = 0.06  # log-cap decay per hour once a coin is past its pump
AGE_DECAY_START_HOURS = 2.0  # age (hours) after which the decay starts
AGE_DECAY_MAX = 0.6        # cap on the total age decay
N_PHANTOM = 10            # phantom newcomer coins at a long horizon (scaled by horizon)
PHANTOM_SIGMA = 0.40
PHANTOM_SPREAD = 0.12     # log10 spacing between successive phantoms' means
PHANTOM_TAU_HOURS = 0.5
TREND_SIZE_WEIGHT = 0.5   # weight of the trend normal in each place's sizes
TREND_SIZE_TAU_HOURS = 0.5
TREND_SIGMA = 0.40        # sd of the trend normal for a place's size
TREND_SLOPE_MAX = 1.2     # cap on the fitted log-age trend slope per place


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def logcap_sol(lamports):
    """log10 of a market cap in SOL, as the scorer takes it."""
    return math.log10(int(lamports)) - 9.0


def load_input():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as fh:
        task = json.load(fh)
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            mark = json.loads(line)
            if mark.get("ok") and mark.get("rows"):
                marks.append(mark)
    if not marks:
        raise ValueError("the history holds no ok mark with rows")
    return task, marks


def build_trajectories(marks):
    """mint -> {"points": [(t, logcap)], "created_at": ts, "last_t": ts}."""
    coins = {}
    for mark in marks:
        t = parse_ts(mark["at"])
        for row in mark["rows"]:
            mint = row["mint"]
            entry = coins.setdefault(
                mint, {"points": [], "created_at": parse_ts(row["created_at"])}
            )
            entry["points"].append((t, logcap_sol(row["market_cap_lamports"])))
    for entry in coins.values():
        entry["points"].sort()
        entry["last_t"] = entry["points"][-1][0]
        entry["last_logcap"] = entry["points"][-1][1]
    return coins


def level_drift_prior(coins):
    """Median-ish hourly log-cap drift as a function of the coin's level.

    Estimated from every consecutive pair of marks a coin appears in,
    bucketed by its starting log-cap, shrunk toward the global mean when a
    bucket is thin.  Returns (bin_edges, drift_per_bin, global_drift).
    """
    edges = [0.0, 1.5, 2.0, 2.5, 3.0, 99.0]
    sums = [0.0] * (len(edges) - 1)
    counts = [0] * (len(edges) - 1)
    gsum = 0.0
    gcount = 0
    for entry in coins.values():
        pts = entry["points"]
        for (t1, c1), (t2, c2) in zip(pts, pts[1:]):
            dt_h = (t2 - t1) / 3600.0
            if dt_h < 0.05:
                continue
            drift = (c2 - c1) / dt_h
            drift = max(-1.5, min(1.5, drift))
            gsum += drift
            gcount += 1
            for i in range(len(edges) - 1):
                if edges[i] <= c1 < edges[i + 1]:
                    sums[i] += drift
                    counts[i] += 1
                    break
    global_drift = gsum / gcount if gcount else 0.0
    drifts = []
    for i in range(len(edges) - 1):
        if counts[i]:
            drifts.append((sums[i] + 30.0 * global_drift) / (counts[i] + 30.0))
        else:
            drifts.append(global_drift)
    return edges, drifts, global_drift


def prior_drift(level, edges, drifts):
    for i in range(len(edges) - 1):
        if edges[i] <= level < edges[i + 1]:
            return drifts[i]
    return drifts[-1]


def project_coins(coins, edges, drifts, read_ts, last_mark_ts, max_logcap):
    """Project every coin's log-cap to one answer read.

    Returns a list of (mint, mean_logcap, sigma) for coins worth simulating.
    """
    ceiling = max_logcap + 1.0
    out = []
    for mint, entry in coins.items():
        pts = entry["points"]
        last_t, last_c = pts[-1]
        dt_h = max(0.0, (read_ts - last_t) / 3600.0)
        # Own recent slope from the last two marks, clipped; shrunk toward
        # the level prior.  One observation only -> prior alone.
        own = None
        if len(pts) >= 2:
            (t_prev, c_prev) = pts[-2]
            span_h = (last_t - t_prev) / 3600.0
            if span_h >= 0.08:
                own = max(-OWN_SLOPE_CLIP,
                          min(OWN_SLOPE_CLIP, (last_c - c_prev) / span_h))
        prior = prior_drift(last_c, edges, drifts)
        slope = (0.65 * own + 0.35 * prior) if own is not None else prior
        # Saturating growth: slope * TAU * (1 - exp(-dt/TAU)).
        mean = last_c + slope * TAU_HOURS * (1.0 - math.exp(-dt_h / TAU_HOURS))
        # Coins fade after their pump: once a coin is past
        # AGE_DECAY_START_HOURS old its cap decays, so a leader from the
        # history cannot be projected to hold the top forever.  This bites
        # only at the final's 7-13 h horizon; practice reads are too soon.
        age_read_h = (read_ts - entry["created_at"]) / 3600.0
        mean -= min(AGE_DECAY_MAX,
                    AGE_DECAY_PER_HOUR * max(0.0, age_read_h - AGE_DECAY_START_HOURS))
        # Coins that fell out of the chart are fading: extra decay + noise.
        stale_h = max(0.0, (last_mark_ts - last_t) / 3600.0)
        if stale_h > 0.5:
            mean -= min(0.6, STALE_DECAY_PER_HOUR * (stale_h - 0.5))
        mean = min(ceiling, max(0.5, mean))
        sigma = math.sqrt(
            SIGMA0 ** 2 + (SIGMA_HOURLY ** 2) * dt_h + (0.10 * stale_h) ** 2
        )
        out.append((mint, mean, sigma))
    # Keep only coins that could plausibly reach the top: within 1.2 log of
    # the best projection, at most 160 of them.
    out.sort(key=lambda item: (-item[1], item[0]))
    if out:
        best = out[0][1]
        out = [item for item in out if item[1] >= best - 1.2][:160]
    return out


def bin_of(value, edges):
    """Size bin 0..len(edges) for a log10 cap in SOL (RULES.txt)."""
    if value < edges[0]:
        return 0
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def place_trends(marks, t0_ts, places):
    """Per-place log-cap trend against log-age since the opening.

    For each place k, regress the log10 cap of the coin at #k in each mark
    on log10(age in minutes).  New winners arrive steadily, so every
    place's cap drifts up with age; the log-age form makes that drift
    self-limiting as the market matures.  Returns a list of (intercept,
    slope) per place; a place with too little data falls back to a flat
    trend at its last observed cap.
    """
    series = {k: [] for k in range(1, places + 1)}
    for mark in marks:
        age_min = max(1.0, (parse_ts(mark["at"]) - t0_ts) / 60.0)
        chart = sorted(mark["rows"], key=lambda r: r["rank"])
        for k in range(1, places + 1):
            if k - 1 < len(chart):
                series[k].append(
                    (math.log10(age_min),
                     logcap_sol(chart[k - 1]["market_cap_lamports"])))
    trends = []
    for k in range(1, places + 1):
        pts = series[k]
        if len(pts) < 2:
            level = pts[0][1] if pts else 2.0
            trends.append((level, 0.0))
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        slope = 0.0
        if sxx > 1e-9:
            slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
        slope = max(0.0, min(TREND_SLOPE_MAX, slope))
        trends.append((my - slope * mx, slope))
    return trends


def normal_bin_p(mean, sd, edges, b):
    """P(a Normal(mean, sd) falls in size bin b)."""
    lo = -math.inf if b == 0 else edges[b - 1]
    hi = math.inf if b == len(edges) else edges[b]
    cdf = lambda x: 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0)))) \
        if x not in (-math.inf, math.inf) else (0.0 if x == -math.inf else 1.0)
    return cdf(hi) - cdf(lo)


def main():
    task, marks = load_input()
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    reads = [parse_ts(r) for r in task["reads"]]
    last_mark_ts = max(parse_ts(m["at"]) for m in marks)

    coins = build_trajectories(marks)
    dedges, drifts, _ = level_drift_prior(coins)
    max_logcap = max(e["last_logcap"] for e in coins.values())
    t0_ts = parse_ts(task["t0"])
    trends = place_trends(marks, t0_ts, places)

    rng = random.Random(SEED)
    # place_counts[place][mint], size_hists[read][place][bin]; place counts
    # are averaged over reads, size mixtures are averaged per read.
    place_counts = {k: {} for k in range(1, places + 1)}
    size_hists = []
    trend_weights = []
    mean_logcap = {}  # mint -> mean projected log-cap across reads (tie-break)
    trend_at_read = []  # per read, the trend's predicted log-cap per place

    for read_ts in reads:
        projected = project_coins(coins, dedges, drifts, read_ts, last_mark_ts,
                                  max_logcap)
        for mint, mean, _sigma in projected:
            mean_logcap[mint] = mean_logcap.get(mint, 0.0) + mean / len(reads)
        dt_h = max(0.0, (read_ts - last_mark_ts) / 3600.0)
        age_min = max(1.0, (read_ts - t0_ts) / 60.0)
        trend = [a + b * math.log10(age_min) for (a, b) in trends]
        trend_at_read.append(trend)
        # The newcomer threat grows with the horizon: a coin born just
        # after the cut has had no time to grow at a 10-minute horizon, but
        # over 7-13 hours newcomers routinely take top places.
        n_phantom = max(1, round(
            N_PHANTOM * (1.0 - math.exp(-dt_h / PHANTOM_TAU_HOURS))))
        trend_weights.append(TREND_SIZE_WEIGHT *
                             (1.0 - math.exp(-dt_h / TREND_SIZE_TAU_HOURS)))
        # Phantom newcomers: coins born after the history ends, centred on
        # the trend's place-1 prediction and spread below it.  They compete
        # in the Monte Carlo but are never named in the output.
        entries = [(mean, sigma, mint) for mint, mean, sigma in projected]
        for j in range(n_phantom):
            entries.append((trend[0] - j * PHANTOM_SPREAD, PHANTOM_SIGMA, None))
        size_hist = {k: {} for k in range(1, places + 1)}
        for _ in range(N_SAMPLES):
            drawn = [(rng.gauss(mean, sigma), mint)
                     for mean, sigma, mint in entries]
            drawn.sort(key=lambda item: (-item[0], item[1] or ""))
            for k in range(1, places + 1):
                if k - 1 < len(drawn):
                    value, mint = drawn[k - 1]
                    if mint is not None:
                        place_counts[k][mint] = place_counts[k].get(mint, 0) + 1
                    b = bin_of(value, edges)
                    size_hist[k][b] = size_hist[k].get(b, 0) + 1
        size_hists.append(size_hist)

    total_draws = N_SAMPLES * len(reads)
    probabilities = []
    for k in range(1, places + 1):
        ranked = sorted(place_counts[k].items(), key=lambda kv: (-kv[1], kv[0]))
        rows = []
        for mint, count in ranked[:MAX_ROWS_PER_PLACE]:
            p = NAMED_SHARE * count / total_draws
            p = math.floor(p * 1_000_000) / 1_000_000
            if p >= MIN_P:
                rows.append({"mint": mint, "place": k, "p": p})
        # Cap the named mass so part of every place stays unnamed.
        total = sum(r["p"] for r in rows)
        if total > MAX_NAMED_SUM:
            scale = MAX_NAMED_SUM / total
            rows = [{"mint": r["mint"], "place": k,
                     "p": math.floor(r["p"] * scale * 1_000_000) / 1_000_000}
                    for r in rows]
            rows = [r for r in rows if r["p"] >= MIN_P]
        probabilities.extend(rows)

    sizes = []
    for k in range(1, places + 1):
        # Mixture, averaged over reads: the Monte Carlo's k-th-largest
        # distribution (which now includes phantom newcomers) and the
        # trend's wide normal, weighted by horizon.
        for b in range(len(edges) + 1):
            ps = []
            for r in range(len(reads)):
                w = trend_weights[r]
                mc_p = size_hists[r][k].get(b, 0) / N_SAMPLES
                trend_p = normal_bin_p(trend_at_read[r][k - 1], TREND_SIGMA,
                                       edges, b)
                ps.append((1.0 - w) * mc_p + w * trend_p)
            p = sum(ps) / len(ps)
            p = math.floor(p * 1_000_000) / 1_000_000
            if p >= MIN_SIZE_P:
                sizes.append({"place": k, "bin": b, "p": p})

    # top5: highest expected agreement weight, tie-break by mean projection.
    weight = {}
    for row in probabilities:
        w = (places + 1 - row["place"]) * row["p"]
        weight[row["mint"]] = weight.get(row["mint"], 0.0) + w
    top5 = sorted(weight, key=lambda m: (-weight[m], -mean_logcap.get(m, 0.0), m))
    top5 = top5[:places]
    # Safety: exactly `places` distinct mints, fall back to best projections.
    if len(top5) < places:
        extra = sorted(mean_logcap, key=lambda m: (-mean_logcap[m], m))
        for mint in extra:
            if mint not in top5:
                top5.append(mint)
            if len(top5) == places:
                break

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))
    sizes.sort(key=lambda r: (r["place"], r["bin"]))
    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
