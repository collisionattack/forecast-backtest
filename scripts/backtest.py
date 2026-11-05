#!/usr/bin/env python3
"""Rolling-origin backtest of baselines (and optionally candidates) on one or many series.

  backtest.py data.csv --date DATE --target Y --horizon 12 [--season 12 [364 ...]] [--origins 5]
              [--step N] [--id COL [COL ...]] [--max-series N] [--date-format FMT]
              [--window N] [--weight volume] [--candidate model.py [more.py ...] [--level 80 95]] [--by-horizon]

A candidate file defines forecast(train: pd.Series, h: int) -> array-like of h values, and for --level also
forecast_interval(train, h, level) -> (lower, upper). Several candidate files are scored on the same windows and
named after their files. With --id, every series is scored on its own and the scores are averaged, with equal
weight or with --weight volume.

shortcut: the volume weight is the series' mean absolute target in the scored windows, in the target's own units.
Upgrade when the weight should come from a business measure such as revenue.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd


def naive(train, h, m):
    return np.repeat(train.iloc[-1], h)


def seasonal_naive(train, h, m):
    return np.resize(train.to_numpy()[-m:], h)  # cycles through the last season


def drift(train, h, m):
    y = train.to_numpy()
    return y[-1] + (y[-1] - y[0]) / (len(y) - 1) * np.arange(1, h + 1)


def load(path, date, target, ids=None, date_format=None):
    """Returns {label: series}: one entry per --id group, or a single entry without --id."""
    df = pd.read_csv(path)
    missing = [c for c in [date, target, *(ids or [])] if c not in df.columns]
    if missing:
        sys.exit(f"not in the CSV: {', '.join(missing)}. Columns are: {', '.join(map(str, df.columns))}")
    try:  # pandas infers the format from the first value and refuses rows that do not match it
        df[date] = pd.to_datetime(df[date], format=date_format)
    except ValueError as e:
        sys.exit(f"cannot parse the dates in column {date}, pass --date-format (for example %d-%m-%Y): "
                 f"{str(e).splitlines()[0].split('. You might')[0]}")
    groups = df.groupby(ids, sort=True, dropna=False) if ids else [("series", df)]
    out, irregular = {}, 0
    for key, g in groups:
        label = "/".join(map(str, key)) if isinstance(key, tuple) else str(key)  # e.g. store/item -> "1/3"
        s = g.sort_values(date).set_index(date)[target]
        if s.index.duplicated().any():
            sys.exit(f"{label}: duplicate dates, pass --id with the columns that identify a series")
        s = s.astype(float)
        freq = pd.infer_freq(s.index) if len(s) >= 3 else None
        irregular += freq is None
        out[label] = s if freq is None else s.asfreq(freq)
    if irregular:
        print(f"warning: {irregular} of {len(out)} series have irregular or gappy dates, the season length may not "
              "mean what you think. If the dates look wrong, check --date-format.", file=sys.stderr)
    return out


class FlatSeries(ValueError):
    """Nothing varied before any origin, so MASE has no scale. mae holds each model's mean absolute error over the origins."""

    def __init__(self, mae):
        super().__init__("no variation anywhere in the history, MASE has no scale")
        self.mae = mae


def backtest(y, models, h, m, origins, step, longest=None, intervals=None, levels=(), cover=None, window=None, volumes=None):
    """models: {name: fn(train, h, m) -> h values}. Returns (summary, mase_by_horizon, n_origins).
    m is the season that scales MASE. longest (default m) is the longest season scored and sets the shortest training
    window, so every model sees the same origins. window trains on the last `window` points instead of everything
    before the origin, and only origins with that many points are used. With intervals {name: fn(train, h, level) ->
    (lower, upper)}, cover[(name, level)] gets one (share of test points inside, mean width / scale) pair per origin.
    volumes gets the
    mean absolute target of each scored test window. The MASE scale is the seasonal-naive error of everything before the
    origin, so a flat rolling window still has one. An origin with no variation anywhere before it is left out. Raises
    FlatSeries, a ValueError that carries each model's MAE, when that holds for every origin. Raises ValueError when this
    series cannot be scored (missing values, too short, non-finite forecast)."""
    if y.isna().any():
        raise ValueError("target has missing values, decide how to handle them, they are never filled silently")
    min_train = max(2 * (longest or m), 10, window or 0)
    cuts = sorted(c for c in (len(y) - h - k * step for k in range(origins)) if c >= min_train)
    if not cuts:
        raise ValueError(f"series too short: {len(y)} points, need at least {min_train + h}")
    def split(c):
        return y.iloc[c - window if window else 0:c], y.iloc[c:c + h].to_numpy()

    def run_models(train, test, c):  # absolute error of every model at origin c
        errors = {}
        for name, fn in models.items():
            f = np.asarray(fn(train, h, m), dtype=float)
            if f.shape != (h,):
                sys.exit(f"{name}: expected {h} forecast values, got shape {f.shape}")
            if not np.isfinite(f).all():
                raise ValueError(f"{name} returned non-finite forecasts at row {c}")
            errors[name] = np.abs(f - test)
        return errors

    abs_err = {name: [] for name in models}
    scales, flat = [], []
    for c in cuts:
        train, test = split(c)
        past = y.iloc[:c].to_numpy()  # the scale reads everything before the origin, not only the window the models train on
        scale = np.abs(past[m:] - past[:-m]).mean()  # in-sample seasonal-naive MAE, the MASE denominator
        if scale == 0:  # nothing varied before this origin, MASE has no scale here
            flat.append(c)
            continue
        scales.append(scale)
        if volumes is not None:
            volumes.append(np.abs(test).mean())
        for name, e in run_models(train, test, c).items():
            abs_err[name].append(e)
        for name, fn in (intervals or {}).items():
            for level in levels:
                lo, hi = (np.asarray(v, dtype=float) for v in fn(train, h, level))
                if lo.shape != (h,) or hi.shape != (h,) or not (np.isfinite(lo).all() and np.isfinite(hi).all()) or (lo > hi).any():
                    raise ValueError(f"{name}: interval at level {level} must be two arrays of {h} finite values with "
                                     f"lower <= upper, row {c}")
                cover.setdefault((name, level), []).append((((lo <= test) & (test <= hi)).mean(), (hi - lo).mean() / scale))
    if not scales:  # flat before every origin: no MASE, but the errors are still worth reporting
        errors = [run_models(*split(c), c) for c in flat]
        raise FlatSeries({name: float(np.mean([e[name] for e in errors])) for name in models})
    scales = np.array(scales)[:, None]
    abs_err = {name: np.array(v) for name, v in abs_err.items()}  # origins x h
    table = pd.DataFrame(
        {name: {"MAE": v.mean(), "MASE": (v / scales).mean()} for name, v in abs_err.items()}
    ).T.sort_values("MASE")
    by_h = pd.DataFrame({name: (v / scales).mean(axis=0) for name, v in abs_err.items()}, index=range(1, h + 1))
    return table, by_h, len(scales)


def beats(table, baselines, name="candidate"):
    """True when the candidate's MASE is strictly below every baseline's."""
    return table.loc[name, "MASE"] < table.loc[baselines, "MASE"].min()


def load_module(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_candidate(path):
    mod = load_module(path)
    return lambda train, h, m: mod.forecast(train.copy(), h)  # train is a slice of the data, an in-place edit would change it


def load_intervals(path):
    """forecast_interval(train, h, level) -> (lower, upper) from the candidate file, or None when it has none."""
    fn = getattr(load_module(path), "forecast_interval", None)
    return fn and (lambda train, h, level: fn(train.copy(), h, level))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv")
    p.add_argument("--date", required=True, help="date column name")
    p.add_argument("--date-format", help="strptime format of the date column, for example %%d-%%m-%%Y")
    p.add_argument("--target", required=True, help="value column name")
    p.add_argument("--horizon", type=int, required=True, help="periods to forecast")
    p.add_argument("--season", type=int, nargs="+", default=[1],
                   help="season length(s) in periods (12 monthly, 7 daily, 364 daily with a yearly pattern). "
                        "Each value adds a seasonal naive scored on the same origins. MASE is scaled by the first value")
    p.add_argument("--origins", type=int, default=5, help="number of rolling forecast origins")
    p.add_argument("--step", type=int, help="periods between origins (default: horizon)")
    p.add_argument("--window", type=int,
                   help="train on the last N points at every origin instead of all earlier data (rolling window), "
                        "at least twice the longest season")
    p.add_argument("--weight", choices=["equal", "volume"], default="equal",
                   help="with --id, average series with equal weight or weighted by each series' mean absolute target "
                        "value in the scored windows")
    p.add_argument("--id", nargs="+", help="column(s) that identify a series, scores are averaged over series")
    p.add_argument("--max-series", type=int, help="score only the first N series in id order, not a random sample")
    p.add_argument("--candidate", nargs="+",
                   help="python file(s) defining forecast(train, h). Several files are scored on the same windows "
                        "and named after the files")
    p.add_argument("--level", type=int, nargs="+",
                   help="prediction interval levels in percent, for example 80 95. Every candidate file that also "
                        "defines forecast_interval(train, h, level) -> (lower, upper) gets an interval report")
    p.add_argument("--by-horizon", action="store_true", help="also print MASE per horizon step")
    a = p.parse_args(argv)
    if min(*a.season, a.horizon, a.origins, a.max_series or 1, a.step or 1) < 1:
        p.error("--season, --horizon, --origins, --step and --max-series must be >= 1")
    if a.window and a.window < max(2 * max(a.season), 10):
        p.error(f"--window must be at least {max(2 * max(a.season), 10)}, twice the longest season")
    if a.weight == "volume" and not a.id:
        p.error("--weight volume needs --id")
    if a.level and not a.candidate:
        p.error("--level needs --candidate")
    if a.level and not all(0 < v < 100 for v in a.level):
        p.error("--level values must be between 1 and 99")

    models = {"naive": naive, "drift": drift}
    for k in a.season:
        if k > 1:  # with season 1, seasonal naive is identical to naive
            name = "seasonal_naive" if len(a.season) == 1 else f"seasonal_naive_{k}"
            models[name] = lambda train, h, m, k=k: seasonal_naive(train, h, k)
    baselines = list(models)
    candidates, intervals = [], {}
    if a.candidate:
        candidates = ["candidate"] if len(a.candidate) == 1 else [pathlib.Path(f).stem for f in a.candidate]
        if len(set(candidates)) != len(candidates) or set(candidates) & set(baselines):
            p.error("candidate files need different names, and none may be named like a baseline")
        for name, path in zip(candidates, a.candidate):
            models[name] = load_candidate(path)
        if a.level:
            intervals = {name: fn for name, path in zip(candidates, a.candidate) if (fn := load_intervals(path))}
            if not intervals:
                p.error(f"{', '.join(a.candidate)} define no forecast_interval(train, h, level)")

    series = load(a.csv, a.date, a.target, a.id, a.date_format)
    if a.max_series:
        series = dict(list(series.items())[:a.max_series])
    results, skipped, covers, volumes, flats = [], {}, [], [], {}
    for label, y in series.items():
        cov, vol = {}, []
        try:
            results.append(backtest(y, models, a.horizon, a.season[0], a.origins, a.step or a.horizon,
                                    max(a.season), intervals, a.level or (), cov, a.window, vol))
            covers.append(cov)
            volumes.append(np.mean(vol))
        except FlatSeries as e:  # never varied: no MASE for it, its MAE goes on a line of its own
            flats[label] = e.mae
        except ValueError as e:
            skipped[label] = str(e)

    def show_skipped_and_flat():
        for label, why in list(skipped.items())[:3]:
            print(f"skipped {label}: {why}")
        if len(skipped) > 3:
            print(f"... and {len(skipped) - 3} more skipped")
        if flats:
            print(f"\nflat series: {len(flats)} with no variation in their history, so MASE has no scale. "
                  "mean absolute error per model:")
            print(pd.DataFrame(flats).mean(axis=1).sort_values().round(3).to_string())

    if not results and not flats:
        label, why = next(iter(skipped.items()))
        sys.exit(f"no series could be scored, for example {label}: {why}")
    if not results:  # everything was flat or skipped, there is no MASE table to print
        if a.id:
            print(f"series: 0 scored, {len(skipped)} skipped, {len(flats)} flat")
        show_skipped_and_flat()
        return None
    tables, by_hs, ks = zip(*results)
    w = np.array(volumes) if a.weight == "volume" else np.ones(len(results))
    if w.sum() == 0:
        sys.exit("every scored series has zero volume in the scored windows, there is nothing to weight by")
    table = (sum(t.sort_index() * wi for t, wi in zip(tables, w)) / w.sum()).sort_values("MASE")
    by_h = sum(b * wi for b, wi in zip(by_hs, w)) / w.sum()

    origins = str(min(ks)) if min(ks) == max(ks) else f"{min(ks)}-{max(ks)}"
    print(f"origins: {origins}, horizon: {a.horizon}, season: {' '.join(map(str, a.season))}"
          + (f", window: {a.window}" if a.window else ""))
    if a.id:
        print(f"series: {len(results)} scored, {len(skipped)} skipped" + (f", {len(flats)} flat" if flats else "")
              + ", scores are averages over series"
              + (" weighted by mean target value in the scored windows" if a.weight == "volume" else ""))
    print("\n" + table.round(3).to_string())
    show_skipped_and_flat()
    if a.by_horizon:
        print("\nMASE by horizon step\n" + by_h.round(3).to_string())
    for name in candidates:
        best = table.loc[baselines, "MASE"].idxmin()
        c, b = table.loc[name, "MASE"], table.loc[best, "MASE"]
        verdict = "beats the baseline" if c < b else "does NOT beat the baseline, prefer the baseline"
        print(f"\n{name} MASE {c:.3f} vs best baseline ({best}) {b:.3f}: {verdict}")
        if a.id:
            print(f"{name} beats the best baseline in {sum(beats(t, baselines, name) for t in tables)} "
                  f"of {len(tables)} series")
    for name in intervals:
        print(f"\nprediction intervals of {'the candidate' if name == 'candidate' else name}, "
              f"over {len(results)} series and {sum(ks)} origins")
        print("level  coverage  width (MASE units)")
        for v in a.level:
            pairs = [q for c in covers for q in c[(name, v)]]
            print(f"{v:>5}  {np.mean([q[0] for q in pairs]):>8.1%}  {np.mean([q[1] for q in pairs]):>18.2f}")
    if min(ks) < 5:
        print(f"note: only {min(ks)} origins, small gaps between models may be noise")
    return table


if __name__ == "__main__":
    main()
