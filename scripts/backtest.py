#!/usr/bin/env python3
"""Rolling-origin backtest of baselines (and optionally one candidate) on one or many series.

  backtest.py data.csv --date DATE --target Y --horizon 12 [--season 12] [--origins 5]
              [--step N] [--id COL [COL ...]] [--max-series N] [--date-format FMT]
              [--candidate model.py] [--by-horizon]

A candidate file defines forecast(train: pd.Series, h: int) -> array-like of h values.
With --id, every series is scored on its own and the scores are averaged.

shortcut: expanding window, no prediction-interval coverage check, equal weight per series.
Upgrade when you need probabilistic evaluation or volume-weighted scores.
"""
import argparse
import importlib.util
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
        if s.isna().any():
            sys.exit(f"{label}: target has missing values, decide how to handle them, they are never filled silently")
        s = s.astype(float)
        freq = pd.infer_freq(s.index) if len(s) >= 3 else None
        irregular += freq is None
        out[label] = s if freq is None else s.asfreq(freq)
    if irregular:
        print(f"warning: {irregular} of {len(out)} series have irregular or gappy dates, the season length may not "
              "mean what you think. If the dates look wrong, check --date-format.", file=sys.stderr)
    return out


def backtest(y, models, h, m, origins, step):
    """models: {name: fn(train, h, m) -> h values}. Returns (summary, mase_by_horizon, n_origins).
    Raises ValueError when this series cannot be scored (too short, flat window, non-finite forecast)."""
    min_train = max(2 * m, 10)
    cuts = sorted(c for c in (len(y) - h - k * step for k in range(origins)) if c >= min_train)
    if not cuts:
        raise ValueError(f"series too short: {len(y)} points, need at least {min_train + h}")
    abs_err = {name: [] for name in models}
    scales = []
    for c in cuts:
        train, test = y.iloc[:c], y.iloc[c:c + h].to_numpy()
        x = train.to_numpy()
        scale = np.abs(x[m:] - x[:-m]).mean()  # in-sample seasonal-naive MAE, the MASE denominator
        if scale == 0:
            raise ValueError(f"training window ending at row {c} has no variation, MASE is undefined")
        scales.append(scale)
        for name, fn in models.items():
            f = np.asarray(fn(train, h, m), dtype=float)
            if f.shape != (h,):
                sys.exit(f"{name}: expected {h} forecast values, got shape {f.shape}")
            if not np.isfinite(f).all():
                raise ValueError(f"{name} returned non-finite forecasts at row {c}")
            abs_err[name].append(np.abs(f - test))
    scales = np.array(scales)[:, None]
    abs_err = {name: np.array(v) for name, v in abs_err.items()}  # origins x h
    table = pd.DataFrame(
        {name: {"MAE": v.mean(), "MASE": (v / scales).mean()} for name, v in abs_err.items()}
    ).T.sort_values("MASE")
    by_h = pd.DataFrame({name: (v / scales).mean(axis=0) for name, v in abs_err.items()}, index=range(1, h + 1))
    return table, by_h, len(cuts)


def beats(table, baselines):
    """True when the candidate's MASE is strictly below every baseline's."""
    return table.loc["candidate", "MASE"] < table.loc[baselines, "MASE"].min()


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return lambda train, h, m: mod.forecast(train.copy(), h)  # train is a slice of the data, an in-place edit would change it


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv")
    p.add_argument("--date", required=True, help="date column name")
    p.add_argument("--date-format", help="strptime format of the date column, for example %%d-%%m-%%Y")
    p.add_argument("--target", required=True, help="value column name")
    p.add_argument("--horizon", type=int, required=True, help="periods to forecast")
    p.add_argument("--season", type=int, default=1, help="season length in periods (12 monthly, 7 daily)")
    p.add_argument("--origins", type=int, default=5, help="number of rolling forecast origins")
    p.add_argument("--step", type=int, help="periods between origins (default: horizon)")
    p.add_argument("--id", nargs="+", help="column(s) that identify a series, scores are averaged over series")
    p.add_argument("--max-series", type=int, help="score only the first N series in id order, not a random sample")
    p.add_argument("--candidate", help="python file defining forecast(train, h)")
    p.add_argument("--by-horizon", action="store_true", help="also print MASE per horizon step")
    a = p.parse_args(argv)
    if min(a.season, a.horizon, a.origins, a.max_series or 1, a.step or 1) < 1:
        p.error("--season, --horizon, --origins, --step and --max-series must be >= 1")

    models = {"naive": naive, "drift": drift}
    if a.season > 1:  # with season 1, seasonal naive is identical to naive
        models["seasonal_naive"] = seasonal_naive
    baselines = list(models)
    if a.candidate:
        models["candidate"] = load_candidate(a.candidate)

    series = load(a.csv, a.date, a.target, a.id, a.date_format)
    if a.max_series:
        series = dict(list(series.items())[:a.max_series])
    results, skipped = [], {}
    for label, y in series.items():
        try:
            results.append(backtest(y, models, a.horizon, a.season, a.origins, a.step or a.horizon))
        except ValueError as e:
            skipped[label] = str(e)
    if not results:
        label, why = next(iter(skipped.items()))
        sys.exit(f"no series could be scored, for example {label}: {why}")
    tables, by_hs, ks = zip(*results)
    table = pd.concat(tables).groupby(level=0).mean().sort_values("MASE")
    by_h = sum(by_hs) / len(by_hs)

    origins = str(min(ks)) if min(ks) == max(ks) else f"{min(ks)}-{max(ks)}"
    print(f"origins: {origins}, horizon: {a.horizon}, season: {a.season}")
    if a.id:
        print(f"series: {len(results)} scored, {len(skipped)} skipped, scores are averages over series")
    print("\n" + table.round(3).to_string())
    for label, why in list(skipped.items())[:3]:
        print(f"skipped {label}: {why}")
    if len(skipped) > 3:
        print(f"... and {len(skipped) - 3} more skipped")
    if a.by_horizon:
        print("\nMASE by horizon step\n" + by_h.round(3).to_string())
    if a.candidate:
        best = table.loc[baselines, "MASE"].idxmin()
        c, b = table.loc["candidate", "MASE"], table.loc[best, "MASE"]
        verdict = "beats the baseline" if c < b else "does NOT beat the baseline, prefer the baseline"
        print(f"\ncandidate MASE {c:.3f} vs best baseline ({best}) {b:.3f}: {verdict}")
        if a.id:
            print(f"candidate beats the best baseline in {sum(beats(t, baselines) for t in tables)} "
                  f"of {len(tables)} series")
    if min(ks) < 5:
        print(f"note: only {min(ks)} origins, small gaps between models may be noise")
    return table


if __name__ == "__main__":
    main()
