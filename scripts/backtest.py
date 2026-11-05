#!/usr/bin/env python3
"""Rolling-origin backtest of baselines (and optionally one candidate) on a single series.

  backtest.py data.csv --date DATE --target Y --horizon 12 [--season 12] [--origins 5]
              [--step N] [--candidate model.py] [--by-horizon]

A candidate file defines forecast(train: pd.Series, h: int) -> array-like of h values.

shortcut: one series, expanding window, no prediction-interval coverage check.
Filter multi-series data to one id first; upgrade when you need per-series or probabilistic evaluation.
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


def load(path, date, target):
    s = pd.read_csv(path, parse_dates=[date]).sort_values(date).set_index(date)[target]
    if s.index.duplicated().any():
        sys.exit("duplicate dates: filter to a single series first")
    if s.isna().any():
        sys.exit("target has missing values: decide how to handle them, they are never filled silently")
    s = s.astype(float)
    freq = pd.infer_freq(s.index) if len(s) >= 3 else None
    if freq is None:
        print("warning: irregular or gappy dates, the season length may not mean what you think", file=sys.stderr)
        return s
    return s.asfreq(freq)


def backtest(y, models, h, m, origins, step):
    """models: {name: fn(train, h, m) -> h values}. Returns (summary, mase_by_horizon, n_origins)."""
    min_train = max(2 * m, 10)
    cuts = sorted(c for c in (len(y) - h - k * step for k in range(origins)) if c >= min_train)
    if not cuts:
        sys.exit(f"series too short: {len(y)} points, need at least {min_train + h}")
    abs_err = {name: [] for name in models}
    scales = []
    for c in cuts:
        train, test = y.iloc[:c], y.iloc[c:c + h].to_numpy()
        x = train.to_numpy()
        scale = np.abs(x[m:] - x[:-m]).mean()  # in-sample seasonal-naive MAE, the MASE denominator
        if scale == 0:
            sys.exit(f"training window ending at row {c} has no variation, MASE is undefined")
        scales.append(scale)
        for name, fn in models.items():
            f = np.asarray(fn(train, h, m), dtype=float)
            if f.shape != (h,):
                sys.exit(f"{name}: expected {h} forecast values, got shape {f.shape}")
            abs_err[name].append(np.abs(f - test))
    scales = np.array(scales)[:, None]
    abs_err = {name: np.array(v) for name, v in abs_err.items()}  # origins x h
    table = pd.DataFrame(
        {name: {"MAE": v.mean(), "MASE": (v / scales).mean()} for name, v in abs_err.items()}
    ).T.sort_values("MASE")
    by_h = pd.DataFrame({name: (v / scales).mean(axis=0) for name, v in abs_err.items()}, index=range(1, h + 1))
    return table, by_h, len(cuts)


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return lambda train, h, m: mod.forecast(train, h)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv")
    p.add_argument("--date", required=True, help="date column name")
    p.add_argument("--target", required=True, help="value column name")
    p.add_argument("--horizon", type=int, required=True, help="periods to forecast")
    p.add_argument("--season", type=int, default=1, help="season length in periods (12 monthly, 7 daily)")
    p.add_argument("--origins", type=int, default=5, help="number of rolling forecast origins")
    p.add_argument("--step", type=int, help="periods between origins (default: horizon)")
    p.add_argument("--candidate", help="python file defining forecast(train, h)")
    p.add_argument("--by-horizon", action="store_true", help="also print MASE per horizon step")
    a = p.parse_args(argv)
    if min(a.season, a.horizon, a.origins) < 1:
        p.error("--season, --horizon and --origins must be >= 1")

    models = {"naive": naive, "drift": drift}
    if a.season > 1:  # with season 1, seasonal naive is identical to naive
        models["seasonal_naive"] = seasonal_naive
    baselines = list(models)
    if a.candidate:
        models["candidate"] = load_candidate(a.candidate)

    table, by_h, k = backtest(load(a.csv, a.date, a.target), models, a.horizon, a.season, a.origins, a.step or a.horizon)
    print(f"origins: {k}, horizon: {a.horizon}, season: {a.season}\n")
    print(table.round(3).to_string())
    if a.by_horizon:
        print("\nMASE by horizon step\n" + by_h.round(3).to_string())
    if a.candidate:
        best = table.loc[baselines, "MASE"].idxmin()
        c, b = table.loc["candidate", "MASE"], table.loc[best, "MASE"]
        verdict = "beats the baseline" if c < b else "does NOT beat the baseline, prefer the baseline"
        print(f"\ncandidate MASE {c:.3f} vs best baseline ({best}) {b:.3f}: {verdict}")
    if k < 5:
        print(f"note: only {k} origins, small gaps between models may be noise")
    return table


if __name__ == "__main__":
    main()
