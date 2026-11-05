---
name: forecast-backtest
description: Use when building, evaluating or comparing time-series forecasts (demand, sales, prices, traffic, energy and similar). Forces baselines first, a rolling-origin backtest, leakage checks and an honest report. Triggers include forecast, backtest, time series, predict the next N periods, compare forecasting models, tahmin, zaman serisi, tahmin modeli karşılaştır, geriye dönük test.
---

# Forecast, then prove it beats a baseline

A model is not "good" until it beats the best baseline on a rolling-origin backtest.
Report that comparison even when it is bad news. Do not jump to a fancy model first.

## Workflow

1. **Frame it.** Pin down, and ask if unclear: horizon, frequency, season length
   (12 monthly, 7 daily, 24 hourly), the target, and the decision the forecast feeds.
   Metric: MASE (scale-free) plus MAE (in real units). Avoid MAPE when the target has
   zeros or near-zeros. For quantile forecasts use pinball loss.
2. **Check the data.** Sort by time, no duplicate timestamps, never fill missing values
   silently, note gaps, and confirm the dates parse the way you think (`--date-format
   %d-%m-%Y` for day-first dates). Leakage check: every feature must be known at forecast time
   (lagged or as-of values only), and scalers/encoders are fit on the training window only.
3. **Baselines first.** Run `scripts/backtest.py` with no candidate. It scores naive,
   seasonal naive and drift over several forecast origins. If the horizon is long or the
   data repeats yearly, also run it with the longest natural season (364 for daily data,
   52 for weekly) and compare MAE across the runs, because MASE is scaled by `--season`
   and cannot be compared between different values. A candidate that beats only the
   short-season baseline has not beaten the best baseline.
4. **Candidate.** Write `forecast(train, h)` in its own file, return `h` values, and run
   the script again with `--candidate`. Start simple (ETS/ARIMA, or a GBM on lag features)
   before anything heavier.
5. **Read the verdict.** The script prints whether the candidate beats the best baseline.
   If it does not, say so and recommend the baseline. If only a few origins fit, small
   gaps are noise: raise `--origins` or shrink `--step`.
6. **Uncertainty.** If the user needs ranges, produce prediction intervals and check their
   empirical coverage on the same backtest. The script does not do this yet.
7. **Report.** Table of MAE and MASE per model, number of origins (and series scored versus
   skipped), MASE by horizon step (`--by-horizon`), and the caveats (gaps, short history,
   structural breaks).

## Usage

```bash
python3 scripts/backtest.py data.csv --date date --target sales --horizon 28 --season 7 --origins 8
python3 scripts/backtest.py data.csv --date date --target sales --horizon 28 --season 7 \
  --candidate my_model.py --by-horizon
# many stores or items: name the columns that identify one series
python3 scripts/backtest.py data.csv --date date --target sales --id store item \
  --horizon 28 --season 7 --max-series 50 --candidate my_model.py
```

Candidate file example (needs `statsmodels`):

```python
# my_model.py
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast(train, h):  # train is a pd.Series with a DatetimeIndex
    return ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7).fit().forecast(h)
```

## Limits

- Many series: `--id` scores each series on its own, then averages MASE with equal weight
  per series (no volume weighting) and counts the series where the candidate beats the best
  baseline. Series that are too short or flat are skipped and counted in the output.
  `--max-series N` takes the first N series in id order, not a random sample.
- Expanding training window only.
- A training window with no variation stops the run, because MASE is undefined there.
- Needs Python 3 with pandas and numpy.
