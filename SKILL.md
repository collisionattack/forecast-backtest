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
   data repeats yearly, give every natural season in one run (`--season 7 364` for daily
   data). Each value adds a seasonal naive, all are scored on the same origins, and MASE is
   scaled by the first value. A longer season needs a longer history, so short series lose
   origins (see the `origins:` line) or are skipped. A candidate that beats only the
   short-season baseline has not beaten the best baseline.
4. **Candidate.** Write `forecast(train, h)` in its own file, return `h` values, and run
   the script again with `--candidate`. Start simple (ETS/ARIMA, or a GBM on lag features)
   before anything heavier.
5. **Read the verdict.** The script prints whether the candidate beats the best baseline.
   If it does not, say so and recommend the baseline. If only a few origins fit, small
   gaps are noise: raise `--origins` or shrink `--step`.
6. **Uncertainty.** If the user needs ranges, add `forecast_interval(train, h, level)` to the
   candidate file, returning `(lower, upper)`, and run with `--level 80 95`. The script prints
   the share of test points inside each interval and the mean width. Coverage far below the
   level means the ranges are too narrow, far above means too wide.
7. **Report.** Table of MAE and MASE per model, number of origins (and series scored versus
   skipped), MASE by horizon step (`--by-horizon`), and the caveats (gaps, short history,
   structural breaks).

## Usage

```bash
python3 scripts/backtest.py data.csv --date date --target sales --horizon 28 --season 7 --origins 8
python3 scripts/backtest.py data.csv --date date --target sales --horizon 28 --season 7 \
  --candidate my_model.py --by-horizon
# long horizon: give the yearly season too, every baseline is scored on the same origins
python3 scripts/backtest.py data.csv --date date --target sales --horizon 90 --season 7 364
# interval coverage: my_model.py also defines forecast_interval(train, h, level), see examples/ets_monthly.py
python3 scripts/backtest.py data.csv --date date --target sales --horizon 28 --season 7 \
  --candidate my_model.py --level 80 95
# many stores or items: name the columns that identify one series
python3 scripts/backtest.py data.csv --date date --target sales --id store item \
  --horizon 28 --season 7 --max-series 50 --candidate my_model.py
# rolling window (last 730 points at every origin) and volume-weighted averages over series
python3 scripts/backtest.py data.csv --date date --target sales --id store item \
  --horizon 28 --season 7 --window 730 --weight volume
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
  per series (`--weight volume` weights by mean absolute target value in the scored windows instead) and
  counts the series where the candidate beats the best baseline. Series that are too short or have missing
  values are skipped and counted in the output. `--max-series N` takes the first N series in id order, not a
  random sample.
- The expanding window is the default. `--window N` switches to a rolling window of N points.
- MASE is undefined where a training window has no variation. That origin is left out, a series is skipped
  only when every origin is flat, and the run stops only when no series can be scored.
- Needs Python 3 with pandas and numpy.
