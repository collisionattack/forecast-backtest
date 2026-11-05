# forecast-backtest

A Claude skill for time-series forecasting. It makes Claude score simple baselines first, so a model only counts as good once it beats them on a rolling-origin backtest.

## What it does

When you ask Claude for a forecast, the skill sets a fixed order of work. Claude pins down the horizon, season length and metric. It checks the data for duplicate dates, silent gaps and leakage. Then it scores naive, seasonal naive and drift, and only after that scores your model against the best of them.

`scripts/backtest.py` does the scoring. If the model loses to a baseline, the output says so and Claude recommends the baseline.

## Install

Clone it straight into your skills folder:

```bash
git clone https://github.com/collisionattack/forecast-backtest ~/.claude/skills/forecast-backtest
```

You need Python 3 with pandas 2 or newer and numpy. The ETS example also needs statsmodels.

## Use

Ask Claude something like "forecast the next 12 months and tell me if it beats a baseline". The skill's description has English and Turkish trigger phrases.

You can also run the script yourself:

```bash
python3 scripts/backtest.py data.csv --date date --target sales --horizon 28 --season 7 --origins 8
```

Dates are read with pandas' format inference. For day-first dates such as `05-02-2010`, add `--date-format %d-%m-%Y`. Without it the script stops with a message instead of guessing.

To score your own model, pass a Python file that defines `forecast(train, h)`. `train` is a pandas Series with a DatetimeIndex and `h` is the number of periods to return. Add `--candidate my_model.py` to the command, and `--by-horizon` to see the error at each step ahead.

For prediction intervals, also define `forecast_interval(train, h, level)` in the same file, returning `(lower, upper)` with `h` values each, and add `--level 80 95`. The script prints the share of test points inside each interval and the mean width in MASE units. Coverage well below the level means the ranges are too narrow, well above means too wide.

For data with many series, name the columns that identify one with `--id`, for example `--id store item`. The script scores each series on its own and averages MASE with equal weight per series. It also prints in how many series the candidate beats the best baseline. Series that are too short, have no variation or have missing target values are skipped and counted, never filled. `--max-series 50` scores only the first 50 series in id order, which keeps a slow model like ETS manageable.

The metric is MASE, the average error divided by the in-sample error of a seasonal naive forecast. Lower is better, and the number compares across series. MAE comes along in the original units. `--season` takes several values, for example `--season 7 364` for daily data with a weekly and a yearly pattern. Each value adds a seasonal naive, all are scored on the same origins, and MASE is scaled by the first value. A longer season needs a longer history, so short series lose origins or are skipped. Run the tests with `python3 scripts/test_backtest.py`.

## Example

Air Passengers, monthly counts from 1949 to 1960 ([Kaggle copy](https://www.kaggle.com/datasets/pattnaiksatyajit/air-passengers), CC0). Horizon 12, season 12, 8 origins six months apart, scored against the Holt-Winters model in `examples/ets_monthly.py`:

```bash
kaggle datasets download pattnaiksatyajit/air-passengers -f AirPassengers.csv
python3 scripts/backtest.py AirPassengers.csv --date Month --target '#Passengers' \
  --horizon 12 --season 12 --origins 8 --step 6 --candidate examples/ets_monthly.py --level 80 95
```

```
                   MAE   MASE
candidate       15.372  0.523
seasonal_naive  36.656  1.252
naive           64.219  2.185
drift           66.106  2.252

candidate MASE 0.523 vs best baseline (seasonal_naive) 1.252: beats the baseline

prediction intervals of the candidate, over 1 series and 8 origins
level  coverage  width (MASE units)
   80     95.8%                2.66
   95     99.0%                3.99
```

The 80% and 95% ranges held 95.8% and 99.0% of the test points, so on this series they are wider than they need to be. It is one small dataset and eight origins, so treat it as a smoke test.

## Limits

- Scores over many series weight every series equally. There is no volume weighting.
- Expanding training window only.
- A training window with no variation makes MASE undefined. That series is skipped and counted, and the run stops only when no series can be scored.

MIT licensed, see [LICENSE](LICENSE).
