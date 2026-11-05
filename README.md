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

You need Python 3 with pandas and numpy. The ETS example also needs statsmodels.

## Use

Ask Claude something like "forecast the next 12 months and tell me if it beats a baseline". The skill's description has English and Turkish trigger phrases.

You can also run the script yourself:

```bash
python3 scripts/backtest.py data.csv --date date --target sales --horizon 28 --season 7 --origins 8
```

To score your own model, pass a Python file that defines `forecast(train, h)`. `train` is a pandas Series with a DatetimeIndex and `h` is the number of periods to return. Add `--candidate my_model.py` to the command, and `--by-horizon` to see the error at each step ahead.

The metric is MASE, the average error divided by the in-sample error of a seasonal naive forecast. Lower is better, and the number compares across series. MAE comes along in the original units. Run the tests with `python3 scripts/test_backtest.py`.

## Example

Air Passengers, monthly counts from 1949 to 1960 ([Kaggle copy](https://www.kaggle.com/datasets/pattnaiksatyajit/air-passengers), CC0). Horizon 12, season 12, 8 origins six months apart, scored against the Holt-Winters model in `examples/ets_monthly.py`:

```bash
kaggle datasets download pattnaiksatyajit/air-passengers -f AirPassengers.csv
python3 scripts/backtest.py AirPassengers.csv --date Month --target '#Passengers' \
  --horizon 12 --season 12 --origins 8 --step 6 --candidate examples/ets_monthly.py
```

```
                   MAE   MASE
candidate       15.372  0.523
seasonal_naive  36.656  1.252
naive           64.219  2.185
drift           66.106  2.252

candidate MASE 0.523 vs best baseline (seasonal_naive) 1.252: beats the baseline
```

It is one small dataset and eight origins, so treat it as a smoke test.

## Limits

- One series per run. For many stores or items, filter to one id or loop over ids.
- Expanding training window only.
- No check of prediction-interval coverage yet.
- A training window with no variation stops the run, because MASE is undefined there.

MIT licensed, see [LICENSE](LICENSE).
