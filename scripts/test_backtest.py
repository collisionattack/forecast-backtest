"""Self-check for backtest.py. Run: python3 test_backtest.py (pytest also works)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt


def make(values, freq="MS"):
    return pd.Series(values, index=pd.date_range("2000-01-01", periods=len(values), freq=freq))


def run(y, h=12, m=12, origins=5, step=None, extra=None):
    models = {"naive": bt.naive, "seasonal_naive": bt.seasonal_naive, "drift": bt.drift, **(extra or {})}
    return bt.backtest(y, models, h, m, origins, step or h)


def test_seasonal_series_seasonal_naive_wins():
    rng = np.random.default_rng(0)
    t = np.arange(144)
    y = make(10 + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.1, 144))
    table, _, k = run(y)
    assert k == 5 and table.index[0] == "seasonal_naive"


def test_random_walk_naive_beats_seasonal_naive():
    rng = np.random.default_rng(1)
    y = make(np.cumsum(rng.normal(size=600)))
    table, _, _ = run(y, h=6, m=12, origins=40, step=6)
    assert table.loc["naive", "MASE"] < table.loc["seasonal_naive", "MASE"]


def test_known_errors_on_a_straight_line():
    # naive misses a +1/step line by 1..h, so MAE = (h+1)/2; drift extrapolates it exactly
    table, _, _ = run(make(np.arange(100.0)), h=4, m=1, origins=3, step=4)
    assert np.isclose(table.loc["naive", "MAE"], 2.5)
    assert np.isclose(table.loc["naive", "MASE"], 2.5)
    assert np.isclose(table.loc["drift", "MAE"], 0.0)


def test_candidate_plugin_matches_naive_when_it_is_naive():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "m.py").write_text("import numpy as np\ndef forecast(train, h):\n    return np.repeat(train.iloc[-1], h)\n")
    y = make(np.arange(60.0) % 12 + 0.1 * np.arange(60))
    table, _, _ = run(y, extra={"candidate": bt.load_candidate(str(tmp / "m.py"))})
    assert np.isclose(table.loc["candidate", "MASE"], table.loc["naive", "MASE"])


def test_bad_input_is_refused():
    tmp = Path(tempfile.mkdtemp())
    pd.DataFrame({"d": ["2020-01-01", "2020-01-01", "2020-01-02"], "y": [1, 2, 3]}).to_csv(tmp / "dup.csv", index=False)
    for call in (
        lambda: bt.load(str(tmp / "dup.csv"), "d", "y"),
        lambda: run(make(np.ones(100))),  # constant series: MASE undefined
        lambda: run(make(np.arange(20.0))),  # too short for horizon 12 + 2 seasons
    ):
        try:
            call()
        except SystemExit:
            continue
        raise AssertionError("expected SystemExit")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
