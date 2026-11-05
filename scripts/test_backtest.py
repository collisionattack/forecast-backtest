"""Self-check for backtest.py. Run: python3 test_backtest.py (pytest also works)."""
import contextlib
import io
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


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    return False


def write_multi_csv(path):
    """Two clean seasonal series (a, b) and one too short to score (tiny)."""
    rng = np.random.default_rng(3)
    t = np.arange(72)
    frames = [
        pd.DataFrame({"id": sid, "date": pd.date_range("2000-01-01", periods=72, freq="MS"),
                      "y": 20 + amp * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.2, 72)})
        for sid, amp in (("a", 5), ("b", 8))
    ]
    frames.append(pd.DataFrame({"id": "tiny", "date": pd.date_range("2000-01-01", periods=10, freq="MS"),
                                "y": np.arange(10.0)}))
    pd.concat(frames).to_csv(path, index=False)


def run_cli(csv, *extra):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        table = bt.main([str(csv), "--date", "date", "--target", "y", "--horizon", "12", "--season", "12", *extra])
    return table, out.getvalue()


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
    assert raises(SystemExit, lambda: bt.load(str(tmp / "dup.csv"), "d", "y"))  # duplicate dates
    assert raises(ValueError, lambda: run(make(np.ones(100))))  # flat series: MASE undefined
    assert raises(ValueError, lambda: run(make(np.arange(20.0))))  # too short for horizon 12 + 2 seasons
    nan_model = {"bad": lambda train, h, m: np.full(h, np.nan)}
    assert raises(ValueError, lambda: run(make(np.arange(100.0) % 12 + np.arange(100.0)), extra=nan_model))


def test_day_first_dates_need_a_format_and_then_parse_correctly():
    tmp = Path(tempfile.mkdtemp())
    d = pd.date_range("2010-02-05", periods=60, freq="7D")
    pd.DataFrame({"d": d.strftime("%d-%m-%Y"), "y": np.arange(60.0)}).to_csv(tmp / "w.csv", index=False)
    assert raises(SystemExit, lambda: bt.load(str(tmp / "w.csv"), "d", "y"))  # refused, not silently misread
    s = bt.load(str(tmp / "w.csv"), "d", "y", date_format="%d-%m-%Y")["series"]
    assert (s.index[1] - s.index[0]).days == 7 and s.index.freq is not None


def test_beats_needs_a_strictly_lower_mase():
    t = pd.DataFrame({"MAE": [1.0, 2.0, 2.0], "MASE": [0.5, 1.0, 1.0]}, index=["candidate", "naive", "drift"])
    assert bt.beats(t, ["naive", "drift"])
    t.loc["candidate", "MASE"] = 1.0
    assert not bt.beats(t, ["naive", "drift"])


def test_multi_series_averages_and_skips_short_series():
    tmp = Path(tempfile.mkdtemp())
    write_multi_csv(tmp / "m.csv")
    table, out = run_cli(tmp / "m.csv", "--id", "id")
    assert table.index[0] == "seasonal_naive"
    assert "series: 2 scored, 1 skipped" in out and "skipped tiny: series too short" in out


def test_max_series_limits_the_run():
    tmp = Path(tempfile.mkdtemp())
    write_multi_csv(tmp / "m.csv")
    _, out = run_cli(tmp / "m.csv", "--id", "id", "--max-series", "1")
    assert "series: 1 scored, 0 skipped" in out


def test_multi_series_counts_candidate_wins():
    # averaging the last 3 seasons removes noise that seasonal naive copies, so it should win on both series
    tmp = Path(tempfile.mkdtemp())
    write_multi_csv(tmp / "m.csv")
    (tmp / "avg3.py").write_text(
        "import numpy as np\ndef forecast(train, h):\n"
        "    return np.resize(train.to_numpy()[-36:].reshape(3, 12).mean(axis=0), h)\n")
    _, out = run_cli(tmp / "m.csv", "--id", "id", "--origins", "3", "--candidate", str(tmp / "avg3.py"))
    assert "candidate beats the best baseline in 2 of 2 series" in out


def test_candidate_editing_train_in_place_does_not_change_the_baselines():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "m.py").write_text("import numpy as np\ndef forecast(train, h):\n    train /= 2\n    return np.repeat(train.iloc[-1], h)\n")
    values = np.arange(100.0) % 12 + 0.1 * np.arange(100)
    clean, _, _ = run(make(values.copy()), h=6, origins=4, step=6)
    dirty, _, _ = run(make(values.copy()), h=6, origins=4, step=6, extra={"candidate": bt.load_candidate(str(tmp / "m.py"))})
    for name in ("naive", "seasonal_naive", "drift"):
        assert np.isclose(clean.loc[name, "MASE"], dirty.loc[name, "MASE"]), name


def test_missing_column_names_the_columns_that_exist():
    tmp = Path(tempfile.mkdtemp())
    pd.DataFrame({"d": ["2020-01-01", "2020-01-02", "2020-01-03"], "y": [1, 2, 3]}).to_csv(tmp / "c.csv", index=False)
    try:
        bt.load(str(tmp / "c.csv"), "d", "sale")
    except SystemExit as e:  # a plain KeyError traceback used to be the only hint
        assert "sale" in str(e) and "Columns are: d, y" in str(e), str(e)
    else:
        raise AssertionError("expected SystemExit")


def test_step_must_be_at_least_one():
    tmp = Path(tempfile.mkdtemp())
    write_multi_csv(tmp / "m.csv")
    with contextlib.redirect_stderr(io.StringIO()) as err:  # argparse prints the usage line here
        assert raises(SystemExit, lambda: run_cli(tmp / "m.csv", "--step", "-1"))
    assert "--step" in err.getvalue()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
