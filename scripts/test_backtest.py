"""Self-check for backtest.py. Run: python3 test_backtest.py (pytest also works)."""
import contextlib
import io
import re
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


def exit_message(fn):
    try:
        fn()
    except SystemExit as e:
        return str(e)
    raise AssertionError("expected SystemExit")


def test_series_with_a_missing_target_is_skipped_and_counted():
    tmp = Path(tempfile.mkdtemp())
    write_multi_csv(tmp / "m.csv")
    df = pd.read_csv(tmp / "m.csv")
    df.loc[df.index[df.id == "b"][5], "y"] = np.nan
    df.to_csv(tmp / "m.csv", index=False)
    _, out = run_cli(tmp / "m.csv", "--id", "id")
    assert "series: 1 scored, 2 skipped" in out and "skipped b: target has missing values" in out, out


def test_a_single_series_with_a_missing_target_stops_with_the_reason():
    tmp = Path(tempfile.mkdtemp())
    y = np.arange(60.0) % 12
    y[30] = np.nan
    pd.DataFrame({"date": pd.date_range("2000-01-01", periods=60, freq="MS"), "y": y}).to_csv(tmp / "n.csv", index=False)
    msg = exit_message(lambda: run_cli(tmp / "n.csv"))
    assert "no series could be scored" in msg and "missing values" in msg, msg


def test_run_stops_when_every_series_is_flat():
    tmp = Path(tempfile.mkdtemp())
    pd.DataFrame({"date": pd.date_range("2000-01-01", periods=60, freq="MS"), "y": 1.0}).to_csv(tmp / "f.csv", index=False)
    msg = exit_message(lambda: run_cli(tmp / "f.csv"))
    assert "no series could be scored" in msg and "no variation" in msg, msg


def test_gappy_dates_print_one_warning():
    tmp = Path(tempfile.mkdtemp())
    d = pd.date_range("2000-01-01", periods=40, freq="MS").delete(10)  # one month missing
    pd.DataFrame({"date": d, "y": np.arange(39.0)}).to_csv(tmp / "g.csv", index=False)
    with contextlib.redirect_stderr(io.StringIO()) as err:
        bt.load(str(tmp / "g.csv"), "date", "y")
    assert "1 of 1 series have irregular or gappy dates" in err.getvalue(), err.getvalue()


def test_by_horizon_prints_one_row_per_step():
    tmp = Path(tempfile.mkdtemp())
    write_multi_csv(tmp / "m.csv")
    _, out = run_cli(tmp / "m.csv", "--id", "id", "--by-horizon")
    rows = [line.split()[0] for line in out.split("MASE by horizon step")[1].splitlines() if line.split()]
    assert "1" in rows and "12" in rows, out


def test_several_seasons_are_scored_on_the_same_windows():
    tmp = Path(tempfile.mkdtemp())
    t = np.arange(900)
    y = 50 + 8 * np.sin(2 * np.pi * t / 7) + 5 * np.sin(2 * np.pi * t / 364) + np.random.default_rng(5).normal(0, 1, 900)
    pd.DataFrame({"date": pd.date_range("2015-01-01", periods=900), "y": y}).to_csv(tmp / "d.csv", index=False)
    args = ("--horizon", "90", "--origins", "3", "--step", "90")
    short, _ = run_cli(tmp / "d.csv", *args, "--season", "7")
    both, out = run_cli(tmp / "d.csv", *args, "--season", "7", "364")
    long_only, _ = run_cli(tmp / "d.csv", *args, "--season", "364")
    assert {"seasonal_naive_7", "seasonal_naive_364"} <= set(both.index)
    assert "origins: 1," in out  # 364 needs 728 training days, only the last origin has them
    assert not np.isclose(short.loc["naive", "MAE"], both.loc["naive", "MAE"])  # --season 7 alone scored 3 origins
    assert np.isclose(long_only.loc["naive", "MAE"], both.loc["naive", "MAE"])  # naive does not depend on the season


INTERVAL_CANDIDATE = (
    "import numpy as np\n"
    "def forecast(train, h):\n    return np.repeat(train.iloc[-1], h)\n"
    "def forecast_interval(train, h, level):\n    return np.repeat(train.iloc[-1], h), np.repeat(train.iloc[-1] + {top}, h)\n"
)


def line_csv(tmp):
    pd.DataFrame({"date": pd.date_range("2000-01-01", periods=100, freq="MS"), "y": np.arange(100.0)}).to_csv(tmp / "line.csv", index=False)


def test_interval_coverage_counts_the_points_inside_and_the_width():
    # straight line: naive misses by 1..4, so the band [last, last + 2] holds the first two steps only
    tmp = Path(tempfile.mkdtemp())
    line_csv(tmp)
    (tmp / "c.py").write_text(INTERVAL_CANDIDATE.format(top=2))
    _, out = run_cli(tmp / "line.csv", "--horizon", "4", "--season", "1", "--origins", "3",
                     "--candidate", str(tmp / "c.py"), "--level", "80", "95")
    assert re.search(r"\b80\s+50\.0%\s+2\.00\b", out) and re.search(r"\b95\s+50\.0%\s+2\.00\b", out), out


def test_level_needs_a_candidate_with_forecast_interval():
    tmp = Path(tempfile.mkdtemp())
    line_csv(tmp)
    (tmp / "plain.py").write_text("import numpy as np\ndef forecast(train, h):\n    return np.repeat(train.iloc[-1], h)\n")
    for args, word in ((("--level", "80"), "--candidate"), (("--candidate", str(tmp / "plain.py"), "--level", "80"), "forecast_interval")):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            assert raises(SystemExit, lambda: run_cli(tmp / "line.csv", "--horizon", "4", *args))
        assert word in err.getvalue(), err.getvalue()


def test_a_backwards_interval_is_refused():
    tmp = Path(tempfile.mkdtemp())
    line_csv(tmp)
    (tmp / "bad.py").write_text(INTERVAL_CANDIDATE.format(top=-2))  # upper below lower
    msg = exit_message(lambda: run_cli(tmp / "line.csv", "--horizon", "4", "--season", "1", "--origins", "3",
                                       "--candidate", str(tmp / "bad.py"), "--level", "80"))
    assert "no series could be scored" in msg and "lower <= upper" in msg, msg


def test_rolling_window_gives_every_origin_the_same_number_of_points():
    seen = []

    def spy(train, h, m):
        seen.append(len(train))
        return np.repeat(train.iloc[-1], h)

    y = make(np.arange(50.0) % 5 + 0.1 * np.arange(50))
    _, _, k = bt.backtest(y, {"spy": spy}, 6, 1, 5, 6, window=30)
    assert seen == [30] * k and k == 3  # cuts 44, 38, 32 fit a window of 30, cuts 26 and 20 do not
    seen.clear()
    _, _, k = bt.backtest(y, {"spy": spy}, 6, 1, 5, 6)
    assert seen == [20, 26, 32, 38, 44] and k == 5  # expanding window by default


def test_window_shorter_than_the_seasons_is_refused():
    tmp = Path(tempfile.mkdtemp())
    write_multi_csv(tmp / "m.csv")
    with contextlib.redirect_stderr(io.StringIO()) as err:
        assert raises(SystemExit, lambda: run_cli(tmp / "m.csv", "--window", "20"))  # season 12 needs 24
    assert "--window must be at least 24" in err.getvalue(), err.getvalue()


def test_volume_weight_follows_the_series_size():
    # y = k * t: naive misses by k * (1..4), so its MAE is 2.5 * k. Mean target is 49.5 * k.
    tmp = Path(tempfile.mkdtemp())
    d = pd.date_range("2000-01-01", periods=100, freq="MS")
    pd.concat([pd.DataFrame({"id": s, "date": d, "y": k * np.arange(100.0)}) for s, k in (("a", 1), ("b", 10))]).to_csv(tmp / "w.csv", index=False)
    args = ("--horizon", "4", "--season", "1", "--origins", "3", "--id", "id")
    equal, _ = run_cli(tmp / "w.csv", *args)
    weighted, out = run_cli(tmp / "w.csv", *args, "--weight", "volume")
    assert np.isclose(equal.loc["naive", "MAE"], 2.5 * (1 + 10) / 2)
    assert np.isclose(weighted.loc["naive", "MAE"], 2.5 * (1 * 49.5 + 10 * 495) / (49.5 + 495))
    assert "weighted by mean target value" in out


def test_volume_weight_needs_id():
    tmp = Path(tempfile.mkdtemp())
    line_csv(tmp)
    with contextlib.redirect_stderr(io.StringIO()) as err:
        assert raises(SystemExit, lambda: run_cli(tmp / "line.csv", "--weight", "volume"))
    assert "--weight volume needs --id" in err.getvalue(), err.getvalue()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
