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


def test_a_series_that_never_varies_gets_its_own_line_instead_of_a_score():
    tmp = Path(tempfile.mkdtemp())
    pd.DataFrame({"date": pd.date_range("2000-01-01", periods=60, freq="MS"), "y": 1.0}).to_csv(tmp / "f.csv", index=False)
    table, out = run_cli(tmp / "f.csv")  # nothing has a MASE, so there is no table
    assert table is None and "flat series: 1" in out and "mean absolute error per model" in out, out
    assert re.search(r"^naive\s+0\.0$", out, re.M), out  # naive reproduces a constant exactly


def test_flat_series_are_counted_next_to_the_scored_ones():
    tmp = Path(tempfile.mkdtemp())
    d = pd.date_range("2000-01-01", periods=72, freq="MS")
    varied = 20 + 5 * np.sin(2 * np.pi * np.arange(72) / 12) + np.random.default_rng(4).normal(0, 0.2, 72)
    pd.concat([pd.DataFrame({"id": "a", "date": d, "y": varied}), pd.DataFrame({"id": "flat", "date": d, "y": 5.0})]).to_csv(tmp / "m.csv", index=False)
    table, out = run_cli(tmp / "m.csv", "--id", "id")
    assert "series: 1 scored, 0 skipped, 1 flat" in out and "flat series: 1" in out and np.isfinite(table["MASE"]).all(), out


def test_a_flat_rolling_window_still_has_a_scale_from_the_history():
    y = make(np.r_[np.arange(40.0) % 7 + 1, np.zeros(60)])  # varies early, then zeros, so every 30-point window is flat
    table, _, k = bt.backtest(y, {"naive": bt.naive}, 4, 1, 3, 4, window=30)
    assert k == 3 and np.isfinite(table["MASE"]).all()


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


def test_yearly_example_beats_last_year_on_a_noisy_seasonal_series():
    t = np.arange(1826)
    noise = np.random.default_rng(11).normal(0, 3, 1826)
    y = pd.Series(np.maximum(20 + 0.004 * t + 6 * np.sin(2 * np.pi * t / 7) + 8 * np.sin(2 * np.pi * t / 365.25) + noise, 0),
                  index=pd.date_range("2013-01-01", periods=1826, freq="D"))
    example = Path(__file__).resolve().parent.parent / "examples" / "yearly_harmonic.py"
    models = {"last_year": lambda train, h, m: bt.seasonal_naive(train, h, 364), "candidate": bt.load_candidate(str(example))}
    table, _, k = bt.backtest(y, models, 90, 7, 3, 90, longest=364)
    # copying a noisy year doubles the noise variance, a fitted pattern does not
    assert k == 3 and table.index[0] == "candidate" and np.isfinite(table["MAE"]).all(), table


def test_a_flat_origin_is_left_out_and_the_series_still_counts():
    y = make(np.r_[np.zeros(60), np.arange(40.0) % 7 + 1])  # the windows ending at rows 46 and 58 are all zeros
    table, _, k = bt.backtest(y, {"naive": bt.naive}, 6, 1, 5, 12)
    assert k == 3 and np.isfinite(table["MAE"]).all()
    assert raises(ValueError, lambda: bt.backtest(make(np.zeros(100)), {"naive": bt.naive}, 6, 1, 5, 12))  # nothing left to score


def test_volume_weight_uses_the_scored_windows_not_the_whole_history():
    tmp = Path(tempfile.mkdtemp())
    d = pd.date_range("2000-01-01", periods=100, freq="MS")
    small = 10 + np.arange(100.0) % 5
    big = np.where(np.arange(100) >= 88, small * 10, small)  # same history, ten times the volume in the scored windows
    pd.concat([pd.DataFrame({"id": s, "date": d, "y": v}) for s, v in (("a", small), ("b", big))]).to_csv(tmp / "v.csv", index=False)
    weighted, _ = run_cli(tmp / "v.csv", "--horizon", "4", "--season", "1", "--origins", "3", "--id", "id", "--weight", "volume")
    mae = [bt.backtest(make(v), {"naive": bt.naive}, 4, 1, 3, 4)[0].loc["naive", "MAE"] for v in (small, big)]
    w = [np.mean([np.abs(v[c:c + 4]).mean() for c in (88, 92, 96)]) for v in (small, big)]
    assert np.isclose(weighted.loc["naive", "MAE"], np.average(mae, weights=w))


def test_volume_weight_stops_when_every_scored_window_is_zero():
    tmp = Path(tempfile.mkdtemp())
    y = np.where(np.arange(100) >= 80, 0.0, 10 + np.arange(100.0) % 5)  # varies in the past, zero in the scored windows
    pd.DataFrame({"id": "a", "date": pd.date_range("2000-01-01", periods=100, freq="MS"), "y": y}).to_csv(tmp / "z.csv", index=False)
    msg = exit_message(lambda: run_cli(tmp / "z.csv", "--horizon", "4", "--season", "1", "--origins", "3", "--id", "id", "--weight", "volume"))
    assert "zero volume" in msg, msg


def test_several_candidates_share_one_table_and_match_their_solo_runs():
    tmp = Path(tempfile.mkdtemp())
    write_multi_csv(tmp / "m.csv")
    (tmp / "repeat_last.py").write_text("import numpy as np\ndef forecast(train, h):\n    return np.repeat(train.iloc[-1], h)\n")
    (tmp / "last_season.py").write_text("import numpy as np\ndef forecast(train, h):\n    return np.resize(train.to_numpy()[-12:], h)\n")
    both, out = run_cli(tmp / "m.csv", "--id", "id", "--candidate", str(tmp / "repeat_last.py"), str(tmp / "last_season.py"))
    assert {"repeat_last", "last_season"} <= set(both.index) and "candidate" not in both.index
    assert "repeat_last MASE" in out and "last_season MASE" in out and out.count("beats the best baseline in") == 2, out
    for name in ("repeat_last", "last_season"):  # the same numbers as when the file runs alone
        solo, _ = run_cli(tmp / "m.csv", "--id", "id", "--candidate", str(tmp / f"{name}.py"))
        assert np.isclose(both.loc[name, "MAE"], solo.loc["candidate", "MAE"]) and np.isclose(both.loc[name, "MASE"], solo.loc["candidate", "MASE"])


def test_candidate_files_with_the_same_name_are_refused():
    tmp = Path(tempfile.mkdtemp())
    line_csv(tmp)
    for sub in ("a", "b"):
        (tmp / sub).mkdir()
        (tmp / sub / "m.py").write_text("import numpy as np\ndef forecast(train, h):\n    return np.repeat(train.iloc[-1], h)\n")
    with contextlib.redirect_stderr(io.StringIO()) as err:
        assert raises(SystemExit, lambda: run_cli(tmp / "line.csv", "--horizon", "4", "--candidate", str(tmp / "a" / "m.py"), str(tmp / "b" / "m.py")))
    assert "different names" in err.getvalue(), err.getvalue()


def test_level_reports_only_the_candidates_that_define_intervals():
    tmp = Path(tempfile.mkdtemp())
    line_csv(tmp)
    (tmp / "with_band.py").write_text(INTERVAL_CANDIDATE.format(top=2))
    (tmp / "plain.py").write_text("import numpy as np\ndef forecast(train, h):\n    return np.repeat(train.iloc[-1], h)\n")
    _, out = run_cli(tmp / "line.csv", "--horizon", "4", "--season", "1", "--origins", "3",
                     "--candidate", str(tmp / "with_band.py"), str(tmp / "plain.py"), "--level", "80")
    assert "prediction intervals of with_band" in out and "prediction intervals of plain" not in out, out
    assert re.search(r"\b80\s+50\.0%\s+2\.00\b", out), out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
