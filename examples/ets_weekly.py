import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _fit(train):  # damped additive trend, additive seasonality over 52 weeks
    return ExponentialSmoothing(train.asfreq("7D"), trend="add", damped_trend=True, seasonal="add", seasonal_periods=52).fit()


def forecast(train, h):
    return _fit(train).forecast(h)


def forecast_interval(train, h, level):  # only needed for --level: percentiles of 1000 simulated paths
    sims = np.asarray(_fit(train).simulate(h, repetitions=1000, error="add", random_state=0))
    return np.percentile(sims, [(100 - level) / 2, 100 - (100 - level) / 2], axis=1)
