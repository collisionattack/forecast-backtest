import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _fit(train):  # additive trend, multiplicative seasonality, monthly data
    return ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()


def forecast(train, h):
    return _fit(train).forecast(h)


def forecast_interval(train, h, level):  # only needed for --level: percentiles of 500 simulated paths
    sims = np.asarray(_fit(train).simulate(h, repetitions=500, error="mul", random_state=0))
    return np.percentile(sims, [(100 - level) / 2, 100 - (100 - level) / 2], axis=1)
