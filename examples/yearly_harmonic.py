import numpy as np
import pandas as pd

YEARLY_HARMONICS = 4


def _design(index, t0):  # intercept, linear trend, 6 day-of-week dummies, yearly sine and cosine pairs
    t = (np.arange(len(index)) + t0) / 365.25
    dow, doy = index.dayofweek.to_numpy(), index.dayofyear.to_numpy()
    cols = [np.ones(len(index)), t] + [(dow == d).astype(float) for d in range(1, 7)]
    for k in range(1, YEARLY_HARMONICS + 1):
        angle = 2 * np.pi * k * doy / 365.25
        cols += [np.sin(angle), np.cos(angle)]
    return np.column_stack(cols)


def forecast(train, h):  # daily data with a weekly and a yearly pattern, about two years of history, no negative values
    beta, *_ = np.linalg.lstsq(_design(train.index, 0), np.log1p(train.to_numpy()), rcond=None)
    future = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    return np.maximum(np.expm1(_design(future, len(train)) @ beta), 0)  # back from the log scale
