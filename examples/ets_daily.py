import warnings

from statsmodels.tsa.holtwinters import ExponentialSmoothing


def forecast(train, h):  # damped additive trend, additive seasonality over 7 days, for daily data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # keeps statsmodels' fit warnings out of the output
        return ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7).fit().forecast(h)
