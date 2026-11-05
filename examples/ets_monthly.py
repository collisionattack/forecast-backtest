from statsmodels.tsa.holtwinters import ExponentialSmoothing


def forecast(train, h):  # additive trend, multiplicative seasonality, monthly data
    return ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit().forecast(h)
