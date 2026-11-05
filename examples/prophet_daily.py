import logging

import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").disabled = True  # setLevel does not silence it, every fit would print two INFO lines


def forecast(train, h):  # daily data: Prophet's default trend, weekly and yearly seasonality, multiplicative
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode="multiplicative")
    model.fit(pd.DataFrame({"ds": train.index, "y": train.to_numpy()}))
    future = pd.DataFrame({"ds": pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")})
    return np.maximum(model.predict(future)["yhat"].to_numpy(), 0)  # sales cannot go below 0
