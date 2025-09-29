
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima_model(data: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit()
    return fitted_model

def forecast_sarima(model, steps=10):
    forecast = model.get_forecast(steps=steps)
    return forecast.predicted_mean.tolist()
