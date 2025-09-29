# app/routes/forecast.py
from fastapi import APIRouter, Query
from app.models.sarima import train_sarima_model, forecast_sarima
import pandas as pd

router = APIRouter()

@router.get("/")
def get_forecast(steps: int = Query(6, ge=1, le=24)):
  
    # Dummy Data
    data = pd.Series(
        [120, 130, 145, 160, 170, 190, 210, 220, 230, 250, 265, 280],
        index=pd.date_range(start="2024-01-01", periods=12, freq="M")
    )
    
    # Train SARIMA model
    model = train_sarima_model(data, order=(1,1,1), seasonal_order=(1,1,1,12))
    
    # Forecast the next `steps` periods
    predictions = forecast_sarima(model, steps=steps)
    
    return {"forecast": predictions}
