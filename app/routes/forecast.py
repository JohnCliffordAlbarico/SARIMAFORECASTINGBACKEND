# app/routes/forecast.py
from fastapi import APIRouter, HTTPException, Depends, Query
from app.services.supabase import supabase
from app.models.sarima import SARIMAForecaster
from app.models.preprocess import DataPreprocessor
from app.models.schemas import ForecastRequest, ForecastResponse, ErrorResponse
from app.utils.middleware import limiter
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=ForecastResponse)
def get_forecast(
    steps: int = Query(6, ge=1, le=100, description="Number of forecast periods"),
    confidence_level: float = Query(0.95, ge=0.5, le=0.99, description="Confidence level"),
    data_source: str = Query("medical_records", description="Data source: medical_records or medicine_stock"),
    medicine_id: Optional[int] = Query(None, description="Medicine ID for stock forecasting"),
    dep=Depends(limiter.limit("10/minute"))
):
    """
    Generate time series forecasts using SARIMA model with real database data
    """
    try:
        # Validate input parameters
        if steps <= 0 or steps > 100:
            raise HTTPException(status_code=400, detail="Steps must be between 1 and 100")
        
        if not 0.5 <= confidence_level <= 0.99:
            raise HTTPException(status_code=400, detail="Confidence level must be between 0.5 and 0.99")
        
        # Get data based on source
        if data_source == "medical_records":
            time_series_data = _get_medical_records_data()
        elif data_source == "medicine_stock":
            time_series_data = _get_medicine_stock_data(medicine_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid data source. Use 'medical_records' or 'medicine_stock'")
        
        # Validate and preprocess data
        preprocessor = DataPreprocessor()
        is_valid, error_msg = preprocessor.validate_time_series_data(time_series_data)
        
        if not is_valid:
            logger.error(f"Data validation failed: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Data validation failed: {error_msg}")
        
        # Clean the data
        cleaned_data = preprocessor.clean_time_series(time_series_data)
        
        # Detect seasonality
        seasonal_period = preprocessor.detect_seasonality(cleaned_data)
        if seasonal_period is None:
            seasonal_period = 7  # Default to weekly seasonality
        
        # Train SARIMA model
        forecaster = SARIMAForecaster()
        
        # Use detected seasonality or default parameters
        seasonal_order = (1, 1, 1, min(seasonal_period, 12))  # Cap at 12 for stability
        
        model = forecaster.train_sarima_model(
            cleaned_data, 
            order=(1, 1, 1), 
            seasonal_order=seasonal_order
        )
        
        if model is None:
            raise HTTPException(status_code=500, detail="Failed to train forecasting model")
        
        # Generate forecast
        forecast_result = forecaster.forecast_sarima(steps=steps, confidence_level=confidence_level)
        
        logger.info(f"Successfully generated {steps}-step forecast for {data_source}")
        return ForecastResponse(**forecast_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

def _get_medical_records_data() -> pd.Series:
    """
    Fetch and process medical records data from database
    """
    try:
        # Fetch medical records
        response = supabase.table("medical_records").select(
            "visit_date, patient_id"
        ).order("visit_date").execute()
        
        if response.error:
            raise Exception(f"Database error: {response.error.message}")
        
        if not response.data:
            raise Exception("No medical records found in database")
        
        # Convert to DataFrame and process
        df = pd.DataFrame(response.data)
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        
        # Count visits per day
        daily_visits = df.groupby('visit_date').size()
        
        # Ensure continuous date range
        if len(daily_visits) > 0:
            date_range = pd.date_range(
                start=daily_visits.index.min(),
                end=daily_visits.index.max(),
                freq='D'
            )
            daily_visits = daily_visits.reindex(date_range, fill_value=0)
        
        logger.info(f"Retrieved {len(daily_visits)} days of medical records data")
        return daily_visits
        
    except Exception as e:
        logger.error(f"Failed to fetch medical records: {str(e)}")
        raise Exception(f"Failed to fetch medical records: {str(e)}")

def _get_medicine_stock_data(medicine_id: Optional[int] = None) -> pd.Series:
    """
    Fetch and process medicine stock movement data from database
    """
    try:
        # Build query
        query = supabase.table("medicine_stock_movements").select(
            "movement_date, quantity, movement_type, inventory_id"
        )
        
        if medicine_id:
            # Join with inventory to filter by medicine_id
            inventory_response = supabase.table("medicine_inventory").select(
                "id"
            ).eq("medicine_id", medicine_id).execute()
            
            if inventory_response.error:
                raise Exception(f"Database error: {inventory_response.error.message}")
            
            if not inventory_response.data:
                raise Exception(f"No inventory found for medicine_id {medicine_id}")
            
            inventory_ids = [item['id'] for item in inventory_response.data]
            query = query.in_("inventory_id", inventory_ids)
        
        response = query.order("movement_date").execute()
        
        if response.error:
            raise Exception(f"Database error: {response.error.message}")
        
        if not response.data:
            raise Exception("No stock movement data found")
        
        # Convert to DataFrame and process
        df = pd.DataFrame(response.data)
        df['movement_date'] = pd.to_datetime(df['movement_date'])
        
        # Calculate net quantity (positive for IN, negative for OUT)
        df['net_quantity'] = df.apply(
            lambda row: row['quantity'] if row['movement_type'] == 'in' else -row['quantity'], 
            axis=1
        )
        
        # Group by date and sum quantities
        daily_movements = df.groupby('movement_date')['net_quantity'].sum()
        
        # Ensure continuous date range
        if len(daily_movements) > 0:
            date_range = pd.date_range(
                start=daily_movements.index.min(),
                end=daily_movements.index.max(),
                freq='D'
            )
            daily_movements = daily_movements.reindex(date_range, fill_value=0)
            
            # Convert to cumulative stock levels
            stock_levels = daily_movements.cumsum()
            
            # Ensure non-negative stock levels
            stock_levels = stock_levels.clip(lower=0)
        else:
            stock_levels = pd.Series(dtype=float)
        
        logger.info(f"Retrieved {len(stock_levels)} days of stock movement data")
        return stock_levels
        
    except Exception as e:
        logger.error(f"Failed to fetch stock data: {str(e)}")
        raise Exception(f"Failed to fetch stock data: {str(e)}")

@router.get("/diagnostics")
def get_model_diagnostics(
    data_source: str = Query("medical_records", description="Data source to analyze"),
    dep=Depends(limiter.limit("5/minute"))
):
    """
    Get model diagnostics and data quality metrics
    """
    try:
        # Get data
        if data_source == "medical_records":
            time_series_data = _get_medical_records_data()
        elif data_source == "medicine_stock":
            time_series_data = _get_medicine_stock_data()
        else:
            raise HTTPException(status_code=400, detail="Invalid data source")
        
        # Data quality metrics
        preprocessor = DataPreprocessor()
        is_valid, validation_msg = preprocessor.validate_time_series_data(time_series_data)
        
        diagnostics = {
            "data_source": data_source,
            "data_points": len(time_series_data),
            "date_range": {
                "start": str(time_series_data.index.min()) if len(time_series_data) > 0 else None,
                "end": str(time_series_data.index.max()) if len(time_series_data) > 0 else None
            },
            "data_valid": is_valid,
            "validation_message": validation_msg,
            "statistics": {
                "mean": float(time_series_data.mean()) if len(time_series_data) > 0 else 0,
                "std": float(time_series_data.std()) if len(time_series_data) > 0 else 0,
                "min": float(time_series_data.min()) if len(time_series_data) > 0 else 0,
                "max": float(time_series_data.max()) if len(time_series_data) > 0 else 0,
                "missing_values": int(time_series_data.isnull().sum())
            }
        }
        
        # Detect seasonality
        seasonal_period = preprocessor.detect_seasonality(time_series_data)
        diagnostics["seasonality"] = {
            "detected_period": seasonal_period,
            "has_seasonality": seasonal_period is not None
        }
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"Diagnostics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Diagnostics failed: {str(e)}")
