# app/models/preprocess.py
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing utilities for time series forecasting
    """
    
    @staticmethod
    def validate_time_series_data(data: pd.Series) -> Tuple[bool, str]:
        """
        Validate time series data for forecasting
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if data is None or data.empty:
            return False, "Data is empty or None"
        
        if len(data) < 10:
            return False, "Insufficient data points (minimum 10 required)"
        
        if data.isnull().sum() > len(data) * 0.3:
            return False, "Too many missing values (>30%)"
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return False, "Data must have datetime index"
        
        # Check for duplicate timestamps
        if data.index.duplicated().any():
            return False, "Duplicate timestamps found"
        
        # Check for negative values in stock/medical data
        if (data < 0).any():
            return False, "Negative values found in data"
        
        return True, "Data is valid"
    
    @staticmethod
    def clean_time_series(data: pd.Series, method: str = 'interpolate') -> pd.Series:
        """
        Clean time series data by handling missing values and outliers
        
        Args:
            data: Raw time series data
            method: Method for handling missing values ('interpolate', 'forward_fill', 'drop')
            
        Returns:
            Cleaned time series data
        """
        cleaned_data = data.copy()
        
        # Handle missing values
        if method == 'interpolate':
            cleaned_data = cleaned_data.interpolate(method='time')
        elif method == 'forward_fill':
            cleaned_data = cleaned_data.fillna(method='ffill')
        elif method == 'drop':
            cleaned_data = cleaned_data.dropna()
        
        # Remove outliers using IQR method
        Q1 = cleaned_data.quantile(0.25)
        Q3 = cleaned_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them to preserve data points
        cleaned_data = cleaned_data.clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Cleaned data: {len(data) - len(cleaned_data)} points removed/modified")
        
        return cleaned_data
    
    @staticmethod
    def prepare_medical_data(medical_records: List[dict], 
                           date_column: str = 'visit_date',
                           value_column: str = 'patient_count') -> pd.Series:
        """
        Convert medical records to time series format
        
        Args:
            medical_records: List of medical record dictionaries
            date_column: Column name for dates
            value_column: Column name for values to forecast
            
        Returns:
            Time series data ready for forecasting
        """
        if not medical_records:
            raise ValueError("No medical records provided")
        
        df = pd.DataFrame(medical_records)
        
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Group by date and count occurrences (or sum values)
        if value_column in df.columns:
            time_series = df.groupby(date_column)[value_column].sum()
        else:
            # Count visits per day if no specific value column
            time_series = df.groupby(date_column).size()
        
        # Ensure continuous date range
        date_range = pd.date_range(start=time_series.index.min(), 
                                 end=time_series.index.max(), 
                                 freq='D')
        time_series = time_series.reindex(date_range, fill_value=0)
        
        return time_series
    
    @staticmethod
    def prepare_medicine_stock_data(inventory_data: List[dict], 
                                  movements_data: List[dict]) -> pd.Series:
        """
        Convert medicine inventory and movements to time series format
        
        Args:
            inventory_data: List of inventory records
            movements_data: List of stock movement records
            
        Returns:
            Time series of stock levels over time
        """
        if not movements_data:
            raise ValueError("No stock movement data provided")
        
        movements_df = pd.DataFrame(movements_data)
        movements_df['movement_date'] = pd.to_datetime(movements_df['movement_date'])
        
        # Calculate daily stock changes
        daily_changes = movements_df.groupby('movement_date').agg({
            'quantity': 'sum'
        })['quantity']
        
        # Create continuous time series
        date_range = pd.date_range(start=daily_changes.index.min(),
                                 end=daily_changes.index.max(),
                                 freq='D')
        daily_changes = daily_changes.reindex(date_range, fill_value=0)
        
        # Calculate cumulative stock levels
        stock_levels = daily_changes.cumsum()
        
        return stock_levels
    
    @staticmethod
    def detect_seasonality(data: pd.Series, max_period: int = 365) -> Optional[int]:
        """
        Detect seasonal patterns in time series data
        
        Args:
            data: Time series data
            max_period: Maximum period to check for seasonality
            
        Returns:
            Detected seasonal period or None
        """
        from scipy import signal
        
        if len(data) < max_period * 2:
            return None
        
        # Use autocorrelation to detect seasonality
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr[1:], height=np.max(autocorr) * 0.1)
        
        if len(peaks) > 0:
            # Return the most prominent seasonal period
            return peaks[0] + 1
        
        return None
    
    @staticmethod
    def validate_forecast_parameters(order: Tuple[int, int, int], 
                                   seasonal_order: Tuple[int, int, int, int],
                                   data_length: int) -> Tuple[bool, str]:
        """
        Validate SARIMA model parameters
        
        Args:
            order: (p, d, q) parameters
            seasonal_order: (P, D, Q, s) parameters
            data_length: Length of the time series data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        p, d, q = order
        P, D, Q, s = seasonal_order
        
        # Check parameter ranges
        if any(param < 0 for param in [p, d, q, P, D, Q]):
            return False, "SARIMA parameters cannot be negative"
        
        if any(param > 5 for param in [p, q, P, Q]):
            return False, "AR/MA parameters should not exceed 5"
        
        if d + D > 2:
            return False, "Total differencing (d + D) should not exceed 2"
        
        if s <= 1:
            return False, "Seasonal period must be greater than 1"
        
        # Check if we have enough data points
        min_required = max(p + d + q + P + D + Q + s, s * 2)
        if data_length < min_required:
            return False, f"Insufficient data points. Need at least {min_required}, got {data_length}"
        
        return True, "Parameters are valid"