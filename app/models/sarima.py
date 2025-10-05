
# app/models/sarima.py
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Tuple, Optional, Dict, Any
import logging
import warnings

logger = logging.getLogger(__name__)

class SARIMAForecaster:
    """
    Enhanced SARIMA forecasting with error handling and model validation
    """
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.model_info = {}
    
    def train_sarima_model(self, 
                          data: pd.Series, 
                          order: Tuple[int, int, int] = (1,1,1), 
                          seasonal_order: Tuple[int, int, int, int] = (1,1,1,12),
                          validate_residuals: bool = True) -> Optional[Any]:
        """
        Train SARIMA model with comprehensive error handling
        
        Args:
            data: Time series data
            order: (p, d, q) parameters
            seasonal_order: (P, D, Q, s) parameters
            validate_residuals: Whether to validate model residuals
            
        Returns:
            Fitted SARIMA model or None if training fails
        """
        try:
            # Validate input data
            if data is None or data.empty:
                raise ValueError("Input data is empty or None")
            
            if len(data) < 10:
                raise ValueError(f"Insufficient data points: {len(data)}. Minimum 10 required.")
            
            # Check for missing values
            if data.isnull().any():
                logger.warning("Missing values detected. Filling with interpolation.")
                data = data.interpolate(method='time')
            
            # Validate SARIMA parameters
            p, d, q = order
            P, D, Q, s = seasonal_order
            
            if any(param < 0 for param in [p, d, q, P, D, Q]):
                raise ValueError("SARIMA parameters cannot be negative")
            
            min_required = max(p + d + q + P + D + Q + s, s * 2)
            if len(data) < min_required:
                raise ValueError(f"Insufficient data for parameters. Need {min_required}, got {len(data)}")
            
            # Suppress convergence warnings for cleaner output
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                # Create and fit model
                self.model = SARIMAX(
                    data, 
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                self.fitted_model = self.model.fit(
                    disp=False,
                    maxiter=100,
                    method='lbfgs'
                )
            
            # Store model information
            self.model_info = {
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'data_points': len(data),
                'training_date': pd.Timestamp.now()
            }
            
            # Validate model residuals if requested
            if validate_residuals:
                self._validate_residuals()
            
            logger.info(f"SARIMA model trained successfully. AIC: {self.fitted_model.aic:.2f}")
            return self.fitted_model
            
        except Exception as e:
            logger.error(f"Failed to train SARIMA model: {str(e)}")
            self.fitted_model = None
            raise ValueError(f"Model training failed: {str(e)}")
    
    def forecast_sarima(self, steps: int = 10, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Generate forecasts with confidence intervals and error handling
        
        Args:
            steps: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary containing forecasts, confidence intervals, and metadata
        """
        try:
            if self.fitted_model is None:
                raise ValueError("Model not trained. Call train_sarima_model first.")
            
            if steps <= 0 or steps > 100:
                raise ValueError("Steps must be between 1 and 100")
            
            if not 0.5 <= confidence_level <= 0.99:
                raise ValueError("Confidence level must be between 0.5 and 0.99")
            
            # Generate forecast
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            
            # Extract predictions and confidence intervals
            predictions = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=1-confidence_level)
            
            # Create forecast dates
            last_date = self.fitted_model.data.dates[-1]
            if isinstance(last_date, pd.Timestamp):
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=steps,
                    freq='D'
                )
            else:
                forecast_dates = range(1, steps + 1)
            
            # Ensure non-negative forecasts for stock/medical data
            predictions = np.maximum(predictions, 0)
            conf_int.iloc[:, 0] = np.maximum(conf_int.iloc[:, 0], 0)  # Lower bound
            conf_int.iloc[:, 1] = np.maximum(conf_int.iloc[:, 1], 0)  # Upper bound
            
            result = {
                'forecast': predictions.tolist(),
                'forecast_dates': [str(date) for date in forecast_dates],
                'confidence_intervals': {
                    'lower': conf_int.iloc[:, 0].tolist(),
                    'upper': conf_int.iloc[:, 1].tolist()
                },
                'confidence_level': confidence_level,
                'model_info': self.model_info,
                'forecast_generated_at': str(pd.Timestamp.now())
            }
            
            logger.info(f"Generated {steps}-step forecast successfully")
            return result
            
        except Exception as e:
            logger.error(f"Forecasting failed: {str(e)}")
            raise ValueError(f"Forecasting failed: {str(e)}")
    
    def _validate_residuals(self) -> None:
        """
        Validate model residuals for model adequacy
        """
        try:
            residuals = self.fitted_model.resid
            
            # Ljung-Box test for autocorrelation in residuals
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            # Check if residuals are white noise (p-value > 0.05)
            if (lb_test['lb_pvalue'] < 0.05).any():
                logger.warning("Model residuals show autocorrelation. Consider different parameters.")
            
            # Check residual variance
            if residuals.var() > residuals.mean() * 2:
                logger.warning("High residual variance detected. Model may not fit well.")
                
        except Exception as e:
            logger.warning(f"Residual validation failed: {str(e)}")
    
    def forecast(self, data: np.ndarray, steps: int = 10, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Main forecast method that handles training and prediction
        
        Args:
            data: Time series data as numpy array
            steps: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary containing forecasts, confidence intervals, and model info
        """
        try:
            # Convert numpy array to pandas Series
            if isinstance(data, np.ndarray):
                data_series = pd.Series(data)
            elif isinstance(data, list):
                data_series = pd.Series(data)
            else:
                data_series = data
            
            # Validate data
            if len(data_series) < 10:
                logger.warning(f"Insufficient data for SARIMA: {len(data_series)} points. Using simple forecast.")
                return self._simple_forecast_fallback(data_series, steps, confidence_level)
            
            # Auto-select SARIMA parameters based on data characteristics
            best_order, best_seasonal_order = self._auto_select_parameters(data_series)
            
            # Train model
            try:
                self.train_sarima_model(data_series, best_order, best_seasonal_order, validate_residuals=False)
                
                # Generate forecast
                forecast_result = self.forecast_sarima(steps, confidence_level)
                
                # Add model performance info
                forecast_result['model_info']['auto_selected_params'] = True
                forecast_result['model_info']['data_quality'] = self._assess_data_quality(data_series)
                
                return forecast_result
                
            except Exception as sarima_error:
                logger.warning(f"SARIMA training failed: {str(sarima_error)}. Using fallback method.")
                return self._simple_forecast_fallback(data_series, steps, confidence_level)
                
        except Exception as e:
            logger.error(f"Forecast method failed: {str(e)}")
            return self._simple_forecast_fallback(pd.Series([1, 1, 1]), steps, confidence_level)
    
    def _auto_select_parameters(self, data: pd.Series) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Auto-select SARIMA parameters based on data characteristics
        """
        try:
            data_length = len(data)
            
            # Simple parameter selection based on data length and characteristics
            if data_length < 20:
                # Simple model for small datasets
                order = (1, 1, 1)
                seasonal_order = (0, 0, 0, 0)  # No seasonality for small data
            elif data_length < 50:
                # Basic seasonal model
                order = (1, 1, 1)
                seasonal_order = (1, 0, 1, 7)  # Weekly seasonality
            else:
                # More complex model for larger datasets
                order = (2, 1, 2)
                seasonal_order = (1, 1, 1, 7)  # Weekly seasonality with trend
            
            # Adjust seasonal period based on data frequency
            if data_length > 30:
                # Check for monthly patterns
                seasonal_order = (seasonal_order[0], seasonal_order[1], seasonal_order[2], min(12, data_length // 4))
            
            return order, seasonal_order
            
        except Exception as e:
            logger.warning(f"Parameter auto-selection failed: {str(e)}. Using default parameters.")
            return (1, 1, 1), (0, 0, 0, 0)
    
    def _assess_data_quality(self, data: pd.Series) -> Dict[str, Any]:
        """
        Assess data quality for forecasting
        """
        try:
            return {
                'length': len(data),
                'missing_values': data.isnull().sum(),
                'zero_values': (data == 0).sum(),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'trend_strength': self._calculate_trend_strength(data),
                'seasonality_detected': self._detect_seasonality(data)
            }
        except Exception as e:
            logger.warning(f"Data quality assessment failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_trend_strength(self, data: pd.Series) -> float:
        """
        Calculate trend strength in the data
        """
        try:
            if len(data) < 10:
                return 0.0
            
            # Simple linear trend calculation
            x = np.arange(len(data))
            correlation = np.corrcoef(x, data)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _detect_seasonality(self, data: pd.Series) -> bool:
        """
        Simple seasonality detection
        """
        try:
            if len(data) < 14:  # Need at least 2 weeks for weekly seasonality
                return False
            
            # Check for weekly patterns (7-day cycle)
            weekly_correlation = 0.0
            if len(data) >= 14:
                for lag in [7, 14]:
                    if lag < len(data):
                        correlation = data.autocorr(lag=lag)
                        if not np.isnan(correlation):
                            weekly_correlation = max(weekly_correlation, abs(correlation))
            
            return weekly_correlation > 0.3
            
        except Exception:
            return False
    
    def _simple_forecast_fallback(self, data: pd.Series, steps: int, confidence_level: float) -> Dict[str, Any]:
        """
        Simple fallback forecasting method when SARIMA fails
        """
        try:
            if len(data) == 0:
                # No data available
                forecast_values = [1.0] * steps
                lower_bounds = [0.5] * steps
                upper_bounds = [1.5] * steps
            else:
                # Use simple moving average with trend
                recent_mean = data.tail(min(7, len(data))).mean()
                overall_mean = data.mean()
                
                # Simple trend calculation
                if len(data) > 1:
                    trend = (data.iloc[-1] - data.iloc[0]) / len(data)
                else:
                    trend = 0
                
                # Generate forecasts
                forecast_values = []
                for i in range(steps):
                    forecast_val = max(0, recent_mean + trend * i)
                    forecast_values.append(forecast_val)
                
                # Simple confidence intervals (±20% of forecast)
                lower_bounds = [max(0, val * 0.8) for val in forecast_values]
                upper_bounds = [val * 1.2 for val in forecast_values]
            
            # Create forecast dates
            forecast_dates = [(pd.Timestamp.now() + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(steps)]
            
            return {
                'forecast': forecast_values,
                'forecast_dates': forecast_dates,
                'confidence_intervals': {
                    'lower': lower_bounds,
                    'upper': upper_bounds
                },
                'confidence_level': confidence_level,
                'model_info': {
                    'model_type': 'simple_fallback',
                    'aic': 999,  # High AIC indicates simple model
                    'data_points': len(data),
                    'fallback_reason': 'SARIMA training failed or insufficient data'
                },
                'forecast_generated_at': str(pd.Timestamp.now())
            }
            
        except Exception as e:
            logger.error(f"Even fallback forecast failed: {str(e)}")
            # Ultimate fallback - return minimal valid response
            return {
                'forecast': [1.0] * steps,
                'forecast_dates': [str(pd.Timestamp.now().date())] * steps,
                'confidence_intervals': {'lower': [0.5] * steps, 'upper': [1.5] * steps},
                'confidence_level': confidence_level,
                'model_info': {'model_type': 'emergency_fallback', 'aic': 9999},
                'forecast_generated_at': str(pd.Timestamp.now())
            }

    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive model diagnostics
        
        Returns:
            Dictionary with model performance metrics
        """
        if self.fitted_model is None:
            return {"error": "Model not trained"}
        
        try:
            residuals = self.fitted_model.resid
            
            diagnostics = {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'log_likelihood': self.fitted_model.llf,
                'residual_mean': float(residuals.mean()),
                'residual_std': float(residuals.std()),
                'residual_skewness': float(residuals.skew()),
                'residual_kurtosis': float(residuals.kurtosis()),
                'model_parameters': self.model_info
            }
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Failed to generate diagnostics: {str(e)}")
            return {"error": f"Diagnostics generation failed: {str(e)}"}

# Backward compatibility functions
def train_sarima_model(data: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """Backward compatibility wrapper"""
    forecaster = SARIMAForecaster()
    return forecaster.train_sarima_model(data, order, seasonal_order)

def forecast_sarima(model, steps=10):
    """Backward compatibility wrapper"""
    if hasattr(model, 'get_forecast'):
        try:
            forecast = model.get_forecast(steps=steps)
            return forecast.predicted_mean.tolist()
        except Exception as e:
            logger.error(f"Forecast failed: {str(e)}")
            return []
    else:
        return []
