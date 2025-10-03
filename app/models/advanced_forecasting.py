# app/models/advanced_forecasting.py
# DEPRECATED: This file is deprecated. Use SARIMAForecastingEngine instead.
# This file is kept for backward compatibility only.

import warnings
from .sarima_forecasting_engine import SARIMAForecastingEngine

# Issue deprecation warning
warnings.warn(
    "AdvancedForecastingEngine is deprecated. Use SARIMAForecastingEngine instead.",
    DeprecationWarning,
    stacklevel=2
)

# Alias for backward compatibility
class AdvancedForecastingEngine(SARIMAForecastingEngine):
    """
    DEPRECATED: Use SARIMAForecastingEngine instead.
    This class is maintained for backward compatibility only.
    """
    
    def __init__(self):
        super().__init__()
        warnings.warn(
            "AdvancedForecastingEngine is deprecated. Use SARIMAForecastingEngine instead.",
            DeprecationWarning,
            stacklevel=2
        )
