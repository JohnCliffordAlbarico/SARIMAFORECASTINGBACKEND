# app/models/schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ForecastRequest(BaseModel):
    """Request model for forecast endpoint"""
    steps: int = Field(default=6, ge=1, le=100, description="Number of forecast periods")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for prediction intervals")
    data_source: str = Field(default="medical_records", description="Data source for forecasting")
    medicine_id: Optional[int] = Field(None, description="Specific medicine ID for stock forecasting")
    
    @validator('steps')
    def validate_steps(cls, v):
        if v <= 0:
            raise ValueError('Steps must be positive')
        if v > 100:
            raise ValueError('Steps cannot exceed 100')
        return v

class ForecastResponse(BaseModel):
    """Response model for forecast endpoint"""
    forecast: List[float] = Field(description="Forecasted values")
    forecast_dates: List[str] = Field(description="Dates for forecast periods")
    confidence_intervals: Dict[str, List[float]] = Field(description="Upper and lower confidence bounds")
    confidence_level: float = Field(description="Confidence level used")
    model_info: Dict[str, Any] = Field(description="Model metadata")
    forecast_generated_at: str = Field(description="Timestamp when forecast was generated")

class MedicalRecordResponse(BaseModel):
    """Response model for medical records"""
    data: List[Dict[str, Any]] = Field(description="Medical records data")
    total_records: Optional[int] = Field(None, description="Total number of records")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range of records")

class MedicineStockResponse(BaseModel):
    """Response model for medicine stock data"""
    inventory: List[Dict[str, Any]] = Field(description="Current inventory data")
    prescriptions: List[Dict[str, Any]] = Field(description="Prescription data")
    movements: List[Dict[str, Any]] = Field(description="Stock movement data")
    summary: Optional[Dict[str, Any]] = Field(None, description="Stock summary statistics")

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(description="Service status")
    message: str = Field(description="Status message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    database_connected: bool = Field(description="Database connection status")
    version: str = Field(default="1.0.0", description="API version")

# Enums for validation
class DataSourceType(str, Enum):
    MEDICAL_RECORDS = "medical_records"
    MEDICINE_STOCK = "medicine_stock"
    CUSTOM = "custom"

class MovementType(str, Enum):
    IN = "in"
    OUT = "out"
    ADJUSTMENT = "adjustment"
    EXPIRED = "expired"
