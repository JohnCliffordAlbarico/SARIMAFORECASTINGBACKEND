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
    medicine_id: Optional[str] = Field(None, description="Specific medicine ID for stock forecasting")
    forecast_period: int = Field(default=30, ge=1, le=365, description="Forecast period in days")
    
    @validator('steps')
    def validate_steps(cls, v):
        if v <= 0:
            raise ValueError('Steps must be positive')
        if v > 100:
            raise ValueError('Steps cannot exceed 100')
        return v

class HealthTrendRequest(BaseModel):
    """Request model for disease trends forecasting"""
    forecast_period: int = Field(default=30, ge=1, le=365, description="Forecast period in days")
    disease_categories: Optional[List[str]] = Field(None, description="Specific disease categories to focus on")
    include_seasonal: bool = Field(default=True, description="Include seasonal analysis")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level")

class PatientVisitRequest(BaseModel):
    """Request model for patient visit forecasting"""
    forecast_period: int = Field(default=30, ge=1, le=365, description="Forecast period in days")
    include_demographics: bool = Field(default=True, description="Include demographic breakdown")
    department_filter: Optional[str] = Field(None, description="Filter by specific department")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level")

class HealthPatternRequest(BaseModel):
    """Request model for health pattern analysis"""
    forecast_period: int = Field(default=30, ge=1, le=365, description="Forecast period in days")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis (comprehensive, demographic, treatment)")
    include_resources: bool = Field(default=True, description="Include resource utilization analysis")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level")

class ForecastResponse(BaseModel):
    """Response model for forecast endpoint"""
    forecast_data: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed forecast data")
    forecast_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of forecast results")
    recommendations: Optional[List[Dict[str, Any]]] = Field(None, description="AI-generated recommendations")
    model_info: Dict[str, Any] = Field(description="Model metadata")
    
    # Legacy fields for backward compatibility
    forecast: Optional[List[float]] = Field(None, description="Forecasted values (legacy)")
    forecast_dates: Optional[List[str]] = Field(None, description="Dates for forecast periods (legacy)")
    confidence_intervals: Optional[Dict[str, List[float]]] = Field(None, description="Upper and lower confidence bounds (legacy)")
    confidence_level: Optional[float] = Field(None, description="Confidence level used (legacy)")
    forecast_generated_at: Optional[str] = Field(None, description="Timestamp when forecast was generated (legacy)")

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
    status: str = Field(description="Health status (healthy, degraded, unhealthy)")
    message: str = Field(description="Health status message")
    database_connected: bool = Field(description="Database connection status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Health check timestamp")
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
