# app/models/schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
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

# New Comprehensive Forecasting Schemas
class TimeAggregation(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class ForecastType(str, Enum):
    PATIENT_VOLUME = "patient_volume"
    DISEASE_TRENDS = "disease_trends"
    MEDICINE_DEMAND = "medicine_demand"
    RESOURCE_UTILIZATION = "resource_utilization"
    APPOINTMENT_PATTERNS = "appointment_patterns"
    REVENUE_FORECAST = "revenue_forecast"

class HistoricalPeriod(str, Enum):
    ONE_YEAR = "1_year"
    TWO_YEARS = "2_years"
    THREE_YEARS = "3_years"
    FOUR_YEARS = "4_years"
    FIVE_YEARS = "5_years"

class ComprehensiveForecastRequest(BaseModel):
    """Comprehensive forecast request with advanced parameters"""
    forecast_type: ForecastType = Field(description="Type of forecast to generate")
    time_aggregation: TimeAggregation = Field(default=TimeAggregation.DAILY, description="Time aggregation level")
    historical_period: HistoricalPeriod = Field(default=HistoricalPeriod.TWO_YEARS, description="Historical data period")
    forecast_horizon: int = Field(default=90, ge=7, le=730, description="Forecast horizon in days")
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99, description="Confidence level")
    
    # Optional filters
    department_filter: Optional[str] = Field(None, description="Filter by department")
    medicine_category: Optional[str] = Field(None, description="Filter by medicine category")
    disease_category: Optional[str] = Field(None, description="Filter by disease category")
    patient_demographics: Optional[Dict[str, Any]] = Field(None, description="Patient demographic filters")
    
    # Advanced options
    include_seasonality: bool = Field(default=True, description="Include seasonal analysis")
    include_trends: bool = Field(default=True, description="Include trend analysis")
    include_anomalies: bool = Field(default=True, description="Include anomaly detection")
    ensemble_methods: bool = Field(default=True, description="Use ensemble forecasting methods")
    
    @validator('forecast_horizon')
    def validate_horizon(cls, v):
        if v < 7:
            raise ValueError('Forecast horizon must be at least 7 days')
        if v > 730:
            raise ValueError('Forecast horizon cannot exceed 2 years')
        return v

class ForecastDataPoint(BaseModel):
    """Individual forecast data point"""
    date: str = Field(description="Date of forecast point")
    predicted_value: float = Field(description="Predicted value")
    confidence_lower: float = Field(description="Lower confidence bound")
    confidence_upper: float = Field(description="Upper confidence bound")
    trend_component: Optional[float] = Field(None, description="Trend component")
    seasonal_component: Optional[float] = Field(None, description="Seasonal component")
    anomaly_score: Optional[float] = Field(None, description="Anomaly detection score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ForecastSummary(BaseModel):
    """Comprehensive forecast summary"""
    total_predicted: float = Field(description="Total predicted value over horizon")
    average_daily: float = Field(description="Average daily predicted value")
    peak_value: float = Field(description="Peak predicted value")
    peak_date: str = Field(description="Date of peak value")
    trend_direction: str = Field(description="Overall trend direction (increasing/decreasing/stable)")
    trend_strength: float = Field(description="Trend strength (0-1)")
    seasonality_detected: bool = Field(description="Whether seasonality was detected")
    seasonal_period: Optional[int] = Field(None, description="Detected seasonal period")
    forecast_accuracy: float = Field(description="Estimated forecast accuracy")
    data_quality_score: float = Field(description="Quality score of input data")
    risk_level: str = Field(description="Risk level (low/medium/high)")

class ModelPerformance(BaseModel):
    """Model performance metrics"""
    model_name: str = Field(description="Name of the forecasting model")
    accuracy_metrics: Dict[str, float] = Field(description="Accuracy metrics (MAE, RMSE, MAPE)")
    training_period: str = Field(description="Training data period")
    validation_score: float = Field(description="Cross-validation score")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    model_parameters: Dict[str, Any] = Field(description="Model parameters used")

class ForecastRecommendation(BaseModel):
    """AI-generated recommendation"""
    category: str = Field(description="Recommendation category")
    title: str = Field(description="Recommendation title")
    description: str = Field(description="Detailed description")
    priority: str = Field(description="Priority level (low/medium/high/critical)")
    confidence: float = Field(description="Confidence in recommendation (0-1)")
    suggested_actions: List[str] = Field(description="List of suggested actions")
    expected_impact: str = Field(description="Expected impact of following recommendation")
    timeline: str = Field(description="Recommended timeline for action")
    resources_required: Optional[List[str]] = Field(None, description="Resources required")

class ComprehensiveForecastResponse(BaseModel):
    """Comprehensive forecast response"""
    forecast_id: str = Field(description="Unique forecast identifier")
    forecast_type: ForecastType = Field(description="Type of forecast")
    time_aggregation: TimeAggregation = Field(description="Time aggregation used")
    generated_at: str = Field(description="Timestamp when forecast was generated")
    
    # Core forecast data
    forecast_data: List[ForecastDataPoint] = Field(description="Detailed forecast data points")
    forecast_summary: ForecastSummary = Field(description="Forecast summary statistics")
    
    # Model information
    model_performance: List[ModelPerformance] = Field(description="Performance of models used")
    best_model: str = Field(description="Name of best performing model")
    
    # Insights and recommendations
    recommendations: List[ForecastRecommendation] = Field(description="AI-generated recommendations")
    insights: List[str] = Field(description="Key insights from the forecast")
    alerts: List[Dict[str, Any]] = Field(description="Important alerts or warnings")
    
    # Historical context
    historical_comparison: Dict[str, Any] = Field(description="Comparison with historical patterns")
    seasonal_patterns: Optional[Dict[str, Any]] = Field(None, description="Detected seasonal patterns")
    
    # Data quality and metadata
    data_sources: List[str] = Field(description="Data sources used")
    data_quality: Dict[str, float] = Field(description="Data quality metrics")
    limitations: List[str] = Field(description="Known limitations of the forecast")

class QuickInsightRequest(BaseModel):
    """Request for quick insights dashboard"""
    insight_types: List[str] = Field(description="Types of insights requested")
    time_range: str = Field(default="30_days", description="Time range for insights")
    departments: Optional[List[str]] = Field(None, description="Specific departments")

class QuickInsightResponse(BaseModel):
    """Quick insights response"""
    insights: List[Dict[str, Any]] = Field(description="Quick insights")
    key_metrics: Dict[str, float] = Field(description="Key performance metrics")
    trends: Dict[str, str] = Field(description="Current trends")
    alerts: List[str] = Field(description="Important alerts")
    generated_at: str = Field(description="Generation timestamp")

# Disease-Specific Prediction Schemas
class DiseasePredictionRequest(BaseModel):
    """Request for disease-specific outbreak predictions with percentages"""
    forecast_period: int = Field(default=30, ge=7, le=365, description="Forecast period in days")
    time_aggregation: TimeAggregation = Field(default=TimeAggregation.DAILY, description="Time aggregation level")
    historical_period: HistoricalPeriod = Field(default=HistoricalPeriod.TWO_YEARS, description="Historical data period")
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99, description="Confidence level")
    
    # Disease filtering options
    disease_categories: Optional[List[str]] = Field(None, description="Specific disease categories to predict")
    severity_filter: Optional[str] = Field(None, description="Filter by severity (mild/moderate/severe)")
    age_group_filter: Optional[str] = Field(None, description="Filter by age group")
    seasonal_focus: bool = Field(default=True, description="Focus on seasonal disease patterns")
    
    # Prediction options
    include_outbreak_probability: bool = Field(default=True, description="Include outbreak probability calculations")
    include_risk_assessment: bool = Field(default=True, description="Include risk level assessment")
    include_historical_comparison: bool = Field(default=True, description="Compare with historical patterns")
    min_confidence_threshold: float = Field(default=0.6, ge=0.5, le=1.0, description="Minimum confidence for predictions")

class DiseaseOutbreakPrediction(BaseModel):
    """Individual disease outbreak prediction with percentage chance"""
    disease_name: str = Field(description="Name of the disease")
    disease_category: str = Field(description="Disease category code")
    category_name: str = Field(description="Human-readable category name")
    
    # Prediction percentages
    outbreak_probability: float = Field(description="Outbreak probability percentage (0-100)")
    confidence_level: float = Field(description="Confidence in prediction (0-100)")
    risk_level: str = Field(description="Risk level (Low/Medium/High/Critical)")
    
    # Detailed predictions
    predicted_cases: Dict[str, float] = Field(description="Predicted cases by time period")
    peak_probability_date: Optional[str] = Field(None, description="Most likely peak date")
    peak_cases_estimate: Optional[float] = Field(None, description="Estimated peak cases")
    
    # Historical context
    historical_average: float = Field(description="Historical average cases for same period")
    percentage_change: float = Field(description="Percentage change from historical average")
    seasonal_pattern: str = Field(description="Seasonal pattern (winter_peak/summer_peak/no_pattern)")
    
    # Clinical metadata
    severity_distribution: Dict[str, float] = Field(description="Expected severity distribution")
    age_group_risk: Dict[str, float] = Field(description="Risk by age group")
    contagious: bool = Field(description="Whether disease is contagious")
    chronic: bool = Field(description="Whether disease is chronic")
    
    # Confidence intervals
    confidence_intervals: Dict[str, Dict[str, float]] = Field(description="Confidence intervals for predictions")

class DiseasePredictionSummary(BaseModel):
    """Summary of disease predictions"""
    total_diseases_analyzed: int = Field(description="Total number of diseases analyzed")
    high_risk_diseases: int = Field(description="Number of high-risk diseases")
    critical_risk_diseases: int = Field(description="Number of critical-risk diseases")
    
    # Overall statistics
    average_outbreak_probability: float = Field(description="Average outbreak probability across all diseases")
    highest_risk_disease: str = Field(description="Disease with highest outbreak probability")
    highest_risk_probability: float = Field(description="Highest outbreak probability percentage")
    
    # Seasonal insights
    seasonal_diseases_count: int = Field(description="Number of diseases with seasonal patterns")
    dominant_seasonal_pattern: str = Field(description="Most common seasonal pattern")
    
    # Confidence metrics
    average_confidence: float = Field(description="Average confidence across predictions")
    reliable_predictions_count: int = Field(description="Number of predictions above confidence threshold")
    
    # Risk distribution
    risk_distribution: Dict[str, int] = Field(description="Count of diseases by risk level")

class DiseasePredictionResponse(BaseModel):
    """Comprehensive disease prediction response with percentages"""
    prediction_id: str = Field(description="Unique prediction identifier")
    generated_at: str = Field(description="Timestamp when prediction was generated")
    forecast_period: int = Field(description="Forecast period in days")
    
    # Core predictions
    disease_predictions: List[DiseaseOutbreakPrediction] = Field(description="Individual disease predictions")
    prediction_summary: DiseasePredictionSummary = Field(description="Summary of all predictions")
    
    # Risk alerts
    high_risk_alerts: List[Dict[str, Any]] = Field(description="Alerts for high-risk diseases")
    outbreak_warnings: List[Dict[str, Any]] = Field(description="Potential outbreak warnings")
    
    # Recommendations
    prevention_recommendations: List[ForecastRecommendation] = Field(description="Disease prevention recommendations")
    resource_recommendations: List[ForecastRecommendation] = Field(description="Resource allocation recommendations")
    
    # Model information
    model_performance: Dict[str, float] = Field(description="Prediction model performance metrics")
    data_quality: Dict[str, float] = Field(description="Data quality assessment")
    
    # Historical context
    historical_comparison: Dict[str, Any] = Field(description="Comparison with historical disease patterns")
    seasonal_analysis: Dict[str, Any] = Field(description="Seasonal disease pattern analysis")
    
    # Metadata
    data_sources: List[str] = Field(description="Data sources used for predictions")
    limitations: List[str] = Field(description="Known limitations of predictions")
    confidence_notes: List[str] = Field(description="Notes about prediction confidence")
