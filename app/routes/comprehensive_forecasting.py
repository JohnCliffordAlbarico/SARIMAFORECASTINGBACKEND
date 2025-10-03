# app/routes/comprehensive_forecasting.py
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import uuid

from ..services.supabase import supabase
from ..services.data_aggregation import data_aggregation_service
from ..models.schemas import (
    ComprehensiveForecastRequest, ComprehensiveForecastResponse,
    QuickInsightRequest, QuickInsightResponse,
    ForecastType, TimeAggregation, HistoricalPeriod
)
from ..models.sarima_forecasting_engine import SARIMAForecastingEngine
from ..utils.middleware import limiter

router = APIRouter(prefix="/comprehensive", tags=["comprehensive-forecasting"])
logger = logging.getLogger(__name__)

# Initialize SARIMA forecasting engine
forecasting_engine = SARIMAForecastingEngine()

@router.post("/forecast", response_model=ComprehensiveForecastResponse)
async def generate_comprehensive_forecast(request: ComprehensiveForecastRequest):
    """
    Generate comprehensive SARIMA forecasts with advanced analytics
    
    This endpoint provides:
    - Optimized SARIMA forecasting with parameter tuning
    - Pre-filtered time aggregations (daily/weekly/monthly/yearly)
    - 1-5 year historical analysis
    - Advanced seasonal pattern detection
    - Medical-specific SARIMA optimization
    - AI-powered recommendations and insights
    """
    try:
        logger.info(f"Starting comprehensive forecast: {request.forecast_type.value} - {request.time_aggregation.value}")
        
        # Fetch and prepare data based on forecast type
        raw_data = await _fetch_forecast_data(request)
        
        if raw_data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for {request.forecast_type.value} forecast"
            )
        
        # Generate comprehensive forecast
        forecast_response = forecasting_engine.generate_comprehensive_forecast(
            data=raw_data,
            forecast_type=request.forecast_type,
            time_aggregation=request.time_aggregation,
            historical_period=request.historical_period,
            forecast_horizon=request.forecast_horizon,
            confidence_level=request.confidence_level,
            department_filter=request.department_filter,
            medicine_category=request.medicine_category,
            disease_category=request.disease_category,
            patient_demographics=request.patient_demographics
        )
        
        logger.info(f"Comprehensive forecast completed: {forecast_response.forecast_id}")
        return forecast_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive forecasting failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {str(e)}")

@router.get("/quick-insights", response_model=QuickInsightResponse)
async def get_quick_insights(
    time_range: str = Query("30_days", description="Time range for insights"),
    departments: Optional[List[str]] = Query(None, description="Filter by departments"),
    insight_types: List[str] = Query(["volume", "trends", "alerts"], description="Types of insights")
):
    """
    Get quick insights and key metrics for dashboard
    
    Provides rapid overview of:
    - Current trends and patterns
    - Key performance indicators
    - Important alerts and recommendations
    """
    try:
        logger.info(f"Generating quick insights for {time_range}")
        
        # Parse time range
        days_map = {
            "7_days": 7, "30_days": 30, "90_days": 90, 
            "6_months": 180, "1_year": 365
        }
        days = days_map.get(time_range, 30)
        
        # Generate insights
        insights_data = await _generate_quick_insights(days, departments, insight_types)
        
        response = QuickInsightResponse(
            insights=insights_data["insights"],
            key_metrics=insights_data["metrics"],
            trends=insights_data["trends"],
            alerts=insights_data["alerts"],
            generated_at=datetime.now().isoformat()
        )
        
        logger.info("Quick insights generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Quick insights generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")

@router.get("/forecast-types")
async def get_available_forecast_types():
    """Get all available forecast types with descriptions"""
    
    forecast_types = {
        "patient_volume": {
            "name": "Patient Volume Forecasting (SARIMA)",
            "description": "SARIMA-based prediction of patient visit volumes and appointment patterns",
            "data_sources": ["medical_records", "appointments"],
            "recommended_aggregation": ["daily", "weekly"],
            "min_historical_period": "1_year",
            "sarima_optimization": "weekly_seasonal_patterns"
        },
        "disease_trends": {
            "name": "Disease Trends Analysis (SARIMA)",
            "description": "SARIMA-based forecasting of disease outbreak patterns and seasonal trends",
            "data_sources": ["medical_records"],
            "recommended_aggregation": ["daily", "weekly", "monthly"],
            "min_historical_period": "2_years",
            "sarima_optimization": "seasonal_disease_patterns"
        },
        "medicine_demand": {
            "name": "Medicine Demand Forecasting (SARIMA)",
            "description": "SARIMA-based prediction of medicine consumption and inventory needs",
            "data_sources": ["medicine_prescriptions", "medicine_inventory"],
            "recommended_aggregation": ["daily", "weekly"],
            "min_historical_period": "1_year",
            "sarima_optimization": "monthly_prescription_patterns"
        },
        "resource_utilization": {
            "name": "Resource Utilization Analysis (SARIMA)",
            "description": "SARIMA-based forecasting of staff, equipment, and facility utilization",
            "data_sources": ["appointments", "medical_records"],
            "recommended_aggregation": ["daily", "weekly", "monthly"],
            "min_historical_period": "1_year",
            "sarima_optimization": "weekly_resource_patterns"
        },
        "appointment_patterns": {
            "name": "Appointment Pattern Analysis (SARIMA)",
            "description": "SARIMA-based analysis and prediction of appointment booking patterns",
            "data_sources": ["appointments"],
            "recommended_aggregation": ["daily", "weekly"],
            "min_historical_period": "1_year",
            "sarima_optimization": "weekly_appointment_cycles"
        },
        "revenue_forecast": {
            "name": "Revenue Forecasting (SARIMA)",
            "description": "SARIMA-based prediction of clinic revenue from services and prescriptions",
            "data_sources": ["appointments", "medicine_prescriptions"],
            "recommended_aggregation": ["weekly", "monthly", "quarterly"],
            "min_historical_period": "2_years",
            "sarima_optimization": "monthly_revenue_cycles"
        }
    }
    
    return {
        "available_types": forecast_types,
        "time_aggregations": {
            "daily": "Daily aggregation - best for short-term operational planning",
            "weekly": "Weekly aggregation - good for staff scheduling and resource planning",
            "monthly": "Monthly aggregation - ideal for strategic planning and budgeting",
            "quarterly": "Quarterly aggregation - for long-term strategic analysis",
            "yearly": "Yearly aggregation - for multi-year planning and trends"
        },
        "historical_periods": {
            "1_year": "1 year of historical data - minimum for basic forecasting",
            "2_years": "2 years of historical data - recommended for seasonal analysis",
            "3_years": "3 years of historical data - better trend detection",
            "4_years": "4 years of historical data - robust pattern recognition",
            "5_years": "5 years of historical data - maximum historical depth"
        }
    }

@router.get("/data-quality")
async def get_data_quality_assessment(
    time_aggregation: TimeAggregation = Query(TimeAggregation.DAILY, description="Time aggregation level"),
    historical_period: HistoricalPeriod = Query(HistoricalPeriod.TWO_YEARS, description="Historical period")
):
    """Get comprehensive data quality assessment for forecasting"""
    
    try:
        logger.info(f"Assessing data quality for {time_aggregation.value} aggregation over {historical_period.value}")
        
        # Get data quality metrics
        quality_metrics = await data_aggregation_service.get_data_quality_metrics(
            time_aggregation=time_aggregation,
            historical_period=historical_period
        )
        
        # Add recommendations based on quality
        recommendations = []
        
        overall_quality = quality_metrics.get("data_quality_overall", 0)
        
        if overall_quality < 0.5:
            recommendations.append({
                "priority": "high",
                "message": "Low data quality detected. Consider improving data collection processes.",
                "action": "Review data entry procedures and implement validation checks"
            })
        elif overall_quality < 0.7:
            recommendations.append({
                "priority": "medium",
                "message": "Moderate data quality. Some forecasts may have reduced accuracy.",
                "action": "Monitor data collection and consider shorter forecast horizons"
            })
        else:
            recommendations.append({
                "priority": "low",
                "message": "Good data quality for reliable forecasting.",
                "action": "Continue current data collection practices"
            })
        
        return {
            "data_quality_metrics": quality_metrics,
            "recommendations": recommendations,
            "assessment_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data quality assessment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data quality assessment failed: {str(e)}")

@router.post("/clear-cache")
async def clear_forecasting_cache():
    """Clear the forecasting data cache"""
    
    try:
        data_aggregation_service.clear_cache()
        logger.info("Forecasting cache cleared successfully")
        
        return {
            "message": "Forecasting cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache clearing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cache clearing failed: {str(e)}")

# Helper Functions

async def _fetch_forecast_data(request: ComprehensiveForecastRequest) -> pd.DataFrame:
    """Fetch data based on forecast type and filters"""
    
    if request.forecast_type == ForecastType.PATIENT_VOLUME:
        return await _fetch_patient_volume_data(request)
    elif request.forecast_type == ForecastType.DISEASE_TRENDS:
        return await _fetch_disease_trends_data(request)
    elif request.forecast_type == ForecastType.MEDICINE_DEMAND:
        return await _fetch_medicine_demand_data(request)
    elif request.forecast_type == ForecastType.RESOURCE_UTILIZATION:
        return await _fetch_resource_utilization_data(request)
    elif request.forecast_type == ForecastType.APPOINTMENT_PATTERNS:
        return await _fetch_appointment_patterns_data(request)
    elif request.forecast_type == ForecastType.REVENUE_FORECAST:
        return await _fetch_revenue_forecast_data(request)
    else:
        raise ValueError(f"Unsupported forecast type: {request.forecast_type}")

async def _fetch_patient_volume_data(request: ComprehensiveForecastRequest) -> pd.DataFrame:
    """Fetch patient volume data using aggregation service"""
    
    return await data_aggregation_service.get_aggregated_patient_volume(
        time_aggregation=request.time_aggregation,
        historical_period=request.historical_period,
        department_filter=request.department_filter
    )

async def _fetch_disease_trends_data(request: ComprehensiveForecastRequest) -> pd.DataFrame:
    """Fetch disease trends data using aggregation service"""
    
    return await data_aggregation_service.get_aggregated_disease_trends(
        time_aggregation=request.time_aggregation,
        historical_period=request.historical_period,
        disease_category=request.disease_category
    )

async def _fetch_medicine_demand_data(request: ComprehensiveForecastRequest) -> pd.DataFrame:
    """Fetch medicine demand data using aggregation service"""
    
    return await data_aggregation_service.get_aggregated_medicine_demand(
        time_aggregation=request.time_aggregation,
        historical_period=request.historical_period,
        medicine_category=request.medicine_category
    )

async def _fetch_resource_utilization_data(request: ComprehensiveForecastRequest) -> pd.DataFrame:
    """Fetch resource utilization data using aggregation service"""
    
    return await data_aggregation_service.get_aggregated_resource_utilization(
        time_aggregation=request.time_aggregation,
        historical_period=request.historical_period
    )

async def _fetch_appointment_patterns_data(request: ComprehensiveForecastRequest) -> pd.DataFrame:
    """Fetch appointment patterns data using aggregation service"""
    
    return await data_aggregation_service.get_aggregated_appointment_patterns(
        time_aggregation=request.time_aggregation,
        historical_period=request.historical_period
    )

async def _fetch_revenue_forecast_data(request: ComprehensiveForecastRequest) -> pd.DataFrame:
    """Fetch revenue forecast data using aggregation service"""
    
    return await data_aggregation_service.get_aggregated_revenue_data(
        time_aggregation=request.time_aggregation,
        historical_period=request.historical_period
    )

def _filter_by_disease_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Filter medical records by disease category"""
    
    disease_keywords = {
        'respiratory': ['cough', 'fever', 'cold', 'flu', 'pneumonia', 'asthma', 'bronchitis'],
        'gastrointestinal': ['stomach', 'diarrhea', 'vomiting', 'nausea', 'abdominal'],
        'infectious': ['infection', 'viral', 'bacterial', 'fungal'],
        'chronic': ['diabetes', 'hypertension', 'heart', 'chronic'],
        'musculoskeletal': ['pain', 'back', 'joint', 'muscle', 'arthritis']
    }
    
    keywords = disease_keywords.get(category.lower(), [])
    if not keywords:
        return df
    
    # Filter based on chief complaint containing keywords
    mask = df['chief_complaint'].str.lower().str.contains('|'.join(keywords), na=False)
    return df[mask]

async def _filter_by_medicine_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Filter prescriptions by medicine category"""
    
    # Get medicine IDs for the category
    medicine_response = supabase.table("medicines").select(
        "id"
    ).ilike("drug_class", f"%{category}%").execute()
    
    if not medicine_response.data:
        return df
    
    medicine_ids = [item['id'] for item in medicine_response.data]
    return df[df['medicine_id'].isin(medicine_ids)]

async def _generate_quick_insights(days: int, departments: Optional[List[str]], 
                                 insight_types: List[str]) -> Dict[str, Any]:
    """Generate quick insights for dashboard"""
    
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    insights = []
    metrics = {}
    trends = {}
    alerts = []
    
    try:
        # Patient volume insights
        if "volume" in insight_types:
            volume_data = await _get_volume_insights(cutoff_date)
            insights.extend(volume_data["insights"])
            metrics.update(volume_data["metrics"])
            trends.update(volume_data["trends"])
        
        # Trend analysis
        if "trends" in insight_types:
            trend_data = await _get_trend_insights(cutoff_date)
            insights.extend(trend_data["insights"])
            trends.update(trend_data["trends"])
        
        # Alert generation
        if "alerts" in insight_types:
            alert_data = await _get_alert_insights(cutoff_date)
            alerts.extend(alert_data["alerts"])
            
    except Exception as e:
        logger.warning(f"Error generating some insights: {str(e)}")
        insights.append("Some insights unavailable due to data issues")
    
    return {
        "insights": insights,
        "metrics": metrics,
        "trends": trends,
        "alerts": alerts
    }

async def _get_volume_insights(cutoff_date: str) -> Dict[str, Any]:
    """Get volume-related insights"""
    
    # Get recent patient volume
    response = supabase.table("medical_records").select(
        "visit_date"
    ).gte("visit_date", cutoff_date).execute()
    
    insights = []
    metrics = {}
    trends = {}
    
    if response.data:
        df = pd.DataFrame(response.data)
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        daily_counts = df.groupby(df['visit_date'].dt.date).size()
        
        metrics["total_visits"] = len(response.data)
        metrics["average_daily_visits"] = float(daily_counts.mean())
        metrics["peak_daily_visits"] = int(daily_counts.max())
        
        # Trend analysis
        if len(daily_counts) > 7:
            recent_avg = daily_counts.tail(7).mean()
            previous_avg = daily_counts.head(7).mean()
            
            if recent_avg > previous_avg * 1.1:
                trends["patient_volume"] = "increasing"
                insights.append("Patient volume is trending upward")
            elif recent_avg < previous_avg * 0.9:
                trends["patient_volume"] = "decreasing"
                insights.append("Patient volume is trending downward")
            else:
                trends["patient_volume"] = "stable"
                insights.append("Patient volume is stable")
    
    return {"insights": insights, "metrics": metrics, "trends": trends}

async def _get_trend_insights(cutoff_date: str) -> Dict[str, Any]:
    """Get trend-related insights"""
    
    insights = []
    trends = {}
    
    # Appointment trends
    response = supabase.table("appointments").select(
        "scheduled_date, status"
    ).gte("scheduled_date", cutoff_date).execute()
    
    if response.data:
        df = pd.DataFrame(response.data)
        df['scheduled_date'] = pd.to_datetime(df['scheduled_date'])
        
        # Completion rate trend
        total_appointments = len(df)
        completed_appointments = len(df[df['status'] == 'completed'])
        completion_rate = completed_appointments / total_appointments if total_appointments > 0 else 0
        
        trends["appointment_completion"] = f"{completion_rate:.1%}"
        
        if completion_rate > 0.9:
            insights.append("Excellent appointment completion rate")
        elif completion_rate < 0.7:
            insights.append("Low appointment completion rate - consider follow-up strategies")
    
    return {"insights": insights, "trends": trends}

async def _get_alert_insights(cutoff_date: str) -> Dict[str, Any]:
    """Get alert-related insights"""
    
    alerts = []
    
    # Check for unusual patterns
    response = supabase.table("medical_records").select(
        "visit_date"
    ).gte("visit_date", cutoff_date).execute()
    
    if response.data:
        df = pd.DataFrame(response.data)
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        daily_counts = df.groupby(df['visit_date'].dt.date).size()
        
        # Check for unusual spikes
        if len(daily_counts) > 0:
            mean_visits = daily_counts.mean()
            std_visits = daily_counts.std()
            
            recent_visits = daily_counts.tail(3).mean()
            
            if recent_visits > mean_visits + 2 * std_visits:
                alerts.append("Unusual spike in patient volume detected")
            elif recent_visits < mean_visits - 2 * std_visits:
                alerts.append("Unusually low patient volume detected")
    
    return {"alerts": alerts}
