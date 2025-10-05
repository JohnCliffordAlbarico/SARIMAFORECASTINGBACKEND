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
    DiseasePredictionRequest, DiseasePredictionResponse,
    ForecastRecommendation, DiseaseOutbreakPrediction, DiseasePredictionSummary,
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
        
        # Additional validation for minimum data requirements
        if len(raw_data) < 10:
            raise HTTPException(
                status_code=422, 
                detail=f"Insufficient data for reliable forecasting. Found {len(raw_data)} records, minimum 10 required for {request.forecast_type.value} forecast"
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
            taxonomy_filter=request.taxonomy_filter,
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

@router.get("/disease-categories")
async def get_enhanced_disease_categories():
    """Get all available enhanced disease categories with medical taxonomy"""
    
    from ..services.disease_classifier import disease_classifier
    
    categories = disease_classifier.get_all_categories()
    
    return {
        "enhanced_categories": categories,
        "category_count": len(categories),
        "features": [
            "ICD-10 inspired classification",
            "Severity level detection",
            "Age group correlation",
            "Seasonal pattern recognition",
            "Contagious disease identification",
            "Chronic condition tracking"
        ],
        "improvements_over_legacy": {
            "categories": f"{len(categories)} vs 5 legacy categories",
            "accuracy": "Medical taxonomy-based classification",
            "metadata": "Rich disease metadata including seasonality and demographics",
            "severity": "Automatic severity level detection",
            "confidence": "Classification confidence scoring"
        }
    }

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

@router.post("/disease-predictions", response_model=DiseasePredictionResponse)
async def generate_disease_predictions(request: DiseasePredictionRequest):
    """
    Generate specific disease outbreak predictions with percentage chances
    
    This endpoint provides:
    - Individual disease outbreak probabilities (0-100%)
    - SARIMA-based confidence intervals for each disease
    - Risk level categorization (Low/Medium/High/Critical)
    - Seasonal pattern analysis for specific diseases
    - Historical trend comparison with percentage changes
    - Age group and severity risk assessments
    """
    try:
        logger.info(f"Starting disease-specific predictions for {request.forecast_period} days")
        
        # Fetch historical medical records data
        historical_data = await _fetch_historical_medical_data(request)
        
        if historical_data.empty:
            raise HTTPException(
                status_code=404, 
                detail="No medical records found for disease prediction analysis"
            )
        
        # Additional validation for minimum data requirements
        if len(historical_data) < 5:
            raise HTTPException(
                status_code=422, 
                detail=f"Insufficient data for reliable disease predictions. Found {len(historical_data)} records, minimum 5 required"
            )
        
        # Generate disease-specific predictions
        prediction_response = await _generate_comprehensive_disease_predictions(
            historical_data=historical_data,
            request=request
        )
        
        logger.info(f"Disease predictions completed: {prediction_response.prediction_id}")
        return prediction_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Disease prediction generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Disease prediction failed: {str(e)}")

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

# Disease Prediction Helper Functions

async def _fetch_historical_medical_data(request: DiseasePredictionRequest) -> pd.DataFrame:
    """Fetch historical medical records for disease prediction analysis"""
    
    try:
        # Calculate date range based on historical period
        historical_days_map = {
            "1_year": 365,
            "2_years": 730,
            "3_years": 1095,
            "4_years": 1460,
            "5_years": 1825
        }
        
        days_back = historical_days_map.get(request.historical_period.value, 730)
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching medical records from date: {cutoff_date} (last {days_back} days)")
        
        # First try simple query without joins to test basic connectivity
        try:
            simple_query = supabase.table("medical_records").select("*").limit(5)
            simple_response = simple_query.execute()
            logger.info(f"Simple query returned {len(simple_response.data)} records")
        except Exception as simple_e:
            logger.error(f"Simple query failed: {str(simple_e)}")
        
        # Try without the inner join first
        try:
            query_no_join = supabase.table("medical_records").select(
                "id, patient_id, visit_date, chief_complaint, management, diagnosis"
            ).gte("visit_date", cutoff_date).order("visit_date", desc=False)
            
            response_no_join = query_no_join.execute()
            logger.info(f"Query without join returned {len(response_no_join.data)} records")
            
            if response_no_join.data:
                # If we have data without join, try with join
                query = supabase.table("medical_records").select(
                    "id, patient_id, visit_date, chief_complaint, management, diagnosis, "
                    "patients(date_of_birth, gender)"
                ).gte("visit_date", cutoff_date).order("visit_date", desc=False)
                
                response = query.execute()
                logger.info(f"Query with join returned {len(response.data)} records")
                
                if not response.data:
                    logger.warning("Join query returned no data, using data without patient info")
                    # Use data without patient info
                    df = pd.DataFrame(response_no_join.data)
                    df['patient_age'] = 35  # Default age
                    df['patient_gender'] = 'unknown'  # Default gender
                else:
                    # Convert to DataFrame and process with patient info
                    df = pd.DataFrame(response.data)
                    
                    # Calculate patient ages safely
                    def safe_age_calc(row):
                        try:
                            if row.get('patients') and row['patients'].get('date_of_birth'):
                                return _calculate_age(row['patients']['date_of_birth'])
                            return 35  # Default age
                        except:
                            return 35
                    
                    def safe_gender_extract(row):
                        try:
                            if row.get('patients') and row['patients'].get('gender'):
                                return row['patients']['gender']
                            return 'unknown'
                        except:
                            return 'unknown'
                    
                    df['patient_age'] = df.apply(safe_age_calc, axis=1)
                    df['patient_gender'] = df.apply(safe_gender_extract, axis=1)
            else:
                logger.warning("No medical records found in date range, trying without date filter")
                # Try without date filter as last resort
                query_all = supabase.table("medical_records").select(
                    "id, patient_id, visit_date, chief_complaint, management, diagnosis"
                ).limit(100).order("visit_date", desc=False)
                
                response_all = query_all.execute()
                logger.info(f"Query without date filter returned {len(response_all.data)} records")
                
                if response_all.data:
                    df = pd.DataFrame(response_all.data)
                    df['patient_age'] = 35  # Default age
                    df['patient_gender'] = 'unknown'  # Default gender
                else:
                    return pd.DataFrame()
        
        except Exception as query_e:
            logger.error(f"Query execution failed: {str(query_e)}")
            return pd.DataFrame()
        
        # Clean and prepare data
        df['chief_complaint'] = df['chief_complaint'].fillna('')
        df['management'] = df['management'].fillna('')
        df['diagnosis'] = df['diagnosis'].fillna('')
        
        logger.info(f"Fetched {len(df)} medical records for analysis")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching historical medical data: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

async def _generate_comprehensive_disease_predictions(
    historical_data: pd.DataFrame, 
    request: DiseasePredictionRequest
) -> DiseasePredictionResponse:
    """Generate comprehensive disease predictions with percentages using real SARIMA analysis"""
    
    from ..services.disease_classifier import disease_classifier
    
    prediction_id = str(uuid.uuid4())
    generated_at = datetime.now().isoformat()
    
    logger.info(f"Starting real SARIMA disease predictions with {len(historical_data)} records")
    
    if historical_data.empty:
        logger.warning("No historical data available for disease predictions")
        return _create_empty_disease_prediction_response(prediction_id, generated_at, request)
    
    # Classify all diseases in historical data
    logger.info("Classifying diseases in historical data...")
    disease_classifications = []
    
    for _, row in historical_data.iterrows():
        try:
            classification = disease_classifier.classify_disease(
                chief_complaint=row.get('chief_complaint', ''),
                management=row.get('management', ''),
                patient_age=row.get('patient_age', 35)
            )
            
            if classification['category_code'] != 'UNCLASSIFIED' and classification['confidence'] > 0.3:
                classification['visit_date'] = row.get('visit_date')
                classification['patient_id'] = row.get('patient_id')
                disease_classifications.append(classification)
        except Exception as e:
            logger.warning(f"Error classifying disease: {str(e)}")
            continue
    
    if not disease_classifications:
        logger.warning("No diseases could be classified from historical data")
        return _create_empty_disease_prediction_response(prediction_id, generated_at, request)
    
    logger.info(f"Successfully classified {len(disease_classifications)} disease cases")
    
    # Convert to DataFrame for analysis
    disease_df = pd.DataFrame(disease_classifications)
    disease_df['visit_date'] = pd.to_datetime(disease_df['visit_date'])
    
    # Get unique disease categories with sufficient data
    category_counts = disease_df['category_code'].value_counts()
    significant_categories = category_counts[category_counts >= 3].index.tolist()
    
    logger.info(f"Found {len(significant_categories)} categories with sufficient data: {significant_categories}")
    
    # Generate predictions for each significant category
    disease_predictions = []
    high_risk_alerts = []
    outbreak_warnings = []
    
    for category_code in significant_categories:
        try:
            category_data = disease_df[disease_df['category_code'] == category_code].copy()
            
            # Generate real SARIMA-based prediction for this category
            prediction = await _generate_real_sarima_disease_prediction(
                category_data, category_code, request, disease_classifier
            )
            
            if prediction:
                disease_predictions.append(prediction)
                
                # Check for high risk alerts
                if prediction.risk_level in ['High', 'Critical']:
                    high_risk_alerts.append({
                        'disease': prediction.disease_name,
                        'risk_level': prediction.risk_level,
                        'probability': prediction.outbreak_probability,
                        'message': f"High outbreak risk detected for {prediction.disease_name}"
                    })
                
                # Check for outbreak warnings
                if prediction.outbreak_probability > 60:
                    outbreak_warnings.append({
                        'disease': prediction.disease_name,
                        'probability': prediction.outbreak_probability,
                        'peak_date': prediction.peak_probability_date,
                        'message': f"Potential outbreak warning: {prediction.disease_name} ({prediction.outbreak_probability:.1f}% probability)"
                    })
                    
        except Exception as e:
            logger.error(f"Error generating prediction for category {category_code}: {str(e)}")
            continue
    
    if not disease_predictions:
        logger.warning("No valid disease predictions could be generated")
        return _create_empty_disease_prediction_response(prediction_id, generated_at, request)
    
    # Calculate prediction summary
    prediction_summary = _calculate_disease_prediction_summary(disease_predictions)
    
    # Generate recommendations
    prevention_recommendations = _generate_prevention_recommendations(disease_predictions, high_risk_alerts)
    resource_recommendations = _generate_resource_recommendations(disease_predictions, prediction_summary)
    
    # Calculate model performance metrics
    model_performance = {
        'total_data_points': len(historical_data),
        'classified_cases': len(disease_classifications),
        'classification_rate': len(disease_classifications) / len(historical_data) if len(historical_data) > 0 else 0,
        'categories_analyzed': len(significant_categories),
        'average_confidence': np.mean([p.confidence_level for p in disease_predictions]),
        'sarima_model_used': True,
        'statistical_significance': 'high' if len(disease_classifications) > 50 else 'medium' if len(disease_classifications) > 20 else 'low'
    }
    
    # Data quality assessment
    data_quality = {
        'completeness': len([d for d in disease_classifications if d['confidence'] > 0.5]) / len(disease_classifications) if disease_classifications else 0,
        'temporal_coverage': (disease_df['visit_date'].max() - disease_df['visit_date'].min()).days if len(disease_df) > 1 else 0,
        'category_diversity': len(significant_categories),
        'overall_quality': min(1.0, len(disease_classifications) / 100)  # Quality based on data volume
    }
    
    # Historical comparison
    historical_comparison = _generate_historical_comparison(disease_df, disease_predictions)
    
    # Seasonal analysis
    seasonal_analysis = _generate_seasonal_analysis(disease_df, disease_predictions)
    
    logger.info(f"Generated {len(disease_predictions)} disease predictions with real SARIMA analysis")
    
    return DiseasePredictionResponse(
        prediction_id=prediction_id,
        generated_at=generated_at,
        forecast_period=request.forecast_period,
        disease_predictions=disease_predictions,
        prediction_summary=prediction_summary,
        high_risk_alerts=high_risk_alerts,
        outbreak_warnings=outbreak_warnings,
        prevention_recommendations=prevention_recommendations,
        resource_recommendations=resource_recommendations,
        model_performance=model_performance,
        data_quality=data_quality,
        historical_comparison=historical_comparison,
        seasonal_analysis=seasonal_analysis,
        data_sources=['medical_records', 'disease_classifier', 'sarima_engine'],
        limitations=[
            'Predictions based on historical patterns and may not account for external factors',
            'Classification accuracy depends on quality of chief complaint descriptions',
            'Small sample sizes for some diseases may reduce prediction reliability',
            'Seasonal patterns may vary due to climate and demographic changes'
        ],
        confidence_notes=[
            f'Analysis based on {len(disease_classifications)} classified disease cases',
            f'SARIMA models applied to {len(significant_categories)} disease categories',
            f'Overall data quality score: {data_quality["overall_quality"]:.2f}',
            'Higher confidence for diseases with more historical data'
        ]
    )

async def _generate_real_sarima_disease_prediction(
    category_data: pd.DataFrame, 
    category_code: str, 
    request: DiseasePredictionRequest,
    disease_classifier
) -> Optional[DiseaseOutbreakPrediction]:
    """Generate real SARIMA-based disease prediction for a specific category"""
    
    try:
        # Get category information
        category_info = disease_classifier.disease_categories.get(category_code)
        if not category_info:
            return None
        
        # Prepare time series data
        category_data['visit_date'] = pd.to_datetime(category_data['visit_date'])
        category_data = category_data.sort_values('visit_date')
        
        # Aggregate by day for SARIMA analysis
        daily_cases = category_data.groupby(category_data['visit_date'].dt.date).size()
        
        # Ensure we have enough data points for SARIMA
        if len(daily_cases) < 10:
            logger.warning(f"Insufficient data for SARIMA analysis: {category_code} has only {len(daily_cases)} data points")
            return _generate_fallback_prediction(category_data, category_code, category_info, request)
        
        # Create complete date range and fill missing dates with 0
        date_range = pd.date_range(
            start=daily_cases.index.min(), 
            end=daily_cases.index.max(), 
            freq='D'
        )
        
        # Reindex to include all dates
        daily_cases_complete = daily_cases.reindex(date_range.date, fill_value=0)
        
        # Apply SARIMA forecasting
        try:
            from ..models.sarima import SARIMAForecaster
            
            sarima_forecaster = SARIMAForecaster()
            
            # Convert to numpy array for SARIMA
            time_series = daily_cases_complete.values.astype(float)
            
            # Generate SARIMA forecast
            forecast_result = sarima_forecaster.forecast(
                data=time_series,
                steps=request.forecast_period,
                confidence_level=request.confidence_level
            )
            
            # Extract forecast values
            forecast_values = forecast_result.get('forecast', [])
            confidence_intervals = forecast_result.get('confidence_intervals', {})
            model_info = forecast_result.get('model_info', {})
            
            if not forecast_values:
                logger.warning(f"SARIMA forecast failed for {category_code}, using fallback")
                return _generate_fallback_prediction(category_data, category_code, category_info, request)
            
            # Calculate outbreak probability based on SARIMA results
            historical_mean = np.mean(time_series) if len(time_series) > 0 else 0
            forecast_mean = np.mean(forecast_values)
            
            # Calculate probability based on increase over historical average
            if historical_mean > 0:
                percentage_increase = (forecast_mean - historical_mean) / historical_mean * 100
                outbreak_probability = min(100, max(0, 30 + percentage_increase * 2))  # Base 30% + increase factor
            else:
                outbreak_probability = min(100, forecast_mean * 20)  # If no historical data, base on forecast magnitude
            
            # Apply seasonal adjustments
            seasonal_factor = _get_seasonal_factor(category_info.seasonal_pattern)
            outbreak_probability *= seasonal_factor
            
            # Ensure probability is within bounds
            outbreak_probability = max(5, min(95, outbreak_probability))
            
            # Determine risk level
            if outbreak_probability >= 70:
                risk_level = "Critical"
            elif outbreak_probability >= 50:
                risk_level = "High"
            elif outbreak_probability >= 30:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            # Calculate confidence level based on data quality and model fit
            confidence_level = _calculate_prediction_confidence(
                len(time_series), model_info, category_data
            )
            
            # Generate predicted cases by time period
            predicted_cases = {}
            forecast_dates = pd.date_range(
                start=datetime.now().date() + timedelta(days=1),
                periods=request.forecast_period,
                freq='D'
            )
            
            for i, date in enumerate(forecast_dates):
                if i < len(forecast_values):
                    predicted_cases[date.strftime('%Y-%m-%d')] = max(0, round(forecast_values[i], 2))
            
            # Find peak date and cases
            if predicted_cases:
                peak_date = max(predicted_cases.keys(), key=lambda k: predicted_cases[k])
                peak_cases = predicted_cases[peak_date]
            else:
                peak_date = None
                peak_cases = 0
            
            # Calculate percentage change from historical average
            percentage_change = ((forecast_mean - historical_mean) / historical_mean * 100) if historical_mean > 0 else 0
            
            # Generate severity distribution based on historical data
            severity_distribution = _calculate_severity_distribution(category_data)
            
            # Generate age group risk based on historical data
            age_group_risk = _calculate_age_group_risk(category_data)
            
            # Generate confidence intervals for the prediction
            prediction_confidence_intervals = {}
            if confidence_intervals:
                prediction_confidence_intervals = {
                    'daily_cases': {
                        'lower': confidence_intervals.get('lower', []),
                        'upper': confidence_intervals.get('upper', [])
                    },
                    'total_forecast': {
                        'lower': sum(confidence_intervals.get('lower', [])),
                        'upper': sum(confidence_intervals.get('upper', []))
                    }
                }
            
            logger.info(f"Generated SARIMA prediction for {category_code}: {outbreak_probability:.1f}% probability, {risk_level} risk")
            
            return DiseaseOutbreakPrediction(
                disease_name=category_info.name,
                disease_category=category_code,
                category_name=category_info.name,
                outbreak_probability=round(outbreak_probability, 1),
                confidence_level=round(confidence_level, 1),
                risk_level=risk_level,
                predicted_cases=predicted_cases,
                peak_probability_date=peak_date,
                peak_cases_estimate=peak_cases,
                historical_average=round(historical_mean, 2),
                percentage_change=round(percentage_change, 1),
                seasonal_pattern=category_info.seasonal_pattern,
                severity_distribution=severity_distribution,
                age_group_risk=age_group_risk,
                contagious=category_info.contagious,
                chronic=category_info.chronic,
                confidence_intervals=prediction_confidence_intervals
            )
            
        except Exception as sarima_error:
            logger.error(f"SARIMA analysis failed for {category_code}: {str(sarima_error)}")
            return _generate_fallback_prediction(category_data, category_code, category_info, request)
            
    except Exception as e:
        logger.error(f"Error in real SARIMA disease prediction for {category_code}: {str(e)}")
        return None

def _generate_fallback_prediction(
    category_data: pd.DataFrame, 
    category_code: str, 
    category_info, 
    request: DiseasePredictionRequest
) -> DiseaseOutbreakPrediction:
    """Generate fallback prediction when SARIMA fails"""
    
    # Calculate basic statistics from available data
    total_cases = len(category_data)
    days_span = (category_data['visit_date'].max() - category_data['visit_date'].min()).days
    daily_average = total_cases / max(days_span, 1) if days_span > 0 else total_cases / 30
    
    # Simple trend-based prediction
    recent_cases = len(category_data[category_data['visit_date'] >= (datetime.now() - timedelta(days=14))])
    older_cases = len(category_data[category_data['visit_date'] < (datetime.now() - timedelta(days=14))])
    
    if older_cases > 0:
        trend_factor = recent_cases / older_cases
    else:
        trend_factor = 1.0
    
    # Calculate outbreak probability
    base_probability = min(50, daily_average * 10)  # Base probability from frequency
    seasonal_factor = _get_seasonal_factor(category_info.seasonal_pattern)
    trend_adjustment = (trend_factor - 1) * 20  # Trend impact
    
    outbreak_probability = max(10, min(90, base_probability * seasonal_factor + trend_adjustment))
    
    # Determine risk level
    if outbreak_probability >= 60:
        risk_level = "High"
    elif outbreak_probability >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    # Generate simple predicted cases
    predicted_cases = {}
    forecast_dates = pd.date_range(
        start=datetime.now().date() + timedelta(days=1),
        periods=min(request.forecast_period, 30),  # Limit fallback predictions
        freq='D'
    )
    
    for date in forecast_dates:
        # Simple prediction based on daily average with some variation
        daily_prediction = max(0, daily_average * trend_factor * np.random.uniform(0.7, 1.3))
        predicted_cases[date.strftime('%Y-%m-%d')] = round(daily_prediction, 1)
    
    # Find peak
    if predicted_cases:
        peak_date = max(predicted_cases.keys(), key=lambda k: predicted_cases[k])
        peak_cases = predicted_cases[peak_date]
    else:
        peak_date = None
        peak_cases = 0
    
    return DiseaseOutbreakPrediction(
        disease_name=category_info.name,
        disease_category=category_code,
        category_name=category_info.name,
        outbreak_probability=round(outbreak_probability, 1),
        confidence_level=60.0,  # Lower confidence for fallback
        risk_level=risk_level,
        predicted_cases=predicted_cases,
        peak_probability_date=peak_date,
        peak_cases_estimate=peak_cases,
        historical_average=round(daily_average, 2),
        percentage_change=round((trend_factor - 1) * 100, 1),
        seasonal_pattern=category_info.seasonal_pattern,
        severity_distribution=_calculate_severity_distribution(category_data),
        age_group_risk=_calculate_age_group_risk(category_data),
        contagious=category_info.contagious,
        chronic=category_info.chronic,
        confidence_intervals={}
    )

def _get_seasonal_factor(seasonal_pattern: str) -> float:
    """Get seasonal adjustment factor based on current month"""
    
    current_month = datetime.now().month
    
    seasonal_factors = {
        'winter_peak': {12: 1.4, 1: 1.6, 2: 1.5, 3: 1.2, 4: 0.9, 5: 0.7, 6: 0.6, 7: 0.6, 8: 0.7, 9: 0.8, 10: 1.0, 11: 1.2},
        'summer_peak': {12: 0.7, 1: 0.6, 2: 0.7, 3: 0.8, 4: 1.0, 5: 1.2, 6: 1.5, 7: 1.6, 8: 1.4, 9: 1.1, 10: 0.9, 11: 0.8},
        'spring_peak': {12: 0.8, 1: 0.7, 2: 0.8, 3: 1.2, 4: 1.5, 5: 1.6, 6: 1.1, 7: 0.9, 8: 0.8, 9: 0.9, 10: 1.0, 11: 0.9},
        'autumn_peak': {12: 1.0, 1: 0.7, 2: 0.6, 3: 0.7, 4: 0.8, 5: 0.9, 6: 0.8, 7: 0.7, 8: 0.8, 9: 1.2, 10: 1.5, 11: 1.6},
        'no_pattern': {i: 1.0 for i in range(1, 13)},
        'seasonal_variation': {12: 1.1, 1: 1.2, 2: 1.0, 3: 0.9, 4: 0.8, 5: 0.8, 6: 0.9, 7: 1.0, 8: 1.0, 9: 1.1, 10: 1.1, 11: 1.2}
    }
    
    factor_map = seasonal_factors.get(seasonal_pattern, seasonal_factors['no_pattern'])
    return factor_map.get(current_month, 1.0)

def _calculate_prediction_confidence(data_points: int, model_info: dict, category_data: pd.DataFrame) -> float:
    """Calculate confidence level for prediction based on data quality"""
    
    # Base confidence from data volume
    if data_points >= 100:
        base_confidence = 90
    elif data_points >= 50:
        base_confidence = 80
    elif data_points >= 20:
        base_confidence = 70
    else:
        base_confidence = 60
    
    # Adjust for model fit quality
    model_fit = model_info.get('aic', 1000)  # Lower AIC is better
    if model_fit < 100:
        base_confidence += 5
    elif model_fit > 500:
        base_confidence -= 10
    
    # Adjust for data recency
    if len(category_data) > 0:
        days_since_last = (datetime.now() - category_data['visit_date'].max()).days
        if days_since_last <= 7:
            base_confidence += 5
        elif days_since_last > 30:
            base_confidence -= 10
    
    return max(50, min(95, base_confidence))

def _calculate_severity_distribution(category_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate severity distribution from historical data"""
    
    if 'severity' not in category_data.columns or len(category_data) == 0:
        return {'mild': 60.0, 'moderate': 30.0, 'severe': 10.0}
    
    severity_counts = category_data['severity'].value_counts()
    total = len(category_data)
    
    return {
        'mild': round(severity_counts.get('mild', 0) / total * 100, 1),
        'moderate': round(severity_counts.get('moderate', 0) / total * 100, 1),
        'severe': round(severity_counts.get('severe', 0) / total * 100, 1)
    }

def _calculate_age_group_risk(category_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate age group risk distribution"""
    
    if 'patient_age' not in category_data.columns or len(category_data) == 0:
        return {'children': 20.0, 'adults': 60.0, 'elderly': 20.0}
    
    # Define age groups
    children = len(category_data[category_data['patient_age'] < 18])
    adults = len(category_data[(category_data['patient_age'] >= 18) & (category_data['patient_age'] < 65)])
    elderly = len(category_data[category_data['patient_age'] >= 65])
    total = len(category_data)
    
    if total == 0:
        return {'children': 20.0, 'adults': 60.0, 'elderly': 20.0}
    
    return {
        'children': round(children / total * 100, 1),
        'adults': round(adults / total * 100, 1),
        'elderly': round(elderly / total * 100, 1)
    }

def _calculate_disease_prediction_summary(disease_predictions: List[DiseaseOutbreakPrediction]) -> DiseasePredictionSummary:
    """Calculate summary statistics for disease predictions"""
    
    if not disease_predictions:
        return DiseasePredictionSummary(
            total_diseases_analyzed=0,
            high_risk_diseases=0,
            critical_risk_diseases=0,
            average_outbreak_probability=0.0,
            highest_risk_disease="None",
            highest_risk_probability=0.0,
            seasonal_diseases_count=0,
            dominant_seasonal_pattern="no_pattern",
            average_confidence=0.0,
            reliable_predictions_count=0,
            risk_distribution={'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
        )
    
    # Calculate statistics
    total_diseases = len(disease_predictions)
    high_risk_count = len([p for p in disease_predictions if p.risk_level == 'High'])
    critical_risk_count = len([p for p in disease_predictions if p.risk_level == 'Critical'])
    
    avg_probability = np.mean([p.outbreak_probability for p in disease_predictions])
    
    # Find highest risk disease
    highest_risk_pred = max(disease_predictions, key=lambda p: p.outbreak_probability)
    
    # Count seasonal diseases
    seasonal_count = len([p for p in disease_predictions if p.seasonal_pattern != 'no_pattern'])
    
    # Find dominant seasonal pattern
    seasonal_patterns = [p.seasonal_pattern for p in disease_predictions]
    if seasonal_patterns:
        dominant_pattern = max(set(seasonal_patterns), key=seasonal_patterns.count)
    else:
        dominant_pattern = 'no_pattern'
    
    # Calculate average confidence
    avg_confidence = np.mean([p.confidence_level for p in disease_predictions])
    
    # Count reliable predictions (confidence > 70%)
    reliable_count = len([p for p in disease_predictions if p.confidence_level > 70])
    
    # Risk distribution
    risk_counts = {}
    for pred in disease_predictions:
        risk_counts[pred.risk_level] = risk_counts.get(pred.risk_level, 0) + 1
    
    return DiseasePredictionSummary(
        total_diseases_analyzed=total_diseases,
        high_risk_diseases=high_risk_count,
        critical_risk_diseases=critical_risk_count,
        average_outbreak_probability=round(avg_probability, 1),
        highest_risk_disease=highest_risk_pred.disease_name,
        highest_risk_probability=highest_risk_pred.outbreak_probability,
        seasonal_diseases_count=seasonal_count,
        dominant_seasonal_pattern=dominant_pattern,
        average_confidence=round(avg_confidence, 1),
        reliable_predictions_count=reliable_count,
        risk_distribution=risk_counts
    )

def _generate_prevention_recommendations(
    disease_predictions: List[DiseaseOutbreakPrediction], 
    high_risk_alerts: List[Dict]
) -> List[ForecastRecommendation]:
    """Generate prevention recommendations based on predictions"""
    
    recommendations = []
    
    # High-risk disease recommendations
    for alert in high_risk_alerts:
        recommendations.append(ForecastRecommendation(
            category="Disease Prevention",
            title=f"Enhanced Monitoring for {alert['disease']}",
            description=f"Implement enhanced surveillance and prevention measures for {alert['disease']} due to {alert['risk_level'].lower()} outbreak risk ({alert['probability']:.1f}% probability).",
            priority="high",
            confidence=0.85,
            suggested_actions=[
                "Increase screening protocols",
                "Prepare additional medical supplies",
                "Brief medical staff on symptoms and treatment",
                "Consider public health notifications"
            ],
            expected_impact="Reduced outbreak severity and faster response time",
            timeline="Immediate (within 1-2 days)",
            resources_required=["Medical staff training", "Additional supplies", "Communication channels"]
        ))
    
    # Seasonal recommendations
    seasonal_diseases = [p for p in disease_predictions if p.seasonal_pattern != 'no_pattern']
    if seasonal_diseases:
        recommendations.append(ForecastRecommendation(
            category="Seasonal Preparedness",
            title="Seasonal Disease Preparedness",
            description=f"Prepare for seasonal patterns affecting {len(seasonal_diseases)} disease categories.",
            priority="medium",
            confidence=0.75,
            suggested_actions=[
                "Stock seasonal medications",
                "Adjust staffing for seasonal peaks",
                "Implement seasonal health campaigns",
                "Monitor weather-related health impacts"
            ],
            expected_impact="Better resource allocation and patient outcomes",
            timeline="1-2 weeks before seasonal peak",
            resources_required=["Inventory management", "Staff scheduling", "Public health materials"]
        ))
    
    return recommendations

def _generate_resource_recommendations(
    disease_predictions: List[DiseaseOutbreakPrediction], 
    summary: DiseasePredictionSummary
) -> List[ForecastRecommendation]:
    """Generate resource allocation recommendations"""
    
    recommendations = []
    
    # Staffing recommendations
    if summary.high_risk_diseases + summary.critical_risk_diseases > 0:
        recommendations.append(ForecastRecommendation(
            category="Staffing",
            title="Increase Medical Staff Availability",
            description=f"Consider increasing staff availability due to {summary.high_risk_diseases + summary.critical_risk_diseases} high-risk disease predictions.",
            priority="high" if summary.critical_risk_diseases > 0 else "medium",
            confidence=0.80,
            suggested_actions=[
                "Schedule additional medical staff",
                "Prepare on-call rotations",
                "Cross-train staff for multiple specialties",
                "Consider temporary staff augmentation"
            ],
            expected_impact="Improved patient care capacity during potential outbreaks",
            timeline="Within 1 week",
            resources_required=["Staff scheduling", "Budget allocation", "Training resources"]
        ))
    
    # Supply chain recommendations
    total_predicted_cases = sum([
        sum(p.predicted_cases.values()) for p in disease_predictions if p.predicted_cases
    ])
    
    if total_predicted_cases > 50:
        recommendations.append(ForecastRecommendation(
            category="Supply Chain",
            title="Medical Supply Inventory Management",
            description=f"Prepare medical supplies for approximately {total_predicted_cases:.0f} predicted cases across all disease categories.",
            priority="medium",
            confidence=0.75,
            suggested_actions=[
                "Review current inventory levels",
                "Order additional supplies for high-risk diseases",
                "Establish emergency supply protocols",
                "Coordinate with suppliers for rapid delivery"
            ],
            expected_impact="Adequate supply availability during increased demand",
            timeline="Within 2 weeks",
            resources_required=["Inventory management", "Procurement budget", "Supplier coordination"]
        ))
    
    return recommendations

def _generate_historical_comparison(
    disease_df: pd.DataFrame, 
    disease_predictions: List[DiseaseOutbreakPrediction]
) -> Dict[str, Any]:
    """Generate historical comparison analysis"""
    
    # Calculate historical trends
    if len(disease_df) > 0:
        # Group by month to see historical patterns
        disease_df['month'] = disease_df['visit_date'].dt.to_period('M')
        monthly_cases = disease_df.groupby('month').size()
        
        # Calculate trend
        if len(monthly_cases) > 1:
            recent_avg = monthly_cases.tail(3).mean()
            older_avg = monthly_cases.head(3).mean()
            trend_direction = "increasing" if recent_avg > older_avg * 1.1 else "decreasing" if recent_avg < older_avg * 0.9 else "stable"
        else:
            trend_direction = "insufficient_data"
        
        # Compare with predictions
        predicted_total = sum([sum(p.predicted_cases.values()) for p in disease_predictions if p.predicted_cases])
        historical_monthly_avg = monthly_cases.mean() if len(monthly_cases) > 0 else 0
        
        return {
            'historical_monthly_average': round(historical_monthly_avg, 1),
            'predicted_monthly_total': round(predicted_total, 1),
            'trend_direction': trend_direction,
            'data_period_months': len(monthly_cases),
            'comparison_note': f"Predictions suggest {predicted_total:.0f} cases vs historical average of {historical_monthly_avg:.1f} per month"
        }
    
    return {
        'historical_monthly_average': 0,
        'predicted_monthly_total': 0,
        'trend_direction': 'no_data',
        'data_period_months': 0,
        'comparison_note': 'Insufficient historical data for comparison'
    }

def _generate_seasonal_analysis(
    disease_df: pd.DataFrame, 
    disease_predictions: List[DiseaseOutbreakPrediction]
) -> Dict[str, Any]:
    """Generate seasonal pattern analysis"""
    
    seasonal_patterns = {}
    
    for prediction in disease_predictions:
        pattern = prediction.seasonal_pattern
        if pattern not in seasonal_patterns:
            seasonal_patterns[pattern] = []
        seasonal_patterns[pattern].append(prediction.disease_name)
    
    # Current season analysis
    current_month = datetime.now().month
    if current_month in [12, 1, 2]:
        current_season = "winter"
    elif current_month in [3, 4, 5]:
        current_season = "spring"
    elif current_month in [6, 7, 8]:
        current_season = "summer"
    else:
        current_season = "autumn"
    
    # Identify diseases that peak in current season
    current_season_diseases = []
    for prediction in disease_predictions:
        if f"{current_season}_peak" in prediction.seasonal_pattern:
            current_season_diseases.append(prediction.disease_name)
    
    return {
        'current_season': current_season,
        'seasonal_pattern_distribution': seasonal_patterns,
        'current_season_risk_diseases': current_season_diseases,
        'seasonal_recommendations': f"Monitor {len(current_season_diseases)} diseases with {current_season} seasonal patterns" if current_season_diseases else "No specific seasonal risks identified for current season"
    }

def _create_empty_disease_prediction_response(
    prediction_id: str, 
    generated_at: str, 
    request: DiseasePredictionRequest
) -> DiseasePredictionResponse:
    """Create empty response when no predictions can be generated"""
    
    return DiseasePredictionResponse(
        prediction_id=prediction_id,
        generated_at=generated_at,
        forecast_period=request.forecast_period,
        disease_predictions=[],
        prediction_summary=DiseasePredictionSummary(
            total_diseases_analyzed=0,
            high_risk_diseases=0,
            critical_risk_diseases=0,
            average_outbreak_probability=0.0,
            highest_risk_disease="None",
            highest_risk_probability=0.0,
            seasonal_diseases_count=0,
            dominant_seasonal_pattern="no_pattern",
            average_confidence=0.0,
            reliable_predictions_count=0,
            risk_distribution={'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
        ),
        high_risk_alerts=[],
        outbreak_warnings=[],
        prevention_recommendations=[],
        resource_recommendations=[],
        model_performance={'error': 'No data available for analysis'},
        data_quality={'completeness': 0, 'overall_quality': 0},
        historical_comparison={'note': 'No historical data available'},
        seasonal_analysis={'note': 'No seasonal analysis possible'},
        data_sources=['medical_records'],
        limitations=['No historical medical data available for analysis'],
        confidence_notes=['Unable to generate predictions due to lack of data']
    )

def _calculate_age(date_of_birth: str) -> int:
    """Calculate age from date of birth"""
    
    if not date_of_birth:
        return 35  # Default age
    
    try:
        birth_date = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
        today = datetime.now().date()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return max(0, min(120, age))  # Reasonable age bounds
    except:
        return 35  # Default age if parsing fails