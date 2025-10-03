# app/services/data_aggregation.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from enum import Enum

from .supabase import supabase
from .disease_classifier import disease_classifier
from ..models.schemas import TimeAggregation, HistoricalPeriod

logger = logging.getLogger(__name__)

class DataAggregationService:
    """
    Service for efficient data aggregation and time-based queries
    Optimizes database queries and provides pre-aggregated data for forecasting
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    async def get_aggregated_patient_volume(self, 
                                          time_aggregation: TimeAggregation,
                                          historical_period: HistoricalPeriod,
                                          department_filter: Optional[str] = None) -> pd.DataFrame:
        """Get aggregated patient volume data"""
        
        cache_key = f"patient_volume_{time_aggregation.value}_{historical_period.value}_{department_filter}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Calculate date range
            cutoff_date = self._get_cutoff_date(historical_period)
            
            # Build optimized query
            query = supabase.table("medical_records").select(
                "visit_date, patient_id"
            ).gte("visit_date", cutoff_date).order("visit_date")
            
            # Apply department filter if needed
            if department_filter:
                # Join with appointments and users to filter by department
                query = query.select(
                    "visit_date, patient_id, medical_records!inner(appointments!inner(doctor_id, users!inner(department)))"
                ).eq("appointments.users.department", department_filter)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame(columns=['date', 'volume'])
            
            # Convert to DataFrame and aggregate
            df = pd.DataFrame(response.data)
            df['visit_date'] = pd.to_datetime(df['visit_date'])
            
            # Aggregate based on time period
            aggregated_df = self._aggregate_by_time_period(df, 'visit_date', time_aggregation)
            
            # Cache the result
            self._cache_data(cache_key, aggregated_df)
            
            return aggregated_df
            
        except Exception as e:
            logger.error(f"Error aggregating patient volume data: {str(e)}")
            return pd.DataFrame(columns=['date', 'volume'])
    
    async def get_aggregated_disease_trends(self,
                                          time_aggregation: TimeAggregation,
                                          historical_period: HistoricalPeriod,
                                          disease_category: Optional[str] = None) -> pd.DataFrame:
        """Get aggregated disease trends data with enhanced classification"""
        
        cache_key = f"disease_trends_enhanced_{time_aggregation.value}_{historical_period.value}_{disease_category}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            cutoff_date = self._get_cutoff_date(historical_period)
            
            # Query medical records with disease information and patient data
            response = supabase.table("medical_records").select(
                "visit_date, chief_complaint, management, patients!inner(date_of_birth)"
            ).gte("visit_date", cutoff_date).not_.is_("chief_complaint", "null").order("visit_date").execute()
            
            if not response.data:
                return pd.DataFrame(columns=['date', 'volume', 'category_breakdown'])
            
            df = pd.DataFrame(response.data)
            df['visit_date'] = pd.to_datetime(df['visit_date'])
            
            # Calculate patient ages
            df['patient_age'] = df.apply(
                lambda row: self._calculate_age(row.get('patients', {}).get('date_of_birth')) 
                if row.get('patients') else None, axis=1
            )
            
            # Enhanced disease classification
            df['disease_classification'] = df.apply(
                lambda row: disease_classifier.classify_disease(
                    chief_complaint=row['chief_complaint'],
                    management=row.get('management', ''),
                    patient_age=row.get('patient_age')
                ), axis=1
            )
            
            # Extract classification details
            df['category_code'] = df['disease_classification'].apply(lambda x: x['category_code'])
            df['category_name'] = df['disease_classification'].apply(lambda x: x['category_name'])
            df['severity'] = df['disease_classification'].apply(lambda x: x['severity'])
            df['confidence'] = df['disease_classification'].apply(lambda x: x['confidence'])
            df['seasonal_pattern'] = df['disease_classification'].apply(
                lambda x: x['metadata']['seasonal_pattern']
            )
            
            # Filter by disease category if specified
            if disease_category:
                df = self._filter_by_enhanced_category(df, disease_category)
            
            # Aggregate with enhanced metadata
            aggregated_df = self._aggregate_disease_trends_enhanced(df, time_aggregation)
            
            self._cache_data(cache_key, aggregated_df)
            return aggregated_df
            
        except Exception as e:
            logger.error(f"Error aggregating enhanced disease trends data: {str(e)}")
            return pd.DataFrame(columns=['date', 'volume', 'category_breakdown'])
    
    async def get_aggregated_medicine_demand(self,
                                           time_aggregation: TimeAggregation,
                                           historical_period: HistoricalPeriod,
                                           medicine_category: Optional[str] = None) -> pd.DataFrame:
        """Get aggregated medicine demand data"""
        
        cache_key = f"medicine_demand_{time_aggregation.value}_{historical_period.value}_{medicine_category}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            cutoff_date = self._get_cutoff_date(historical_period)
            
            # Query prescription data
            query = supabase.table("medicine_prescriptions").select(
                "created_at, quantity_prescribed, medicine_id"
            ).gte("created_at", cutoff_date).order("created_at")
            
            # Filter by medicine category if specified
            if medicine_category:
                # Join with medicines table to filter by drug class
                query = query.select(
                    "created_at, quantity_prescribed, medicine_id, medicines!inner(drug_class)"
                ).ilike("medicines.drug_class", f"%{medicine_category}%")
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame(columns=['date', 'volume'])
            
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['volume'] = df['quantity_prescribed']
            
            # Aggregate
            aggregated_df = self._aggregate_by_time_period(df, 'created_at', time_aggregation, value_col='volume')
            
            self._cache_data(cache_key, aggregated_df)
            return aggregated_df
            
        except Exception as e:
            logger.error(f"Error aggregating medicine demand data: {str(e)}")
            return pd.DataFrame(columns=['date', 'volume'])
    
    async def get_aggregated_appointment_patterns(self,
                                                time_aggregation: TimeAggregation,
                                                historical_period: HistoricalPeriod) -> pd.DataFrame:
        """Get aggregated appointment patterns data"""
        
        cache_key = f"appointment_patterns_{time_aggregation.value}_{historical_period.value}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            cutoff_date = self._get_cutoff_date(historical_period)
            
            response = supabase.table("appointments").select(
                "scheduled_date, status, created_at"
            ).gte("scheduled_date", cutoff_date).order("scheduled_date").execute()
            
            if not response.data:
                return pd.DataFrame(columns=['date', 'volume'])
            
            df = pd.DataFrame(response.data)
            df['scheduled_date'] = pd.to_datetime(df['scheduled_date'])
            
            # Aggregate
            aggregated_df = self._aggregate_by_time_period(df, 'scheduled_date', time_aggregation)
            
            self._cache_data(cache_key, aggregated_df)
            return aggregated_df
            
        except Exception as e:
            logger.error(f"Error aggregating appointment patterns data: {str(e)}")
            return pd.DataFrame(columns=['date', 'volume'])
    
    async def get_aggregated_resource_utilization(self,
                                                time_aggregation: TimeAggregation,
                                                historical_period: HistoricalPeriod) -> pd.DataFrame:
        """Get aggregated resource utilization data"""
        
        cache_key = f"resource_utilization_{time_aggregation.value}_{historical_period.value}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            cutoff_date = self._get_cutoff_date(historical_period)
            
            # Use appointments as proxy for resource utilization
            response = supabase.table("appointments").select(
                "scheduled_date, status"
            ).gte("scheduled_date", cutoff_date).order("scheduled_date").execute()
            
            if not response.data:
                return pd.DataFrame(columns=['date', 'volume'])
            
            df = pd.DataFrame(response.data)
            df['scheduled_date'] = pd.to_datetime(df['scheduled_date'])
            
            # Weight by appointment status (completed = 1, others = 0.5)
            df['weight'] = df['status'].apply(lambda x: 1.0 if x == 'completed' else 0.5)
            
            # Aggregate with weights
            aggregated_df = self._aggregate_by_time_period(df, 'scheduled_date', time_aggregation, value_col='weight')
            
            self._cache_data(cache_key, aggregated_df)
            return aggregated_df
            
        except Exception as e:
            logger.error(f"Error aggregating resource utilization data: {str(e)}")
            return pd.DataFrame(columns=['date', 'volume'])
    
    async def get_aggregated_revenue_data(self,
                                        time_aggregation: TimeAggregation,
                                        historical_period: HistoricalPeriod) -> pd.DataFrame:
        """Get aggregated revenue data"""
        
        cache_key = f"revenue_data_{time_aggregation.value}_{historical_period.value}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            cutoff_date = self._get_cutoff_date(historical_period)
            
            # Use completed appointments as revenue proxy
            response = supabase.table("appointments").select(
                "scheduled_date, status"
            ).gte("scheduled_date", cutoff_date).eq("status", "completed").order("scheduled_date").execute()
            
            if not response.data:
                return pd.DataFrame(columns=['date', 'volume'])
            
            df = pd.DataFrame(response.data)
            df['scheduled_date'] = pd.to_datetime(df['scheduled_date'])
            df['revenue'] = 100.0  # Placeholder: $100 per completed appointment
            
            # Aggregate
            aggregated_df = self._aggregate_by_time_period(df, 'scheduled_date', time_aggregation, value_col='revenue')
            
            self._cache_data(cache_key, aggregated_df)
            return aggregated_df
            
        except Exception as e:
            logger.error(f"Error aggregating revenue data: {str(e)}")
            return pd.DataFrame(columns=['date', 'volume'])
    
    def _get_cutoff_date(self, historical_period: HistoricalPeriod) -> str:
        """Calculate cutoff date based on historical period"""
        
        period_days = {
            HistoricalPeriod.ONE_YEAR: 365,
            HistoricalPeriod.TWO_YEARS: 730,
            HistoricalPeriod.THREE_YEARS: 1095,
            HistoricalPeriod.FOUR_YEARS: 1460,
            HistoricalPeriod.FIVE_YEARS: 1825
        }
        
        days = period_days.get(historical_period, 730)
        cutoff_date = datetime.now() - timedelta(days=days)
        return cutoff_date.strftime('%Y-%m-%d')
    
    def _aggregate_by_time_period(self, df: pd.DataFrame, 
                                 date_col: str, 
                                 time_aggregation: TimeAggregation,
                                 value_col: str = None) -> pd.DataFrame:
        """Aggregate DataFrame by time period"""
        
        if df.empty:
            return pd.DataFrame(columns=['date', 'volume'])
        
        # Set date column as index
        df_copy = df.copy()
        df_copy.set_index(date_col, inplace=True)
        
        # Determine aggregation frequency
        freq_map = {
            TimeAggregation.DAILY: 'D',
            TimeAggregation.WEEKLY: 'W',
            TimeAggregation.MONTHLY: 'M',
            TimeAggregation.QUARTERLY: 'Q',
            TimeAggregation.YEARLY: 'Y'
        }
        
        freq = freq_map.get(time_aggregation, 'D')
        
        # Aggregate data
        if value_col and value_col in df_copy.columns:
            # Sum the specified value column
            aggregated = df_copy.resample(freq)[value_col].sum()
        else:
            # Count records
            aggregated = df_copy.resample(freq).size()
        
        # Convert to DataFrame
        result_df = pd.DataFrame({
            'date': aggregated.index,
            'volume': aggregated.values
        })
        
        # Fill missing dates with zeros
        if not result_df.empty:
            date_range = pd.date_range(
                start=result_df['date'].min(),
                end=result_df['date'].max(),
                freq=freq
            )
            
            result_df = result_df.set_index('date').reindex(date_range, fill_value=0).reset_index()
            result_df.columns = ['date', 'volume']
        
        return result_df
    
    def _filter_by_disease_category(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Filter medical records by disease category (legacy method)"""
        
        disease_keywords = {
            'respiratory': ['cough', 'fever', 'cold', 'flu', 'pneumonia', 'asthma', 'bronchitis', 'respiratory'],
            'gastrointestinal': ['stomach', 'diarrhea', 'vomiting', 'nausea', 'abdominal', 'gastro'],
            'infectious': ['infection', 'viral', 'bacterial', 'fungal', 'infectious'],
            'chronic': ['diabetes', 'hypertension', 'heart', 'chronic', 'blood pressure'],
            'musculoskeletal': ['pain', 'back', 'joint', 'muscle', 'arthritis', 'bone'],
            'cardiovascular': ['heart', 'cardiac', 'blood pressure', 'chest pain', 'cardiovascular'],
            'neurological': ['headache', 'migraine', 'seizure', 'neurological', 'brain'],
            'dermatological': ['skin', 'rash', 'dermatitis', 'allergy', 'dermatological']
        }
        
        keywords = disease_keywords.get(category.lower(), [])
        if not keywords:
            return df
        
        # Create pattern for case-insensitive matching
        pattern = '|'.join(keywords)
        
        # Filter based on chief complaint or management containing keywords
        mask = (
            df['chief_complaint'].str.lower().str.contains(pattern, na=False) |
            df['management'].str.lower().str.contains(pattern, na=False)
        )
        
        return df[mask]
    
    def _filter_by_enhanced_category(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Filter medical records by enhanced disease category"""
        
        # Map legacy categories to new category codes
        category_mapping = {
            'respiratory': ['RESP_UPPER', 'RESP_LOWER'],
            'gastrointestinal': ['GI_ACUTE', 'GI_CHRONIC'],
            'infectious': ['INF_VIRAL', 'INF_BACTERIAL'],
            'chronic': ['GI_CHRONIC', 'CARDIO', 'ENDO_META', 'MSK_CHRONIC'],
            'musculoskeletal': ['MSK_ACUTE', 'MSK_CHRONIC'],
            'cardiovascular': ['CARDIO'],
            'neurological': ['NEURO'],
            'dermatological': ['DERM'],
            'mental_health': ['MENTAL'],
            'pediatric': ['PEDIATRIC'],
            'reproductive': ['REPRO']
        }
        
        # Get category codes for filtering
        target_codes = category_mapping.get(category.lower(), [category.upper()])
        
        # Filter by category code
        if 'category_code' in df.columns:
            return df[df['category_code'].isin(target_codes)]
        else:
            # Fallback to legacy filtering
            return self._filter_by_disease_category(df, category)
    
    def _calculate_age(self, date_of_birth: str) -> Optional[int]:
        """Calculate age from date of birth"""
        
        if not date_of_birth:
            return None
        
        try:
            birth_date = pd.to_datetime(date_of_birth)
            today = pd.Timestamp.now()
            age = (today - birth_date).days // 365
            return max(0, age)  # Ensure non-negative age
        except Exception:
            return None
    
    def _aggregate_disease_trends_enhanced(self, df: pd.DataFrame, 
                                         time_aggregation: TimeAggregation) -> pd.DataFrame:
        """Aggregate disease trends with enhanced metadata"""
        
        # Basic time aggregation
        aggregated_df = self._aggregate_by_time_period(df, 'visit_date', time_aggregation)
        
        # Add category breakdown for each time period
        if not df.empty and 'category_code' in df.columns:
            # Group by date and category
            freq_map = {
                TimeAggregation.DAILY: 'D',
                TimeAggregation.WEEKLY: 'W',
                TimeAggregation.MONTHLY: 'M',
                TimeAggregation.QUARTERLY: 'Q',
                TimeAggregation.YEARLY: 'Y'
            }
            
            freq = freq_map.get(time_aggregation, 'D')
            df_grouped = df.set_index('visit_date').groupby([
                pd.Grouper(freq=freq), 'category_code'
            ]).agg({
                'severity': lambda x: x.mode().iloc[0] if not x.empty else 'mild',
                'confidence': 'mean',
                'seasonal_pattern': lambda x: x.mode().iloc[0] if not x.empty else 'no_pattern'
            }).reset_index()
            
            # Create category breakdown for each date
            category_breakdown = df_grouped.groupby('visit_date').apply(
                lambda group: {
                    'categories': group['category_code'].tolist(),
                    'avg_confidence': group['confidence'].mean(),
                    'dominant_severity': group['severity'].mode().iloc[0] if not group.empty else 'mild',
                    'seasonal_patterns': group['seasonal_pattern'].unique().tolist()
                }
            ).to_dict()
            
            # Merge with aggregated data
            aggregated_df['category_breakdown'] = aggregated_df['date'].map(
                lambda x: category_breakdown.get(pd.Timestamp(x), {
                    'categories': [],
                    'avg_confidence': 0.0,
                    'dominant_severity': 'mild',
                    'seasonal_patterns': []
                })
            )
        else:
            # Add empty breakdown if no classification data
            aggregated_df['category_breakdown'] = [{}] * len(aggregated_df)
        
        return aggregated_df
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_ttl
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with timestamp"""
        
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        # Clean old cache entries (keep only last 50)
        if len(self.cache) > 50:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Data aggregation cache cleared")
    
    async def get_data_quality_metrics(self, 
                                     time_aggregation: TimeAggregation,
                                     historical_period: HistoricalPeriod) -> Dict[str, Any]:
        """Get data quality metrics for the specified parameters"""
        
        try:
            cutoff_date = self._get_cutoff_date(historical_period)
            
            # Check data availability across different tables
            medical_records_count = await self._count_records("medical_records", "visit_date", cutoff_date)
            appointments_count = await self._count_records("appointments", "scheduled_date", cutoff_date)
            prescriptions_count = await self._count_records("medicine_prescriptions", "created_at", cutoff_date)
            
            # Calculate completeness scores
            total_days = (datetime.now() - datetime.strptime(cutoff_date, '%Y-%m-%d')).days
            
            return {
                "data_availability": {
                    "medical_records": medical_records_count,
                    "appointments": appointments_count,
                    "prescriptions": prescriptions_count
                },
                "completeness_scores": {
                    "medical_records": min(1.0, medical_records_count / total_days),
                    "appointments": min(1.0, appointments_count / total_days),
                    "prescriptions": min(1.0, prescriptions_count / total_days)
                },
                "historical_period_days": total_days,
                "data_quality_overall": min(1.0, (medical_records_count + appointments_count) / (total_days * 2))
            }
            
        except Exception as e:
            logger.error(f"Error calculating data quality metrics: {str(e)}")
            return {"error": str(e)}
    
    async def _count_records(self, table: str, date_col: str, cutoff_date: str) -> int:
        """Count records in a table since cutoff date"""
        
        try:
            response = supabase.table(table).select("id", count="exact").gte(date_col, cutoff_date).execute()
            return response.count if hasattr(response, 'count') else len(response.data or [])
        except Exception:
            return 0

# Global instance
data_aggregation_service = DataAggregationService()
