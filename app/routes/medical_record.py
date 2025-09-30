# app/routes/medical_record.py
from fastapi import APIRouter, HTTPException, Depends, Query
from app.services.supabase import supabase
from app.utils.middleware import limiter
from app.models.schemas import MedicalRecordResponse, ErrorResponse
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=MedicalRecordResponse)
def get_medical_records(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    patient_id: Optional[int] = Query(None, description="Filter by specific patient ID"),
    dep=Depends(limiter.limit("30/second"))
):
    """
    Retrieve medical records with filtering and pagination
    """
    try:
        # Validate date parameters
        if start_date:
            try:
                start_date_parsed = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
        
        if end_date:
            try:
                end_date_parsed = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
        
        if start_date and end_date:
            if start_date_parsed > end_date_parsed:
                raise HTTPException(status_code=400, detail="start_date cannot be after end_date")
        
        # Build query
        query = supabase.table("medical_records").select(
            "id, patient_id, visit_date, chief_complaint, medication"
        )
        
        # Apply filters
        if patient_id:
            query = query.eq("patient_id", patient_id)
        
        if start_date:
            query = query.gte("visit_date", start_date)
        
        if end_date:
            query = query.lte("visit_date", end_date)
        
        # Apply pagination and ordering
        response = query.order("visit_date", desc=True).range(offset, offset + limit - 1).execute()
        
        if response.error:
            logger.error(f"Database error: {response.error.message}")
            raise HTTPException(status_code=500, detail=f"Database error: {response.error.message}")
        
        # Get total count for pagination info
        count_response = supabase.table("medical_records").select("id", count="exact").execute()
        total_records = count_response.count if count_response.count else 0
        
        # Calculate date range of returned data
        date_range = None
        if response.data:
            dates = [record['visit_date'] for record in response.data if record['visit_date']]
            if dates:
                date_range = {
                    "start": min(dates),
                    "end": max(dates)
                }
        
        result = {
            "data": response.data,
            "total_records": total_records,
            "date_range": date_range
        }
        
        logger.info(f"Retrieved {len(response.data)} medical records")
        return MedicalRecordResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve medical records: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve medical records: {str(e)}")

@router.get("/summary")
def get_medical_records_summary(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    dep=Depends(limiter.limit("10/minute"))
):
    """
    Get summary statistics for medical records
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get records in date range
        response = supabase.table("medical_records").select(
            "id, patient_id, visit_date, chief_complaint"
        ).gte("visit_date", start_date.strftime("%Y-%m-%d")).lte(
            "visit_date", end_date.strftime("%Y-%m-%d")
        ).execute()
        
        if response.error:
            raise HTTPException(status_code=500, detail=response.error.message)
        
        records = response.data
        
        # Calculate statistics
        total_visits = len(records)
        unique_patients = len(set(record['patient_id'] for record in records if record['patient_id']))
        
        # Group by date for daily statistics
        daily_counts = {}
        for record in records:
            visit_date = record['visit_date']
            if visit_date:
                date_key = visit_date.split('T')[0]  # Extract date part
                daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
        
        avg_daily_visits = sum(daily_counts.values()) / max(len(daily_counts), 1)
        
        # Most common complaints
        complaints = [record['chief_complaint'] for record in records if record['chief_complaint']]
        complaint_counts = {}
        for complaint in complaints:
            complaint_counts[complaint] = complaint_counts.get(complaint, 0) + 1
        
        top_complaints = sorted(complaint_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary = {
            "period_days": days,
            "date_range": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d")
            },
            "total_visits": total_visits,
            "unique_patients": unique_patients,
            "average_daily_visits": round(avg_daily_visits, 2),
            "busiest_day": max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else None,
            "top_complaints": [{"complaint": complaint, "count": count} for complaint, count in top_complaints]
        }
        
        logger.info(f"Generated summary for {days} days: {total_visits} visits")
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")
