# app/routes/forecast_routes.py
from fastapi import APIRouter, HTTPException, Depends
from app.services.supabase import supabase
from app.utils.middleware import limiter

router = APIRouter()


@router.get("/")
def get_medical_records(dep=Depends(limiter.limit("30/second"))):
    try:
        response = supabase.table("medical_records").select(
            "id, patient_id, visit_date, chief_complaint, medication"
        ).execute()

        if response.error:
            raise HTTPException(status_code=500, detail=response.error.message)

        return {"data": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
