from fastapi import APIRouter, HTTPException, Depends
from app.services.supabase import supabase
from app.utils.middleware import limiter

router = APIRouter()

@router.get("/")
def get_medicine_stock_suggestion(dep=Depends(limiter.limit("30/second"))):
    try:
        # Inventory
        inventory = supabase.table("medicine_inventory").select(
            "id, medicine_id, current_stock, minimum_stock_level, maximum_stock_level, expiry_date"
        ).execute()
        if inventory.error:
            raise HTTPException(status_code=500, detail=inventory.error.message)

        # Prescriptions
        prescriptions = supabase.table("medicine_prescriptions").select(
            "id, medicine_id, quantity_prescribed, quantity_dispensed, created_at"
        ).execute()
        if prescriptions.error:
            raise HTTPException(status_code=500, detail=prescriptions.error.message)

        # Stock movements
        movements = supabase.table("medicine_stock_movements").select(
            "inventory_id, movement_type, quantity, movement_date"
        ).execute()
        if movements.error:
            raise HTTPException(status_code=500, detail=movements.error.message)

        return {
            "inventory": inventory.data,
            "prescriptions": prescriptions.data,
            "movements": movements.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
