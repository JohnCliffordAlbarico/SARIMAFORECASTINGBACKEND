from fastapi import APIRouter, HTTPException, Depends, Query
from app.services.supabase import supabase
from app.utils.middleware import limiter
from app.models.schemas import MedicineStockResponse, ErrorResponse
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=MedicineStockResponse)
def get_medicine_stock_data(
    medicine_id: Optional[int] = Query(None, description="Filter by specific medicine ID"),
    include_expired: bool = Query(False, description="Include expired medicines"),
    low_stock_only: bool = Query(False, description="Show only low stock items"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records")
):
    """
    Retrieve medicine stock data with filtering options
    """
    try:
        # Build inventory query
        inventory_query = supabase.table("medicine_inventory").select(
            "id, medicine_id, current_stock, minimum_stock_level, maximum_stock_level, expiry_date"
        )
        
        if medicine_id:
            inventory_query = inventory_query.eq("medicine_id", medicine_id)
        
        if not include_expired:
            current_date = datetime.now().strftime("%Y-%m-%d")
            inventory_query = inventory_query.gte("expiry_date", current_date)
        
        inventory_response = inventory_query.limit(limit).execute()
        
        if not hasattr(inventory_response, 'data') or inventory_response.data is None:
            logger.error("Database connection failed for inventory query")
            raise HTTPException(status_code=500, detail="Database connection failed")

        inventory_data = inventory_response.data
        
        # Filter low stock items if requested
        if low_stock_only:
            inventory_data = [
                item for item in inventory_data 
                if item['current_stock'] <= item['minimum_stock_level']
            ]

        # Get prescriptions data
        prescriptions_query = supabase.table("medicine_prescriptions").select(
            "id, medicine_id, quantity_prescribed, quantity_dispensed, created_at"
        )
        
        if medicine_id:
            prescriptions_query = prescriptions_query.eq("medicine_id", medicine_id)
        
        prescriptions_response = prescriptions_query.limit(limit).execute()
        
        if not hasattr(prescriptions_response, 'data') or prescriptions_response.data is None:
            logger.error("Database connection failed for prescriptions query")
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Get stock movements data
        movements_query = supabase.table("medicine_stock_movements").select(
            "inventory_id, movement_type, quantity, movement_date"
        )
        
        if medicine_id and inventory_data:
            # Filter movements by inventory IDs
            inventory_ids = [item['id'] for item in inventory_data]
            movements_query = movements_query.in_("inventory_id", inventory_ids)
        
        movements_response = movements_query.order("movement_date", desc=True).limit(limit).execute()
        
        if not hasattr(movements_response, 'data') or movements_response.data is None:
            logger.error("Database connection failed for movements query")
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Calculate summary statistics
        summary = _calculate_stock_summary(inventory_data, prescriptions_response.data, movements_response.data)

        result = {
            "inventory": inventory_data,
            "prescriptions": prescriptions_response.data,
            "movements": movements_response.data,
            "summary": summary
        }
        
        logger.info(f"Retrieved stock data: {len(inventory_data)} inventory items")
        return MedicineStockResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve stock data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stock data: {str(e)}")

def _calculate_stock_summary(inventory_data, prescriptions_data, movements_data):
    """Calculate summary statistics for stock data"""
    try:
        total_medicines = len(inventory_data)
        total_stock = sum(item['current_stock'] for item in inventory_data if item['current_stock'])
        
        # Low stock items
        low_stock_items = [
            item for item in inventory_data 
            if item['current_stock'] <= item['minimum_stock_level']
        ]
        
        # Expired items
        current_date = datetime.now()
        expired_items = []
        expiring_soon = []
        
        for item in inventory_data:
            if item['expiry_date']:
                try:
                    expiry_date = datetime.strptime(item['expiry_date'], "%Y-%m-%d")
                    if expiry_date < current_date:
                        expired_items.append(item)
                    elif expiry_date < current_date + timedelta(days=30):
                        expiring_soon.append(item)
                except ValueError:
                    continue
        
        # Recent movements
        recent_movements = [
            movement for movement in movements_data
            if movement['movement_date'] and 
            datetime.strptime(movement['movement_date'].split('T')[0], "%Y-%m-%d") >= current_date - timedelta(days=7)
        ]
        
        # Calculate movement totals
        total_in = sum(
            movement['quantity'] for movement in recent_movements 
            if movement['movement_type'] == 'in'
        )
        total_out = sum(
            movement['quantity'] for movement in recent_movements 
            if movement['movement_type'] == 'out'
        )
        
        summary = {
            "total_medicines": total_medicines,
            "total_stock_units": total_stock,
            "low_stock_count": len(low_stock_items),
            "expired_count": len(expired_items),
            "expiring_soon_count": len(expiring_soon),
            "recent_movements": {
                "total_in": total_in,
                "total_out": total_out,
                "net_change": total_in - total_out,
                "period_days": 7
            },
            "alerts": []
        }
        
        # Generate alerts
        if low_stock_items:
            summary["alerts"].append({
                "type": "low_stock",
                "message": f"{len(low_stock_items)} medicines are below minimum stock level",
                "severity": "warning"
            })
        
        if expired_items:
            summary["alerts"].append({
                "type": "expired",
                "message": f"{len(expired_items)} medicines have expired",
                "severity": "critical"
            })
        
        if expiring_soon:
            summary["alerts"].append({
                "type": "expiring_soon",
                "message": f"{len(expiring_soon)} medicines expire within 30 days",
                "severity": "info"
            })
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to calculate summary: {str(e)}")
        return {"error": f"Summary calculation failed: {str(e)}"}

@router.get("/alerts")
def get_stock_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: info, warning, critical")
):
    """
    Get stock alerts and recommendations
    """
    try:
        # Get all inventory data
        inventory_response = supabase.table("medicine_inventory").select(
            "id, medicine_id, current_stock, minimum_stock_level, maximum_stock_level, expiry_date"
        ).execute()
        
        if not hasattr(inventory_response, 'data') or inventory_response.data is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        inventory_data = inventory_response.data
        current_date = datetime.now()
        alerts = []
        
        for item in inventory_data:
            medicine_id = item['medicine_id']
            current_stock = item['current_stock'] or 0
            min_level = item['minimum_stock_level'] or 0
            max_level = item['maximum_stock_level'] or 0
            
            # Low stock alert
            if current_stock <= min_level:
                alert_severity = "critical" if current_stock == 0 else "warning"
                alerts.append({
                    "medicine_id": medicine_id,
                    "type": "low_stock",
                    "severity": alert_severity,
                    "message": f"Stock level ({current_stock}) is below minimum ({min_level})",
                    "recommendation": f"Reorder {max_level - current_stock} units",
                    "current_stock": current_stock,
                    "minimum_level": min_level
                })
            
            # Expiry alerts
            if item['expiry_date']:
                try:
                    expiry_date = datetime.strptime(item['expiry_date'], "%Y-%m-%d")
                    days_to_expiry = (expiry_date - current_date).days
                    
                    if days_to_expiry < 0:
                        alerts.append({
                            "medicine_id": medicine_id,
                            "type": "expired",
                            "severity": "critical",
                            "message": f"Medicine expired {abs(days_to_expiry)} days ago",
                            "recommendation": "Remove from inventory immediately",
                            "expiry_date": item['expiry_date'],
                            "days_overdue": abs(days_to_expiry)
                        })
                    elif days_to_expiry <= 30:
                        alerts.append({
                            "medicine_id": medicine_id,
                            "type": "expiring_soon",
                            "severity": "info" if days_to_expiry > 7 else "warning",
                            "message": f"Medicine expires in {days_to_expiry} days",
                            "recommendation": "Use soon or consider discounting",
                            "expiry_date": item['expiry_date'],
                            "days_remaining": days_to_expiry
                        })
                except ValueError:
                    continue
        
        # Filter by severity if requested
        if severity:
            alerts = [alert for alert in alerts if alert['severity'] == severity]
        
        # Sort by severity (critical first)
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        result = {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "summary": {
                "critical": len([a for a in alerts if a['severity'] == 'critical']),
                "warning": len([a for a in alerts if a['severity'] == 'warning']),
                "info": len([a for a in alerts if a['severity'] == 'info'])
            },
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Generated {len(alerts)} stock alerts")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate alerts: {str(e)}")
