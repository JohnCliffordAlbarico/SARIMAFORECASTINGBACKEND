
# app/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.services.supabase import supabase
from app.utils.middleware import init_middlewares, limiter
from app.utils.logging_config import setup_logging
from app.models.schemas import HealthCheckResponse, ErrorResponse
from app.routes import forecast 
from app.routes import medical_record
from app.routes import medicine_stock
from app.routes import health_forecasting
import logging
from datetime import datetime

# Setup logging
logger = setup_logging()

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="HealthGuard Forecasting API",
    description="Advanced medical forecasting and inventory management system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize middleware (rate limiter)
init_middlewares(app)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"path": str(request.url.path)}
        ).dict()
    )

# Health check endpoint
@app.get("/", response_model=HealthCheckResponse)
@limiter.limit("30/second")  
def health_check(request: Request):
    """
    Health check endpoint with database connectivity test
    """
    try:
        # Test database connection
        test_response = supabase.table("users").select("id").limit(1).execute()
        
        # Check if response is successful (newer Supabase client)
        database_connected = hasattr(test_response, 'data') and test_response.data is not None
        
        if database_connected:
            status = "healthy"
            message = "HealthGuard API is running successfully 🚑"
        else:
            status = "degraded"
            message = "API is running but database connection issues detected"
            
        logger.info(f"Health check: {status}")
        
        return HealthCheckResponse(
            status=status,
            message=message,
            database_connected=database_connected
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            database_connected=False
        )

# API link for testing connections between the database and the backend
@app.get("/users")
@limiter.limit("30/second") 
def get_users(request: Request):
    """
    Test endpoint for database connectivity
    """
    try:
        response = supabase.table("users").select("*").execute()
        
        # Check if response has data (newer Supabase client)
        if not hasattr(response, 'data') or response.data is None:
            logger.error("Database connection failed or no data returned")
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        logger.info(f"Retrieved {len(response.data)} users")
        return {"data": response.data, "count": len(response.data)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /users: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# API routes
app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecasting"])
app.include_router(medical_record.router, prefix="/api/medical_record", tags=["Medical Records"])
app.include_router(medicine_stock.router, prefix="/api/medicine_stock", tags=["Medicine Stock"])
app.include_router(health_forecasting.router, prefix="/api", tags=["Health Forecasting"])

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("HealthGuard API startup completed")
    logger.info("Available endpoints:")
    logger.info("  - GET / (Health Check)")
    logger.info("  - GET /api/forecast/ (Generate Forecasts)")
    logger.info("  - GET /api/forecast/diagnostics (Model Diagnostics)")
    logger.info("  - GET /api/medical_record/ (Medical Records)")
    logger.info("  - GET /api/medical_record/summary (Records Summary)")
    logger.info("  - GET /api/medicine_stock/ (Stock Data)")
    logger.info("  - GET /api/medicine_stock/alerts (Stock Alerts)")
    logger.info("  - POST /api/forecast/disease-trends (Disease Trends Forecasting)")
    logger.info("  - POST /api/forecast/patient-visits (Patient Visit Forecasting)")
    logger.info("  - POST /api/forecast/health-patterns (Health Pattern Analysis)")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("HealthGuard API shutting down")
