
from fastapi import FastAPI, Request
from app.services.supabase import supabase
from app.utils.middleware import init_middlewares, limiter  # Import middleware
from app.routes import forecast 
from app.routes import medical_record
from app.routes import medicine_stock

app = FastAPI(title="HealthGuard API")

# Initialize middleware (rate limiter)
init_middlewares(app)

# Limits IP from fetching continuously 
@app.get("/")
@limiter.limit("30/second")  
def read_root(request: Request):
    return {"message": "Hello from HealthGuard API 🚑"}

# API link for testing connections between the database and the backend
@app.get("/users")
@limiter.limit("30/second") 
def get_users(request: Request):
    data = supabase.table("users").select("*").execute()
    return data.data


# API routes
app.include_router(forecast.router, prefix="/api/forecast")
app.include_router(medical_record.router, prefix="/api/medical_record")
app.include_router(medicine_stock.router, prefix = "/api/medicine_stock")
