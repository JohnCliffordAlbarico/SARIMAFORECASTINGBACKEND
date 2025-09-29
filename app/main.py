
from fastapi import FastAPI, Request
from app.services.supabase import supabase
from app.utils.middleware import init_middlewares, limiter  # Import middleware
from app.routes import forecast # imports the /routes/forecast.py file (package by __init__.py)

# Create FastAPI instance
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

app.include_router(forecast.router, prefix="/api/forecast")