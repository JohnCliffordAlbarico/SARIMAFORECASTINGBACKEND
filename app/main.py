
from fastapi import FastAPI, Request
from app.services.supabase import supabase
from app.utils.middleware import init_middlewares, limiter  # Import middleware

# Create FastAPI instance
app = FastAPI(title="HealthGuard API")

# Initialize middleware (rate limiter)
init_middlewares(app)

# Limits IP from fetching continuously 
@app.get("/")
@limiter.limit("30/second")  
def read_root(request: Request):
    return {"message": "Hello from HealthGuard API 🚑"}

@app.get("/users")
@limiter.limit("30/second") 
def get_users(request: Request):
    data = supabase.table("users").select("*").execute()
    return data.data
