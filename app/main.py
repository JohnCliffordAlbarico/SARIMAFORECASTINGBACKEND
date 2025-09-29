from fastapi import FastAPI
from app.services.supabase import supabase

app = FastAPI(title="HealthGuard API")

@app.get("/")
def read_root():
    # Just to test connection
    return {"message": "Hello from HealthGuard API 🚑"}

@app.get("/users")
def get_users():
    data = supabase.table("users").select("*").execute()
    return data.data
