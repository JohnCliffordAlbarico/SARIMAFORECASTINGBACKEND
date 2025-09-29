# app/services/supabase.py
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get Supabase credentials
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Optional: fail fast if env vars are missing
if not url or not key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables")

# Create Supabase client
supabase: Client = create_client(url, key)
