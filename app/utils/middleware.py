# app/utils/middleware.py
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from fastapi import FastAPI

limiter = Limiter(key_func=get_remote_address)

def init_middlewares(app: FastAPI):
    # Attach the limiter to FastAPI state
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
