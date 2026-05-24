# app/api/router.py
from fastapi import APIRouter

from app.api.endpoints import health, recommendations, clustering


api_router = APIRouter()


# http://127.0.0.1:8000/api/ai/health
# http://127.0.0.1:8000/api/ai/recommendations
# http://127.0.0.1:8000/api/ai/refresh/clustering
# http://127.0.0.1:8000/api/ai/refresh/popularity

# /api/ai/health (POST)
api_router.include_router(health.router, prefix="/ai", tags=["health"])

# /api/ai/recommendations (POST)
api_router.include_router(recommendations.router, prefix="/ai", tags=["recommendations"])

# /api/ai/refresh/clustering (POST)
# /api/ai/refresh/popularity (POST)
api_router.include_router(clustering.router, prefix="/ai", tags=["refresh"])
