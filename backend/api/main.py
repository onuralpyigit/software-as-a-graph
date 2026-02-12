"""
Distributed System Graph API

FastAPI application exposing graph generation, import, analysis, and query capabilities.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import routers
from api.routers import (
    health,
    graph,
    analysis,
    components,
    statistics,
    simulation,
    classification,
    validation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Distributed System Graph API",
    description="API for generating, analyzing, and querying distributed system graphs",
    version="1.0.0"
)

# Configure CORS to allow frontend access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(graph.router)
app.include_router(analysis.router)
app.include_router(components.router)
app.include_router(statistics.router)
app.include_router(simulation.router)
app.include_router(classification.router)
app.include_router(validation.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
