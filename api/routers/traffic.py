"""
Traffic simulation endpoints.

Provides analytical pub-sub network and broker load estimation for a set of
selected topics given a message frequency (Hz) and duration.
"""

from fastapi import APIRouter, Depends, HTTPException
import logging

from api.dependencies import get_repository
from api.models import (
    TrafficTopicsRequest,
    TopicsListResponse,
    AppsListResponse,
    TrafficSimulationRequest,
    TrafficSimulationResponse,
)
from saag.core.ports.graph_repository import IGraphRepository
from saag.simulation.traffic_simulator import TrafficSimulator

router = APIRouter(prefix="/api/v1/traffic", tags=["traffic"])
logger = logging.getLogger(__name__)


@router.post("/topics", response_model=TopicsListResponse)
async def list_topics(
    request: TrafficTopicsRequest,
    repo: IGraphRepository = Depends(get_repository),
):
    """
    Return all Topic nodes in the graph together with publisher/subscriber/broker metadata.

    This endpoint is used to populate the topic multi-select on the traffic
    simulator page.
    """
    try:
        sim = TrafficSimulator(repo)
        topics = sim.get_all_topics()
        return TopicsListResponse(success=True, count=len(topics), topics=topics)
    except Exception as exc:
        logger.error("Failed to list topics: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to list topics: {exc}")


@router.post("/apps", response_model=AppsListResponse)
async def list_apps(
    request: TrafficTopicsRequest,
    repo: IGraphRepository = Depends(get_repository),
):
    """
    Return all Application nodes with the topic IDs they publish/subscribe to.

    Used to populate the application selector on the traffic simulator page so
    users can select apps and automatically pull in all associated topics.
    """
    try:
        sim = TrafficSimulator(repo)
        apps = sim.get_all_apps()
        return AppsListResponse(success=True, count=len(apps), apps=apps)
    except Exception as exc:
        logger.error("Failed to list apps: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to list apps: {exc}")


@router.post("/simulate", response_model=TrafficSimulationResponse)
async def simulate_traffic(
    request: TrafficSimulationRequest,
    repo: IGraphRepository = Depends(get_repository),
):
    """
    Compute expected pub-sub traffic metrics for the selected topics.

    Returns:
    - **summary** — aggregate totals (messages, bandwidth, brokers involved).
    - **per_topic** — per-topic breakdown of message rates and bandwidth.
    - **broker_usage** — per-broker inbound/outbound message rates and bandwidth.

    All bandwidth values are in bytes per second (bps) unless suffixed otherwise.
    """
    try:
        sim = TrafficSimulator(repo)
        result = sim.simulate(
            topic_ids=request.topic_ids,
            frequency_hz=request.frequency_hz,
            duration_sec=request.duration_sec,
            message_size_bytes=request.message_size_bytes,
            per_topic_params={
                tid: p.model_dump()
                for tid, p in (request.per_topic_params or {}).items()
            } or None,
        )
        return TrafficSimulationResponse(success=True, **result)
    except Exception as exc:
        logger.error("Traffic simulation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Traffic simulation failed: {exc}")
