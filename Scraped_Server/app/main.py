"""
VisionAssist FastAPI server with WebRTC streaming.
Implements WebRTC endpoints for real-time video streaming with YOLO + MiDaS processing.
"""

import asyncio
import base64
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.webrtc import WebRTCManager
from app.schemas import (
    HealthResponse,
)

# Set up logger
logger = logging.getLogger(__name__)


# ---------- Global State ----------
app_state: Dict[str, Any] = {
    "start_time": time.time(),
    "models_loaded": False,
    "gpu_available": False,
    "device": None,
    "yolo_model": None,
    "midas_model": None,
    "midas_transform": None,
}

# WebRTC Manager
webrtc_manager: Optional[WebRTCManager] = None


# ---------- Lifespan Management ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup, cleanup on shutdown."""
    print("[Init] Starting VisionAssist server...")
    
    # Import vision processing functions
    from vision import init_device, init_yolo, init_midas
    
    # Check GPU availability and initialize device
    app_state["gpu_available"] = torch.cuda.is_available()
    print(f"[Init] GPU available: {app_state['gpu_available']}")
    
    # Initialize device
    app_state["device"] = init_device(force_cpu=settings.force_cpu)
    print(f"[Init] Using device: {app_state['device']}")
    
    # Load YOLO model
    print(f"[Init] Loading YOLO model: {settings.yolo_weights}")
    app_state["yolo_model"] = init_yolo(settings.yolo_weights, app_state["device"])
    print("[Init] YOLO model loaded successfully")
    
    # Load MiDaS depth model
    print(f"[Init] Loading MiDaS model: {settings.midas_model}")
    app_state["midas_model"], app_state["midas_transform"] = init_midas(
        app_state["device"], 
        settings.midas_model
    )
    print("[Init] MiDaS model loaded successfully")
    
    # Initialize WebRTC manager
    global webrtc_manager
    webrtc_manager = WebRTCManager(app_state)
    print("[Init] WebRTC manager initialized")
    
    app_state["models_loaded"] = True
    print("[Init] ✓ All models loaded! Server ready to process frames.")
    
    yield
    
    print("[Shutdown] Cleaning up...")
    # Clean up WebRTC connections
    if webrtc_manager:
        for peer_id in list(webrtc_manager.peers.keys()):
            await webrtc_manager.cleanup_peer(peer_id)
    # Clean up models
    app_state["yolo_model"] = None
    app_state["midas_model"] = None
    app_state["midas_transform"] = None


app = FastAPI(title="VisionAssist Backend", version="0.1.0", lifespan=lifespan)


# ---------- WebRTC Endpoints ----------
@app.post("/webrtc/offer")
async def webrtc_offer(request: dict):
    """
    Handle WebRTC offer and return answer.
    Request format: {"peer_id": "unique_id", "offer": {"sdp": "...", "type": "offer"}}
    """
    try:
        peer_id = request.get("peer_id")
        offer = request.get("offer")
        
        if not peer_id or not offer:
            raise HTTPException(status_code=400, detail="Missing peer_id or offer")
            
        if not webrtc_manager:
            raise HTTPException(status_code=500, detail="WebRTC manager not initialized")
            
        answer = await webrtc_manager.handle_offer(peer_id, offer)
        
        return {
            "peer_id": peer_id,
            "answer": answer,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"WebRTC offer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webrtc/ice")
async def webrtc_ice_candidate(request: dict):
    """
    Handle WebRTC ICE candidate.
    Request format: {"peer_id": "unique_id", "candidate": {...}}
    """
    try:
        peer_id = request.get("peer_id")
        candidate = request.get("candidate")
        
        if not peer_id or not candidate:
            raise HTTPException(status_code=400, detail="Missing peer_id or candidate")
            
        if not webrtc_manager:
            raise HTTPException(status_code=500, detail="WebRTC manager not initialized")
            
        await webrtc_manager.handle_ice_candidate(peer_id, candidate)
        
        return {
            "peer_id": peer_id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"WebRTC ICE candidate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/webrtc/peer/{peer_id}")
async def webrtc_disconnect(peer_id: str):
    """Disconnect a WebRTC peer and clean up resources."""
    try:
        if not webrtc_manager:
            raise HTTPException(status_code=500, detail="WebRTC manager not initialized")
            
        await webrtc_manager.cleanup_peer(peer_id)
        
        return {
            "peer_id": peer_id,
            "status": "disconnected"
        }
        
    except Exception as e:
        logger.error(f"WebRTC disconnect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Health Check Endpoint ----------
@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns server status, GPU availability, and model info.
    """
    uptime = time.time() - app_state["start_time"]
    
    return HealthResponse(
        status="ok" if app_state["models_loaded"] else "initializing",
        gpu=app_state["gpu_available"],
        models={
            "yolo": settings.yolo_weights,
            "midas": settings.midas_model,
            "loaded": app_state["models_loaded"],
        },
        uptime_seconds=round(uptime, 2),
    )


# ---------- Root Endpoint ----------
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "VisionAssist Backend",
        "version": "0.1.0",
        "endpoints": {
            "webrtc_offer": "/webrtc/offer",
            "webrtc_ice": "/webrtc/ice", 
            "webrtc_disconnect": "/webrtc/peer/{peer_id}",
            "health": "/healthz",
        },
        "docs": "/docs",
        "description": "WebRTC-based vision assistance with real-time object detection and depth estimation"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )

