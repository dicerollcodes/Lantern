"""
Configuration for VisionAssist server.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Server configuration."""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    
    # Performance
    max_frame_backlog: int = 3  # Drop frames if queue grows beyond this
    target_latency_ms: float = 300.0  # Target end-to-end latency
    
    # Models (paths or identifiers)
    yolo_weights: str = "yolov8m.pt"  # Using yolov8m for better accuracy
    midas_model: str = "midas_v21_small"
    
    # Inference
    yolo_conf: float = 0.25
    yolo_imgsz: int = 640
    force_cpu: bool = False
    
    # Frame handling
    max_frame_size_mb: float = 10.0  # Maximum frame size in MB
    supported_formats: list[str] = ["image/jpeg", "image/jpg"]
    
    class Config:
        env_prefix = "VISION_"


settings = Settings()

