"""
Pydantic schemas for VisionAssist WebSocket messages.
Strict validation for init, frame, intent, detections, guidance, error.
"""

from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
import base64


# ---------- Client → Server Messages ----------

class VideoConfig(BaseModel):
    """Video stream configuration."""
    width: int = Field(..., ge=160, le=3840)
    height: int = Field(..., ge=120, le=2160)
    fps: int = Field(..., ge=1, le=60)


class DeviceConfig(BaseModel):
    """Device capabilities."""
    model: str
    has_gyro: bool = False


class IntentData(BaseModel):
    """Intent/target specification."""
    target_class: Optional[str] = None
    target_color: Optional[str] = None


class InitMessage(BaseModel):
    """Initial handshake from client."""
    type: Literal["init"]
    session_id: str
    video: VideoConfig
    device: DeviceConfig
    intent: Optional[IntentData] = None


class PoseData(BaseModel):
    """Device orientation."""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class FrameMessage(BaseModel):
    """Frame data from client. Supports base64 or will be handled as binary."""
    type: Literal["frame"]
    ts: float  # Unix timestamp
    image_b64: Optional[str] = None  # base64-encoded JPEG
    pose: Optional[PoseData] = None
    intent: Optional[IntentData] = None

    @field_validator("image_b64")
    @classmethod
    def validate_base64(cls, v: Optional[str]) -> Optional[str]:
        """Validate that image_b64 is valid base64 if provided."""
        if v is not None:
            try:
                base64.b64decode(v, validate=True)
            except Exception as e:
                raise ValueError(f"Invalid base64 encoding: {e}")
        return v


class ParsedIntent(BaseModel):
    """Parsed intent from NLU."""
    target_class: Optional[str] = None
    target_color: Optional[str] = None


class IntentMessage(BaseModel):
    """Intent update from client (e.g., voice command)."""
    type: Literal["intent"]
    utterance: str
    parsed: Optional[ParsedIntent] = None


# ---------- Server → Client Messages ----------

class ColorInfo(BaseModel):
    """Color detection result."""
    name: str
    score: float = Field(..., ge=0.0, le=1.0)


class DetectedObject(BaseModel):
    """A single detected object."""
    id: int
    cls: str
    conf: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., min_length=4, max_length=4)  # [x, y, w, h]
    color: Optional[ColorInfo] = None
    depth_m: Optional[float] = None
    center: Optional[List[float]] = Field(None, min_length=2, max_length=2)  # [cx, cy]


class DetectionsMessage(BaseModel):
    """Detection results for a frame."""
    type: Literal["detections"]
    ts: float
    fps: float
    objects: List[DetectedObject]


class TargetInfo(BaseModel):
    """Information about the target object."""
    id: int
    cls: str
    color: Optional[str] = None


class DirectionInfo(BaseModel):
    """Directional guidance."""
    horizontal: Optional[str] = None  # "left" | "right" | "center"
    vertical: Optional[str] = None    # "up" | "down" | "center"
    magnitude: Optional[str] = None   # "little" | "some" | "far"


class HapticsInfo(BaseModel):
    """Haptic feedback metadata."""
    pattern: str  # "pulse" | "tap" | "sweep"
    intensity: float = Field(..., ge=0.0, le=1.0)
    direction: List[float] = Field(..., min_length=2, max_length=2)  # [dx, dy]


class GuidanceMessage(BaseModel):
    """Guidance for reaching a target."""
    type: Literal["guidance"]
    ts: float
    target: TargetInfo
    direction: DirectionInfo
    distance_m: Optional[float] = None
    speak: str
    haptics: HapticsInfo


class ErrorMessage(BaseModel):
    """Error response."""
    type: Literal["error"]
    message: str
    code: Optional[str] = None


class IntentAckMessage(BaseModel):
    """Acknowledgment of intent update."""
    type: Literal["intent_ack"]
    intent: ParsedIntent


# ---------- Health Check Response ----------

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gpu: bool
    models: Dict[str, Any]
    uptime_seconds: float

