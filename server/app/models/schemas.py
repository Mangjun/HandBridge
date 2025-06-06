from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = None

class HandKeypoints(BaseModel):
    landmarks: List[Landmark]
    handedness: str 
    confidence: float

class PoseKeypoints(BaseModel):
    landmarks: List[Landmark]
    confidence: float

class FrameData(BaseModel):
    frame: int
    timestamp: float
    hands: List[HandKeypoints]
    pose: Optional[PoseKeypoints] = None

class VideoInfo(BaseModel):
    filename: str
    original_name: str
    processed_at: datetime
    total_frames: int
    fps: float
    duration: float
    resolution: tuple[int, int]

class ProcessingResult(BaseModel):
    video_info: Dict[str, Any]
    keypoints: List[Any]
    files: Dict[str, str]

class HealthCheck(BaseModel):
    status: str
    version: str
    mediapipe_version: str
    opencv_version: str

class ErrorResponse(BaseModel):
    detail: str 