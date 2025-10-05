"""
WebRTC implementation for VisionAssist real-time video streaming.
Provides lower latency and better compression than WebSockets.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
import uuid

import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import VideoStreamTrack
from av import VideoFrame

from app.schemas import (
    InitMessage,
    FrameMessage,
    IntentMessage,
    DetectionsMessage,
    GuidanceMessage,
    ErrorMessage,
    IntentAckMessage,
    DetectedObject,
)

logger = logging.getLogger(__name__)


class VisionTrack(VideoStreamTrack):
    """
    Custom video track that processes frames through YOLO + MiDaS pipeline.
    """
    
    def __init__(self, app_state: Dict[str, Any]):
        super().__init__()
        self.app_state = app_state
        self.session_data = {
            'id': str(uuid.uuid4()),
            'intent': 'navigate',
            'initialized': False,
            'last_frame_time': 0,
            'frame_count': 0,
        }
        self.data_channel: Optional[RTCDataChannel] = None
        
    def set_data_channel(self, channel: RTCDataChannel):
        """Set the data channel for sending detection results."""
        self.data_channel = channel
        
    async def recv(self):
        """
        Process and return video frames with vision processing.
        """
        try:
            pts, time_base = await self.next_timestamp()
            
            # Create a dummy frame for now - in real implementation, 
            # you'd get this from camera or input stream
            frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Apply vision processing if models are loaded
            if self.app_state["models_loaded"]:
                processed_frame = await self.process_frame(frame_bgr)
            else:
                processed_frame = frame_bgr
                
            # Convert BGR to RGB for WebRTC
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Create video frame
            frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            frame.pts = pts
            frame.time_base = time_base
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in VisionTrack.recv: {e}")
            # Return a black frame on error
            frame_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            frame.pts = pts
            frame.time_base = time_base
            return frame
            
    async def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Process frame through YOLO + MiDaS pipeline.
        """
        try:
            self.session_data['frame_count'] += 1
            current_time = time.time()
            
            # YOLO detection
            boxes = await self.run_yolo_async(frame_bgr)
            
            # MiDaS depth estimation
            depth_map = None
            if self.session_data['frame_count'] % 3 == 1:  # Run depth every 3 frames
                depth_map = await self.run_midas_async(frame_bgr)
            
            # Draw visualizations
            processed_frame = self.draw_detections(frame_bgr.copy(), boxes, depth_map)
            
            # Send detection data via data channel
            if self.data_channel and boxes:
                await self.send_detections(boxes, depth_map)
                
            self.session_data['last_frame_time'] = current_time
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame_bgr
            
    async def run_yolo_async(self, frame_bgr: np.ndarray):
        """Run YOLO detection asynchronously."""
        try:
            yolo_model = self.app_state["yolo_model"]
            device = self.app_state["device"]
            
            if not yolo_model or not device:
                return []
                
            # Run YOLO in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: yolo_model.predict(
                    source=frame_bgr, 
                    imgsz=640, 
                    conf=0.25, 
                    verbose=False,
                    device=str(device)
                )[0]
            )
            
            # Parse results
            boxes = []
            if results.boxes is not None:
                xyxy = results.boxes.xyxy.detach().cpu().numpy()
                confs = results.boxes.conf.detach().cpu().numpy()
                clss = results.boxes.cls.detach().cpu().numpy().astype(int)
                names = results.names if hasattr(results, "names") else yolo_model.model.names
                
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    boxes.append({
                        'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                        'conf': float(c), 'cls_id': int(k),
                        'cls_name': str(names.get(int(k), str(int(k))))
                    })
                    
            return boxes
            
        except Exception as e:
            logger.error(f"YOLO error: {e}")
            return []
            
    async def run_midas_async(self, frame_bgr: np.ndarray):
        """Run MiDaS depth estimation asynchronously."""
        try:
            midas_model = self.app_state["midas_model"]
            midas_transform = self.app_state["midas_transform"]
            device = self.app_state["device"]
            
            if not midas_model or not midas_transform or not device:
                return None
                
            # Run MiDaS in thread pool
            loop = asyncio.get_event_loop()
            depth_map = await loop.run_in_executor(
                None,
                self._run_midas_sync,
                frame_bgr, 
                midas_model, 
                midas_transform, 
                device
            )
            
            return depth_map
            
        except Exception as e:
            logger.error(f"MiDaS error: {e}")
            return None
            
    def _run_midas_sync(self, frame_bgr, midas_model, midas_transform, device):
        """Synchronous MiDaS processing."""
        import torch
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = midas_transform(img_rgb).to(device)
        
        with torch.inference_mode():
            pred = midas_model(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
            
        depth = pred[0].detach().cpu().numpy().astype(np.float32)
        return depth
        
    def draw_detections(self, frame: np.ndarray, boxes: list, depth_map: Optional[np.ndarray]) -> np.ndarray:
        """Draw bounding boxes and depth information on frame."""
        H, W = frame.shape[:2]
        
        for box in boxes:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            cls_name, conf = box['cls_name'], box['conf']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw midpoint
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.circle(frame, (mid_x, mid_y), 4, (0, 255, 255), -1)
            cv2.circle(frame, (mid_x, mid_y), 6, (0, 0, 0), 2)
            
            # Calculate depth if available
            depth_text = ""
            if depth_map is not None:
                try:
                    d_roi = depth_map[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                    if d_roi.size > 0:
                        med_depth = float(np.median(d_roi))
                        # Simple depth mapping (you can improve this)
                        gmin = float(np.percentile(depth_map.reshape(-1), 1))
                        gmax = float(np.percentile(depth_map.reshape(-1), 99))
                        if gmax - gmin > 1e-6:
                            depth_norm = (med_depth - gmin) / (gmax - gmin)
                            # Map to approximate distance (0.2m to 5.0m range)
                            meters = 5.0 - depth_norm * (5.0 - 0.2)
                            cm = int(round(meters * 100.0))
                            depth_text = f" • {cm}cm"
                except Exception:
                    pass
            
            # Draw label
            label = f"{cls_name} {conf:.2f}{depth_text}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - h - 10)), (x1 + w, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        return frame
        
    async def send_detections(self, boxes: list, depth_map: Optional[np.ndarray]):
        """Send detection results via data channel."""
        try:
            if not self.data_channel or self.data_channel.readyState != "open":
                return
                
            # Convert boxes to DetectedObject format
            detected_objects = []
            for box in boxes:
                obj = DetectedObject(
                    class_name=box['cls_name'],
                    confidence=box['conf'],
                    bounding_box={
                        'x1': box['x1'], 'y1': box['y1'],
                        'x2': box['x2'], 'y2': box['y2']
                    },
                    distance_cm=None  # Will be calculated if depth available
                )
                detected_objects.append(obj)
                
            # Create detection message
            detection_msg = DetectionsMessage(
                type="detections",
                timestamp=time.time(),
                objects=detected_objects,
                frame_count=self.session_data['frame_count']
            )
            
            # Send via data channel
            self.data_channel.send(detection_msg.model_dump_json())
            
        except Exception as e:
            logger.error(f"Error sending detections: {e}")


class WebRTCManager:
    """Manages WebRTC peer connections and tracks."""
    
    def __init__(self, app_state: Dict[str, Any]):
        self.app_state = app_state
        self.peers: Dict[str, RTCPeerConnection] = {}
        self.tracks: Dict[str, VisionTrack] = {}
        
    async def create_peer_connection(self, peer_id: str) -> RTCPeerConnection:
        """Create a new RTCPeerConnection."""
        pc = RTCPeerConnection()
        self.peers[peer_id] = pc
        
        # Create vision track
        track = VisionTrack(self.app_state)
        self.tracks[peer_id] = track
        
        # Add video track
        pc.addTrack(track)
        
        # Create data channel for sending detection results
        data_channel = pc.createDataChannel("detections")
        track.set_data_channel(data_channel)
        
        @data_channel.on("open")
        def on_data_channel_open():
            logger.info(f"Data channel opened for peer {peer_id}")
            
        @data_channel.on("message")
        def on_data_channel_message(message):
            logger.info(f"Received data channel message: {message}")
            # Handle incoming messages (intent changes, etc.)
            try:
                data = json.loads(message)
                if data.get("type") == "intent":
                    intent_msg = IntentMessage(**data)
                    track.session_data['intent'] = intent_msg.intent
                    logger.info(f"Updated intent for peer {peer_id}: {intent_msg.intent}")
            except Exception as e:
                logger.error(f"Error handling data channel message: {e}")
        
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"Connection state for {peer_id}: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                await self.cleanup_peer(peer_id)
                
        return pc
        
    async def cleanup_peer(self, peer_id: str):
        """Clean up resources for a disconnected peer."""
        if peer_id in self.peers:
            await self.peers[peer_id].close()
            del self.peers[peer_id]
            
        if peer_id in self.tracks:
            del self.tracks[peer_id]
            
        logger.info(f"Cleaned up peer {peer_id}")
        
    async def handle_offer(self, peer_id: str, offer: dict) -> dict:
        """Handle WebRTC offer and return answer."""
        try:
            pc = await self.create_peer_connection(peer_id)
            
            # Set remote description
            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=offer["sdp"], 
                type=offer["type"]
            ))
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }
            
        except Exception as e:
            logger.error(f"Error handling offer for peer {peer_id}: {e}")
            raise
            
    async def handle_ice_candidate(self, peer_id: str, candidate: dict):
        """Handle ICE candidate."""
        try:
            if peer_id in self.peers:
                pc = self.peers[peer_id]
                await pc.addIceCandidate(candidate)
        except Exception as e:
            logger.error(f"Error handling ICE candidate for peer {peer_id}: {e}")