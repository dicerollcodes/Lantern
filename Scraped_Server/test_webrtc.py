#!/usr/bin/env python3
"""
Test script to validate WebRTC setup and basic functionality.
"""

import asyncio
import json
import time
from app.webrtc import WebRTCManager

# Mock app state for testing
mock_app_state = {
    "start_time": time.time(),
    "models_loaded": False,
    "gpu_available": False,
    "device": "cpu",
    "yolo_model": None,
    "midas_model": None,
    "midas_transform": None,
}

async def test_webrtc_manager():
    """Test basic WebRTC manager functionality."""
    print("Testing WebRTC Manager...")
    
    # Create WebRTC manager
    manager = WebRTCManager(mock_app_state) 
    
    # Test peer creation
    peer_id = "test_peer_123"
    pc = await manager.create_peer_connection(peer_id)
    
    print(f"✓ Created peer connection for {peer_id}")
    print(f"✓ Connection state: {pc.connectionState}")
    
    # Test cleanup
    await manager.cleanup_peer(peer_id)
    print(f"✓ Cleaned up peer {peer_id}")
    
    print("WebRTC Manager test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_webrtc_manager())