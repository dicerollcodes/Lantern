#!/usr/bin/env python3
"""
Simple test client for VisionAssist WebSocket API.
Demonstrates how to connect and send frames.
"""

import asyncio
import base64
import json
import time
from pathlib import Path

import cv2
import websockets


async def test_websocket_client():
    """Test WebSocket connection with sample frames."""
    
    uri = "ws://localhost:8080/ws/stream"
    
    print(f"[Client] Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("[Client] Connected!")
        
        # 1. Send INIT message
        init_msg = {
            "type": "init",
            "session_id": "test-session-001",
            "video": {
                "width": 640,
                "height": 480,
                "fps": 15
            },
            "device": {
                "model": "Test Device",
                "has_gyro": False
            },
            "intent": {
                "target_class": None,
                "target_color": None
            }
        }
        
        await websocket.send(json.dumps(init_msg))
        print("[Client] Sent INIT message")
        
        # 2. Test with a dummy frame (create a simple test image)
        # Create a 640x480 test image
        test_img = cv2.imread("test_frame.jpg") if Path("test_frame.jpg").exists() else None
        
        if test_img is None:
            # Create a simple colored frame
            test_img = cv2.rectangle(
                cv2.UMat(480, 640, cv2.CV_8UC3, (100, 150, 200)),
                (100, 100), (540, 380), (255, 0, 0), 3
            )
            test_img = test_img.get()  # Convert back to numpy array
        
        # Encode as JPEG
        _, jpeg_buffer = cv2.imencode('.jpg', test_img)
        jpeg_bytes = jpeg_buffer.tobytes()
        
        # Test 1: Send as base64
        print("\n[Client] Test 1: Sending frame as base64...")
        frame_msg_b64 = {
            "type": "frame",
            "ts": time.time(),
            "image_b64": base64.b64encode(jpeg_bytes).decode('utf-8'),
            "pose": {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0
            },
            "intent": {
                "target_class": "person",
                "target_color": "red"
            }
        }
        
        await websocket.send(json.dumps(frame_msg_b64))
        print("[Client] Sent base64 frame")
        
        # Receive response
        response = await websocket.recv()
        result = json.loads(response)
        print(f"[Client] Received: {result.get('type', 'unknown')}")
        print(f"[Client] Response: {json.dumps(result, indent=2)}")
        
        # Test 2: Send as binary
        print("\n[Client] Test 2: Sending frame as binary...")
        await websocket.send(jpeg_bytes)
        print("[Client] Sent binary frame")
        
        # Receive response
        response = await websocket.recv()
        result = json.loads(response)
        print(f"[Client] Received: {result.get('type', 'unknown')}")
        print(f"[Client] Response: {json.dumps(result, indent=2)}")
        
        # Test 3: Update intent
        print("\n[Client] Test 3: Updating intent...")
        intent_msg = {
            "type": "intent",
            "utterance": "find the red shirt",
            "parsed": {
                "target_class": "shirt",
                "target_color": "red"
            }
        }
        
        await websocket.send(json.dumps(intent_msg))
        print("[Client] Sent intent update")
        
        # Receive acknowledgment
        response = await websocket.recv()
        result = json.loads(response)
        print(f"[Client] Received: {result.get('type', 'unknown')}")
        print(f"[Client] Response: {json.dumps(result, indent=2)}")
        
        # Test 4: Send invalid message
        print("\n[Client] Test 4: Sending invalid frame (unsupported format)...")
        invalid_msg = {
            "type": "frame",
            "ts": time.time(),
            "image_b64": "not-valid-base64!!!"
        }
        
        await websocket.send(json.dumps(invalid_msg))
        print("[Client] Sent invalid frame")
        
        # Should receive error
        response = await websocket.recv()
        result = json.loads(response)
        print(f"[Client] Received: {result.get('type', 'unknown')}")
        print(f"[Client] Error: {result.get('message', 'N/A')}")
        
        print("\n[Client] All tests completed!")


async def test_health_check():
    """Test the health check endpoint."""
    import aiohttp
    
    url = "http://localhost:8080/healthz"
    
    print(f"\n[Client] Testing health check at {url}...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                print(f"[Client] Health check OK!")
                print(f"[Client] Response: {json.dumps(data, indent=2)}")
            else:
                print(f"[Client] Health check failed: {response.status}")


async def main():
    """Run all tests."""
    try:
        # Test health check first
        await test_health_check()
        
        # Test WebSocket
        await test_websocket_client()
        
    except ConnectionRefusedError:
        print("[Client] ERROR: Could not connect to server.")
        print("[Client] Make sure the server is running: python -m app.main")
    except Exception as e:
        print(f"[Client] ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())

