#!/bin/bash
# Start VisionAssist Backend Server

echo "Starting VisionAssist Backend Server..."
echo ""
echo "Endpoints:"
echo "  WebRTC Offer: http://localhost:8080/webrtc/offer"
echo "  WebRTC ICE:   http://localhost:8080/webrtc/ice"
echo "  Health:       http://localhost:8080/healthz"
echo "  Docs:         http://localhost:8080/docs"
echo "  Test Client:  webrtc_client.html"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start server
python -m app.main

