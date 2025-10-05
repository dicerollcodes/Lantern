@echo off
REM Start VisionAssist Backend Server (Windows)

echo Starting VisionAssist Backend Server...
echo.
echo Endpoints:
echo   WebRTC Offer: http://localhost:8080/webrtc/offer
echo   WebRTC ICE:   http://localhost:8080/webrtc/ice
echo   Health:       http://localhost:8080/healthz
echo   Docs:         http://localhost:8080/docs
echo   Test Client:  webrtc_client.html
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo No virtual environment found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Start server
python -m app.main

