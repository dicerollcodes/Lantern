#!/usr/bin/env python3
"""
Setup script for VisionAssist WebRTC implementation.
Installs required dependencies and validates the setup.
"""

import subprocess
import sys
import importlib

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is not installed")
        return False

def main():
    """Main setup function."""
    print("VisionAssist WebRTC Setup")
    print("=" * 40)
    
    # Required packages for WebRTC
    webrtc_packages = [
        ("aiortc", "aiortc"),
        ("aiofiles", "aiofiles"),
        ("av", "av"),  # PyAV for video processing
    ]
    
    print("\nChecking WebRTC dependencies...")
    missing_packages = []
    
    for package, import_name in webrtc_packages:
        if not check_package(package, import_name):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            if not install_package(package):
                print(f"Failed to install {package}. Please install manually.")
                return False
    
    print("\n" + "=" * 40)
    print("WebRTC setup complete!")
    
    # Additional setup info
    print("\nWebRTC Implementation Features:")
    print("• Lower latency than WebSockets")
    print("• Built-in video compression")
    print("• Peer-to-peer connection (after signaling)")
    print("• Data channel for detection results")
    print("• Automatic frame rate adaptation")
    
    print("\nTo test the WebRTC implementation:")
    print("1. Start the server: python -m app.main")
    print("2. Open webrtc_client.html in a web browser")
    print("3. Click 'Connect' to establish WebRTC connection")
    
    print("\nAPI Endpoints:")
    print("• POST /webrtc/offer - Handle WebRTC offer")
    print("• POST /webrtc/ice - Handle ICE candidates")
    print("• DELETE /webrtc/peer/{peer_id} - Disconnect peer")
    print("• WebSocket /ws/stream - Legacy WebSocket endpoint (still available)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)