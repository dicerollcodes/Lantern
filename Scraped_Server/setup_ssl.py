#!/usr/bin/env python3
"""
Generate self-signed SSL certificate for development.
"""

import subprocess
import sys
import os

def create_ssl_certificate():
    """Create self-signed SSL certificate for local development."""
    
    # Check if OpenSSL is available
    try:
        subprocess.run(["openssl", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("OpenSSL not found. Please install OpenSSL or use Option 1/2 above.")
        return False
    
    # Create self-signed certificate
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", "server.key", "-out", "server.crt",
        "-days", "365", "-nodes", "-subj",
        "/C=US/ST=Dev/L=Dev/O=VisionAssist/CN=10.250.81.2"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ SSL certificate created successfully!")
        print("Files created: server.key, server.crt")
        print("Now you can run: python start_https_server.py")
        print("Android should connect to: https://10.250.81.2:8080")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create SSL certificate: {e}")
        return False

if __name__ == "__main__":
    create_ssl_certificate()