#!/usr/bin/env python3
"""
Start VisionAssist server with HTTPS support.
"""

import ssl
import uvicorn
from app.main import app

if __name__ == "__main__":
    # Create SSL context for HTTPS
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # For development only - create self-signed certificate
    # In production, use proper certificates
    try:
        ssl_context.load_cert_chain("server.crt", "server.key")
        print("Using existing SSL certificate")
    except FileNotFoundError:
        print("SSL certificate not found. Run setup_ssl.py to create one")
        print("For now, starting without HTTPS...")
        ssl_context = None
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        ssl_keyfile="server.key" if ssl_context else None,
        ssl_certfile="server.crt" if ssl_context else None,
        log_level="info"
    )