#!/usr/bin/env python3
"""
Start script for FastAPI server
Run this file to start the pneumonia detection API server
"""

import uvicorn
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Start the FastAPI server"""
    print("🚀 Starting Pneumonia Detection API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔄 Auto-reload enabled for development")
    print("-" * 50)
    
    try:
        # Start the server with uvicorn using import string for reload
        uvicorn.run(
            "api:app",  # Import string format required for reload
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
