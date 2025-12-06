#!/usr/bin/env python3
"""
Run the Ambient Loop server.

Usage:
    python run.py              # Start server on default port (2024)
    python run.py --port 8000  # Start server on custom port
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SERVER_DIR = PROJECT_ROOT / "server"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Ambient Loop server")
    parser.add_argument("--port", type=int, default=2024, help="Server port (default: 2024)")
    args = parser.parse_args()

    print("=" * 60)
    print("🌀 Ambient Loop - Customer Support Agent")
    print("=" * 60)
    print()
    print(f"Starting server on port {args.port}...")
    print()
    
    try:
        # Run langgraph dev with integrated FastAPI app
        subprocess.run(
            ["langgraph", "dev", "--port", str(args.port)],
            cwd=SERVER_DIR,
            check=True,
        )
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server exited with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
