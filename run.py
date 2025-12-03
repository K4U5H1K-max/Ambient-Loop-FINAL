#!/usr/bin/env python3
"""
Run script to start both FastAPI server and LangGraph dev server.

Usage:
    python run.py          # Start both servers
    python run.py --api    # Start only FastAPI server
    python run.py --graph  # Start only LangGraph dev server
"""

import subprocess
import sys
import os
import signal
import time
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
CONFIG_DIR = PROJECT_ROOT / "config"

# Server configurations
FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = 8000
LANGGRAPH_PORT = 2024


def start_langgraph_server():
    """Start LangGraph dev server in config directory."""
    print(f"🚀 Starting LangGraph dev server on port {LANGGRAPH_PORT}...")
    return subprocess.Popen(
        ["langgraph", "dev", "--port", str(LANGGRAPH_PORT)],
        cwd=CONFIG_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def start_fastapi_server():
    """Start FastAPI server."""
    print(f"🚀 Starting FastAPI server on port {FASTAPI_PORT}...")
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "api.server:app",
            "--host", FASTAPI_HOST,
            "--port", str(FASTAPI_PORT),
            "--reload",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def stream_output(process, prefix):
    """Stream process output with prefix."""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"{prefix} {line.rstrip()}")
    except Exception:
        pass


def main():
    import argparse
    import threading

    parser = argparse.ArgumentParser(description="Run Ambient Loop servers")
    parser.add_argument("--api", action="store_true", help="Start only FastAPI server")
    parser.add_argument("--graph", action="store_true", help="Start only LangGraph dev server")
    args = parser.parse_args()

    # If no flags, start both
    start_api = args.api or (not args.api and not args.graph)
    start_graph = args.graph or (not args.api and not args.graph)

    processes = []
    threads = []

    print("=" * 60)
    print("🌀 Ambient Loop - Customer Support Agent")
    print("=" * 60)

    try:
        # Start LangGraph server first (FastAPI depends on it)
        if start_graph:
            langgraph_proc = start_langgraph_server()
            processes.append(("LangGraph", langgraph_proc))
            
            # Stream output in background thread
            t = threading.Thread(
                target=stream_output,
                args=(langgraph_proc, "[LangGraph]"),
                daemon=True
            )
            t.start()
            threads.append(t)
            
            # Wait for LangGraph to initialize
            print("⏳ Waiting for LangGraph server to initialize...")
            time.sleep(5)

        if start_api:
            fastapi_proc = start_fastapi_server()
            processes.append(("FastAPI", fastapi_proc))
            
            # Stream output in background thread
            t = threading.Thread(
                target=stream_output,
                args=(fastapi_proc, "[FastAPI]"),
                daemon=True
            )
            t.start()
            threads.append(t)

        print()
        print("=" * 60)
        print("✅ Servers started successfully!")
        print("=" * 60)
        if start_graph:
            print(f"📊 LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:{LANGGRAPH_PORT}")
            print(f"📚 LangGraph API:    http://127.0.0.1:{LANGGRAPH_PORT}/docs")
        if start_api:
            print(f"🌐 FastAPI Server:   http://127.0.0.1:{FASTAPI_PORT}")
            print(f"📚 FastAPI Docs:     http://127.0.0.1:{FASTAPI_PORT}/docs")
        print()
        print("Press Ctrl+C to stop all servers")
        print("=" * 60)

        # Wait for processes
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"⚠️  {name} server exited with code {proc.returncode}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")

    finally:
        # Cleanup: terminate all processes
        for name, proc in processes:
            if proc.poll() is None:
                print(f"   Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()


