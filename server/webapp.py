"""
FastAPI app integrated with LangGraph server.
Handles Gmail Pub/Sub push notifications and custom routes.
"""

import os
import sys
import asyncio
import base64
import json
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT_DIR, ".env"))


# =============================================================================
# LIFESPAN - Setup Gmail Watch and optional ngrok
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup Gmail Watch and ngrok on startup."""
    from integration.mail_api import get_gmail_service
    
    service = get_gmail_service()
    
    # Print authenticated account
    try:
        profile = service.users().getProfile(userId='me').execute()
        gmail_email = profile.get('emailAddress', 'Unknown')
        print(f"📧 Gmail account: {gmail_email}")
    except Exception as e:
        print(f"⚠️  Could not get Gmail profile: {e}")
    
    # Setup Gmail Watch
    gmail_topic = os.getenv("GMAIL_PUBSUB_TOPIC")
    if gmail_topic:
        try:
            watch_response = service.users().watch(
                userId='me', 
                body={'topicName': gmail_topic, 'labelIds': ['INBOX']}
            ).execute()
            history_id = watch_response.get('historyId')
            expiration_ms = watch_response.get('expiration')
            
            if expiration_ms:
                expiration_dt = datetime.fromtimestamp(int(expiration_ms) / 1000)
                days = (expiration_dt - datetime.now()).days
                print(f"✅ Gmail Watch active! Expires in {days} days")
            else:
                print(f"✅ Gmail Watch active! History ID: {history_id}")
        except Exception as e:
            print(f"⚠️  Could not setup Gmail Watch: {e}")
    else:
        print("⚠️  GMAIL_PUBSUB_TOPIC not set")
    
    # Start ngrok if configured
    ngrok_domain = os.getenv("NGROK_DOMAIN")
    ngrok_process = None
    port = int(os.getenv("PORT", "2024"))
    
    if ngrok_domain and os.getenv("NGROK_ENABLED", "true").lower() != "false":
        try:
            print(f"🌐 Starting ngrok: {ngrok_domain}")
            ngrok_process = subprocess.Popen(
                ["ngrok", "http", str(port), "--domain", ngrok_domain],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await asyncio.sleep(3)
            print(f"✅ ngrok active: https://{ngrok_domain}")
            print(f"📧 Pub/Sub endpoint: https://{ngrok_domain}/gmail/push")
        except FileNotFoundError:
            print("⚠️  ngrok not found")
        except Exception as e:
            print(f"⚠️  ngrok error: {e}")
    
    print("🚀 Server ready!")
    
    yield
    
    # Cleanup
    if ngrok_process:
        ngrok_process.terminate()
        try:
            ngrok_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ngrok_process.kill()


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Ambient Loop - Customer Support Agent",
    description="Gmail → LangGraph → Auto-reply",
    version="3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# GMAIL PUB/SUB ENDPOINT
# =============================================================================

@app.post("/gmail/push")
async def gmail_push(request: Request):
    """
    Pub/Sub push endpoint for Gmail notifications.
    This is the main entry point for processing incoming emails.
    """
    from integration.mail_api import process_new_emails, get_gmail_service
    
    envelope = await request.json()
    pubsub_msg = envelope.get("message")
    
    if not pubsub_msg:
        return {"status": "no-message"}
    
    data_b64 = pubsub_msg.get("data")
    if not data_b64:
        return {"status": "no-data"}
    
    decoded = base64.b64decode(data_b64).decode("utf-8")
    data = json.loads(decoded)
    
    history_id = data.get("historyId")
    if not history_id:
        return {"status": "no-history-id"}
    
    # Process emails
    service = get_gmail_service()
    asyncio.create_task(process_new_emails(service, str(history_id)))
    
    return {"status": "processing"}


# =============================================================================
# TICKET APIs (for testing/debugging)
# =============================================================================

class TicketRequest(BaseModel):
    ticket_id: str
    ticket_description: str
    customer_id: str
    received_date: datetime


class TicketResponse(BaseModel):
    ticket_id: str
    status: str
    message: str
    problems: Optional[List[str]] = None
    policy_name: Optional[str] = None
    action_taken: Optional[str] = None


def get_db():
    from db import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/tickets", response_model=TicketResponse)
async def create_ticket(
    ticket: TicketRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Manual ticket creation for testing."""
    from db import Ticket
    from db.session import save_ticket_state
    from agent.graph import graph_app
    from agent.state import SupportAgentState
    from langchain_core.messages import HumanMessage
    
    existing = db.query(Ticket).filter(Ticket.ticket_id == ticket.ticket_id).first()
    if existing:
        return {"ticket_id": ticket.ticket_id, "status": "error", "message": "Already exists"}
    
    new_ticket = Ticket(
        ticket_id=ticket.ticket_id,
        customer_id=ticket.customer_id,
        description=ticket.ticket_description,
        received_date=ticket.received_date,
        status="processing",
    )
    db.add(new_ticket)
    db.commit()
    
    def process():
        initial_state = SupportAgentState(messages=[HumanMessage(content=ticket.ticket_description)])
        final_state = graph_app.invoke(initial_state)
        from db import SessionLocal
        session = SessionLocal()
        save_ticket_state(
            {"ticket_id": ticket.ticket_id, "customer_id": ticket.customer_id, 
             "description": ticket.ticket_description, "received_date": ticket.received_date},
            final_state, session
        )
        session.close()
    
    background_tasks.add_task(process)
    
    return {"ticket_id": ticket.ticket_id, "status": "processing", "message": "Started"}


@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str, db: Session = Depends(get_db)):
    from db import Ticket, TicketState
    
    ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Not found")
    
    ts = db.query(TicketState).filter(TicketState.ticket_id == ticket.id).first()
    
    return {
        "ticket_id": ticket.ticket_id,
        "status": ticket.status,
        "customer_id": ticket.customer_id,
        "description": ticket.description,
        "problems": ts.problems if ts else None,
        "policy_name": ts.policy_name if ts else None,
        "action_taken": ts.action_taken if ts else None,
    }


@app.get("/tickets")
async def list_tickets(db: Session = Depends(get_db)):
    from db import Ticket, TicketState
    
    tickets = db.query(Ticket).all()
    result = []
    
    for t in tickets:
        ts = db.query(TicketState).filter(TicketState.ticket_id == t.id).first()
        result.append({
            "ticket_id": t.ticket_id,
            "status": t.status,
            "customer_id": t.customer_id,
            "problems": ts.problems if ts else None,
            "action_taken": ts.action_taken if ts else None,
        })
    
    return result


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ambient-loop"}
