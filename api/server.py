"""
FastAPI server for Customer Support Agent + Gmail Pub/Sub Push Integration
---------------------------------------------------------------------------
Flow:
Gmail WATCH ---> Pub/Sub Topic ---> Subscription (PUSH) ---> /gmail/push
/gmail/push ---> process_gmail_history() ---> internal queue
email_worker processes emails, graph calls /send-reply when done
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any, List
import uvicorn
import asyncio
import os
import subprocess
from dotenv import load_dotenv

from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

# Graph components (for manual ticket creation)
from agent.graph import graph_app
from agent.state import SupportAgentState
from langchain_core.messages import HumanMessage

# Database components
from data.ticket_db import get_db, save_ticket_state, Ticket, TicketState, SessionLocal


# ---------------------------------------------------------------------------
# Lifespan — start internal workers for Pub/Sub push events
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup:
      - Launch Gmail push worker (email_worker)
      - Start ngrok tunnel with static domain (if enabled)
      - Set up Gmail Watch for Pub/Sub notifications
    """
    from integration.mail_api import email_worker, get_gmail_service

    service = get_gmail_service()
    
    # Get and print authenticated Gmail account
    try:
        profile = service.users().getProfile(userId='me').execute()
        gmail_email = profile.get('emailAddress', 'Unknown')
        print(f"📧 Authenticated Gmail account: {gmail_email}")
    except Exception as e:
        print(f"⚠️  Could not get Gmail profile: {e}")
        gmail_email = "Unknown"
    
    loop = asyncio.get_event_loop()

    loop.create_task(email_worker(service))
    # Note: human_resolution_monitor removed - graph now calls /send-reply directly

    # Set up Gmail Watch for Pub/Sub
    gmail_topic = os.getenv("GMAIL_PUBSUB_TOPIC")
    if gmail_topic:
        try:
            print(f"📧 Setting up Gmail Watch for topic: {gmail_topic}")
            print(f"   Using Gmail account: {gmail_email}")
            request = {
                'topicName': gmail_topic,
                'labelIds': ['INBOX'],
            }
            watch_response = service.users().watch(userId='me', body=request).execute()
            history_id = watch_response.get('historyId')
            expiration_ms = watch_response.get('expiration')
            
            if expiration_ms:
                expiration_dt = datetime.fromtimestamp(int(expiration_ms) / 1000)
                days_until_expiry = (expiration_dt - datetime.now()).days
                print(f"✅ Gmail Watch active! History ID: {history_id}")
                print(f"   Expires in: {days_until_expiry} days ({expiration_dt.strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                print(f"✅ Gmail Watch active! History ID: {history_id}")
        except Exception as e:
            print(f"⚠️  Warning: Could not set up Gmail Watch: {e}")
            print("   Server will run but Gmail notifications may not work.")
            print("   Make sure GMAIL_PUBSUB_TOPIC is set correctly in .env")
    else:
        print("⚠️  GMAIL_PUBSUB_TOPIC not set in .env - Gmail Watch not configured")
        print("   Add GMAIL_PUBSUB_TOPIC=projects/YOUR_PROJECT/topics/YOUR_TOPIC to .env")

    # Start ngrok only if domain is configured in .env
    ngrok_domain = os.getenv("NGROK_DOMAIN")
    ngrok_process = None
    
    # Only start ngrok if domain is provided and not explicitly disabled
    if ngrok_domain and os.getenv("NGROK_ENABLED", "true").lower() != "false":
        try:
            print(f"🌐 Starting ngrok tunnel with domain: {ngrok_domain}")
            ngrok_process = subprocess.Popen(
                ["ngrok", "http", "8000", "--domain", ngrok_domain],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for ngrok to initialize
            await asyncio.sleep(3)
            
            ngrok_url = f"https://{ngrok_domain}"
            print(f"✅ ngrok tunnel active!")
            print(f"📍 Local: http://localhost:8000")
            print(f"🌐 Public: {ngrok_url}")
            print(f"📧 Pub/Sub endpoint: {ngrok_url}/gmail/push")
            print(f"📋 Update your Pub/Sub subscription endpoint to: {ngrok_url}/gmail/push")
        except FileNotFoundError:
            print("⚠️  ngrok not found. Install it: brew install ngrok")
            print("   Running locally without ngrok tunnel")
        except Exception as e:
            print(f"⚠️  Error starting ngrok: {e}")
            print(f"   Running locally without ngrok tunnel")
    else:
        if not ngrok_domain:
            print("📍 Running locally: http://localhost:8000")
            print("   (No NGROK_DOMAIN set in .env - ngrok disabled)")
        else:
            print("📍 Running locally: http://localhost:8000")
            print("   (NGROK_ENABLED=false - ngrok disabled)")

    print("🚀 Gmail Pub/Sub Push workers started.")

    yield

    # Cleanup: Stop ngrok when server shuts down
    if ngrok_process:
        print("🛑 Stopping ngrok tunnel...")
        ngrok_process.terminate()
        try:
            ngrok_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ngrok_process.kill()
    
    print("🛑 Server shutting down...")


# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Support Agent API",
    description="Gmail → Pub/Sub Push → LangGraph → DB",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Gmail Pub/Sub PUSH Endpoint
# ---------------------------------------------------------------------------
@app.post("/gmail/push")
async def gmail_push(request: Request):
    """
    Pub/Sub push endpoint for Gmail notifications.

    Envelope:
    {
        "message": {
            "data": "<base64-encoded JSON: { 'emailAddress': ..., 'historyId': ... }>"
        }
    }
    """

    envelope = await request.json()
    pubsub_msg = envelope.get("message")

    if not pubsub_msg:
        return {"status": "no-message"}

    data_b64 = pubsub_msg.get("data")
    if not data_b64:
        return {"status": "no-data"}

    import base64, json as pyjson

    decoded = base64.b64decode(data_b64).decode("utf-8")
    data = pyjson.loads(decoded)

    history_id = data.get("historyId")
    if not history_id:
        return {"status": "no-history-id"}

    # Trigger Gmail history processing
    from integration.mail_api import process_gmail_history, get_gmail_service

    service = get_gmail_service()
    asyncio.create_task(process_gmail_history(service, str(history_id)))

    return {"status": "queued"}


# ---------------------------------------------------------------------------
# Graph Callback Endpoint - Called when graph finishes processing
# ---------------------------------------------------------------------------
class SendReplyRequest(BaseModel):
    gmail_msg_id: str
    gmail_thread_id: str
    sender_email: str
    email_subject: str
    email_reply: str
    action_taken: Optional[str] = None
    tier_level: Optional[str] = None


@app.post("/send-reply")
async def send_reply_endpoint(data: SendReplyRequest):
    """
    Callback endpoint called by the graph when email reply is ready.
    Sends Gmail reply and marks message as read.
    """
    from integration.mail_api import (
        get_gmail_service, 
        send_reply, 
        mark_message_as_read,
        set_message_status,
        get_full_message,
        parse_message_metadata
    )
    
    try:
        service = get_gmail_service()
        
        # Get original message for reply headers
        gmail_msg = get_full_message(service, data.gmail_msg_id)
        email_meta = parse_message_metadata(gmail_msg)
        
        # Send the reply
        success = send_reply(
            service,
            to_addr=data.sender_email,
            subject=data.email_subject,
            reply_text=data.email_reply,
            gmail_thread_id=data.gmail_thread_id,
            in_reply_to=email_meta.get("message_id_header"),
        )
        
        if success:
            # Mark original message as read
            mark_message_as_read(service, data.gmail_msg_id)
            
            # Update status in state.json
            set_message_status(
                data.gmail_msg_id,
                gmail_thread_id=data.gmail_thread_id,
                langgraph_thread_id=data.gmail_thread_id,  # Use gmail_thread_id as reference
                status="completed",
            )
            
            print(f"✅ Reply sent for msg_id={data.gmail_msg_id}, tier={data.tier_level}")
            return {"status": "sent", "msg_id": data.gmail_msg_id}
        else:
            print(f"❌ Failed to send reply for msg_id={data.gmail_msg_id}")
            return {"status": "failed", "msg_id": data.gmail_msg_id}
            
    except Exception as e:
        print(f"❌ Error in send_reply_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Manual Ticket REST APIs (optional debugging/testing)
# ---------------------------------------------------------------------------
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
    customer_id: Optional[str] = None
    description: Optional[str] = None
    received_date: Optional[datetime] = None
    messages: Optional[List[Dict[str, Any]]] = None


class TicketDetailResponse(TicketResponse):
    processed_date: Optional[datetime] = None
    policy_desc: Optional[str] = None
    reason: Optional[str] = None
    reasoning: Optional[Dict[str, Any]] = None
    thought_process: Optional[List[Dict[str, Any]]] = None


def process_ticket_task(ticket_data: Dict[str, Any]):
    """
    Manual fallback ticket processor (not used for Gmail).
    """
    try:
        initial_state = SupportAgentState(
            messages=[HumanMessage(content=ticket_data["description"])]
        )

        final_state = graph_app.invoke(initial_state)

        db = SessionLocal()
        save_ticket_state(ticket_data, final_state, db)
        db.close()
    except Exception as e:
        print(f"Error processing ticket {ticket_data['ticket_id']}: {e}")
        raise e


@app.post("/tickets", response_model=TicketResponse)
async def create_ticket(
    ticket: TicketRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Manual test endpoint (not Gmail).
    """

    existing_ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket.ticket_id).first()
    if existing_ticket:
        return {
            "ticket_id": ticket.ticket_id,
            "status": "error",
            "message": "Ticket already exists",
        }

    new_ticket = Ticket(
        ticket_id=ticket.ticket_id,
        customer_id=ticket.customer_id,
        description=ticket.ticket_description,
        received_date=ticket.received_date,
        status="processing",
    )
    db.add(new_ticket)
    db.commit()

    background_tasks.add_task(
        process_ticket_task,
        {
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "description": ticket.ticket_description,
            "received_date": ticket.received_date,
            "status": "processing",
        }
    )

    return {
        "ticket_id": ticket.ticket_id,
        "status": "processing",
        "message": "Processing started.",
    }


@app.get("/tickets/{ticket_id}", response_model=TicketDetailResponse)
async def get_ticket(ticket_id: str, db: Session = Depends(get_db)):

    ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ts = db.query(TicketState).filter(TicketState.ticket_id == ticket.id).first()
    if not ts:
        return {
            "ticket_id": ticket.ticket_id,
            "status": ticket.status,
            "message": "Not finished",
        }

    return {
        "ticket_id": ticket.ticket_id,
        "status": ticket.status,
        "message": "OK",
        "customer_id": ticket.customer_id,
        "description": ticket.description,
        "received_date": ticket.received_date,
        "processed_date": ticket.processed_date,
        "problems": ts.problems,
        "policy_name": ts.policy_name,
        "policy_desc": ts.policy_desc,
        "action_taken": ts.action_taken,
        "reason": ts.reason,
        "reasoning": ts.reasoning,
        "thought_process": ts.thought_process,
        "messages": ts.messages,
    }


@app.get("/tickets", response_model=List[TicketResponse])
async def list_tickets(db: Session = Depends(get_db)):
    tickets = db.query(Ticket).all()
    response = []

    for t in tickets:
        item = {
            "ticket_id": t.ticket_id,
            "status": t.status,
            "message": "OK",
            "description": t.description,
            "customer_id": t.customer_id,
        }

        ts = db.query(TicketState).filter(TicketState.ticket_id == t.id).first()
        if ts:
            item["problems"] = ts.problems
            item["policy_name"] = ts.policy_name
            item["action_taken"] = ts.action_taken
            item["messages"] = ts.messages

        response.append(item)

    return response


# ---------------------------------------------------------------------------
# Run Server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)