"""
FastAPI server for Customer Support Agent + Gmail Pub/Sub Push Integration
---------------------------------------------------------------------------
Flow:
Gmail WATCH ---> Pub/Sub Topic ---> Subscription (PUSH) ---> /gmail/push
/gmail/push ---> process_gmail_history() ---> internal queue
email_worker + human_resolution_monitor run as background workers
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any, List
import uvicorn
import asyncio

from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

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
      - Launch Gmail push workers (email_worker + human_resolution_monitor)
    """
    from integration.mail_api import email_worker, human_resolution_monitor, get_gmail_service

    service = get_gmail_service()
    loop = asyncio.get_event_loop()

    loop.create_task(email_worker(service))
    loop.create_task(human_resolution_monitor(service))

    print("🚀 Gmail Pub/Sub Push workers started.")

    yield

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
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)