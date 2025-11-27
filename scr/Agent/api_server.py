# api_server.py
"""
FastAPI server for Customer Support Agent + Ambient Gmail Integration
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any, List
import uvicorn

from sqlalchemy.orm import Session

# Import for async/threading
from contextlib import asynccontextmanager
import threading

# Import graph components
from graph import graph_app
from state import SupportAgentState
from langchain_core.messages import HumanMessage

# Import database components
from database.ticket_db import get_db, save_ticket_state, Ticket, TicketState, SessionLocal


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start Gmail Ambient Agent (pub/sub) in background thread.
    """
    import asyncio
    from mail_api import main as gmail_ambient_main

    def run_ambient():
        asyncio.run(gmail_ambient_main())

    thread = threading.Thread(target=run_ambient, daemon=True)
    thread.start()

    print("🚀 Ambient Gmail pub/sub agent running.")

    yield

    print("🛑 Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Customer Support Agent API",
    description="API for processing customer support tickets using LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request and response
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


# Process ticket in background
def process_ticket_task(ticket_data: Dict[str, Any]):
    """
    Process a ticket using the LangGraph workflow and save results to database.
    Runs in a FastAPI BackgroundTask thread.
    """
    try:
        # Create initial state with customer message
        initial_state = SupportAgentState(
            messages=[HumanMessage(content=ticket_data["description"])]
        )

        # Execute the local graph (this uses your local StateGraph, not Agent Inbox)
        final_state = graph_app.invoke(initial_state)

        print(f"Workflow completed for ticket {ticket_data['ticket_id']}")
        print(f"Final state: {final_state}")

        # Handle different state object types
        if hasattr(final_state, "get"):
            # Dict-like (e.g., AddableValuesDict)
            problems = final_state.get("problems", [])
            actions = final_state.get("actions", None)
            action_taken = actions[0] if isinstance(actions, list) and actions else None
            messages = final_state.get("messages", [])
        else:
            # Attribute-based state object
            problems = getattr(final_state, "problems", []) if hasattr(final_state, "problems") else []
            action_taken = getattr(final_state, "action_taken", None) if hasattr(final_state, "action_taken") else None
            messages = getattr(final_state, "messages", []) if hasattr(final_state, "messages") else []

        print(f"Problems identified: {problems}")
        print(f"Action taken: {action_taken}")

        # Save ticket and state to database
        print(f"Saving ticket and state to database: {ticket_data}, {final_state}")
        try:
            db_session = SessionLocal()
            try:
                save_ticket_state(ticket_data, final_state, db_session)
            finally:
                db_session.close()
        except Exception as e:
            print(f"Error saving ticket state: {str(e)}")

        return final_state
    except Exception as e:
        print(f"Error processing ticket {ticket_data['ticket_id']}: {str(e)}")
        raise e


@app.post("/tickets", response_model=TicketResponse)
async def create_ticket(
    ticket: TicketRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Create a new support ticket and process it asynchronously via the local graph.
    """
    print("\n=== RECEIVED TICKET REQUEST ===")
    print(f"ticket_id: {ticket.ticket_id}")
    print(f"ticket_description: {ticket.ticket_description[:50]}...")
    print(f"customer_id: {ticket.customer_id}")
    print(f"received_date: {ticket.received_date}")
    print("===============================\n")
    try:
        # Check if ticket already exists
        existing_ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket.ticket_id).first()
        if existing_ticket:
            return {
                "ticket_id": ticket.ticket_id,
                "status": "error",
                "message": f"Ticket with ID {ticket.ticket_id} already exists",
            }

        # Create a new ticket in the database with 'processing' status
        new_ticket = Ticket(
            ticket_id=ticket.ticket_id,
            customer_id=ticket.customer_id,
            description=ticket.ticket_description,
            received_date=ticket.received_date,
            status="processing",
        )
        db.add(new_ticket)
        db.commit()
        print(f"New ticket created: {new_ticket}")

        # Prepare ticket data for background processing
        ticket_data = {
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "description": ticket.ticket_description,
            "received_date": ticket.received_date,
            "status": "processing",
        }

        # Process ticket asynchronously
        background_tasks.add_task(process_ticket_task, ticket_data)

        return {
            "ticket_id": ticket.ticket_id,
            "status": "processing",
            "message": "Ticket received and being processed. Check status later using GET /tickets/{ticket_id}",
        }
    except Exception as e:
        db.rollback()
        return {
            "ticket_id": ticket.ticket_id,
            "status": "error",
            "message": f"Error creating ticket: {str(e)}",
        }


@app.get("/tickets/{ticket_id}", response_model=TicketDetailResponse)
async def get_ticket(ticket_id: str, db: Session = Depends(get_db)):
    """
    Get ticket details by ticket ID.
    """
    ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    try:
        ticket_state = db.query(
            TicketState.problems,
            TicketState.policy_name,
            TicketState.policy_desc,
            TicketState.action_taken,
            TicketState.reason,
            TicketState.reasoning,
            TicketState.thought_process,
            TicketState.messages,
        ).filter(TicketState.ticket_id == ticket.id).first()

        if not ticket_state:
            return {
                "ticket_id": ticket.ticket_id,
                "status": ticket.status,
                "message": "Ticket found but processing not complete",
                "customer_id": ticket.customer_id,
                "description": ticket.description,
                "received_date": ticket.received_date,
                "processed_date": ticket.processed_date,
                "messages": [],
            }

        return {
            "ticket_id": ticket.ticket_id,
            "status": ticket.status,
            "message": "Ticket processing complete",
            "customer_id": ticket.customer_id,
            "description": ticket.description,
            "received_date": ticket.received_date,
            "processed_date": ticket.processed_date,
            "problems": ticket_state.problems,
            "policy_name": ticket_state.policy_name,
            "policy_desc": ticket_state.policy_desc,
            "action_taken": ticket_state.action_taken,
            "reason": ticket_state.reason,
            "reasoning": ticket_state.reasoning,
            "thought_process": ticket_state.thought_process,
            "messages": ticket_state.messages if ticket_state.messages else [],
        }
    except Exception as e:
        print(f"Error accessing ticket state data: {str(e)}")
        db.rollback()
        return {
            "ticket_id": ticket.ticket_id,
            "status": ticket.status,
            "message": f"Error retrieving ticket state: {str(e)}",
            "customer_id": ticket.customer_id,
            "description": ticket.description,
            "received_date": ticket.received_date,
            "processed_date": ticket.processed_date,
            "messages": [],
        }


@app.get("/tickets", response_model=List[TicketResponse])
async def list_tickets(db: Session = Depends(get_db)):
    """
    List all tickets.
    """
    tickets = db.query(Ticket).all()

    result: List[TicketResponse] = []
    for ticket in tickets:
        ticket_data: Dict[str, Any] = {
            "ticket_id": ticket.ticket_id,
            "status": ticket.status,
            "message": "Ticket found",
            "description": ticket.description,
            "customer_id": ticket.customer_id,
        }

        try:
            ticket_state = db.query(
                TicketState.problems,
                TicketState.policy_name,
                TicketState.action_taken,
                TicketState.messages,
            ).filter(TicketState.ticket_id == ticket.id).first()

            if ticket_state:
                ticket_data.update(
                    {
                        "problems": ticket_state.problems,
                        "policy_name": ticket_state.policy_name,
                        "action_taken": ticket_state.action_taken,
                        "messages": ticket_state.messages if ticket_state.messages else [],
                    }
                )
        except Exception as e:
            print(f"Error accessing ticket state data: {str(e)}")
            db.rollback()
            ticket_data["messages"] = []

        result.append(ticket_data)

    return result


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)