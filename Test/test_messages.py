#!/usr/bin/env python
"""
Test script to verify that messages are being properly serialized and stored in the database.
"""
from state import SupportAgentState
from langchain_core.messages import HumanMessage, AIMessage
from database.ticket_db import save_ticket_state, get_db, Ticket, TicketState
from datetime import datetime
import uuid

# Create a test ticket with messages
def test_messages():
    # Create a unique ticket ID
    ticket_id = f"TEST-MESSAGES-{uuid.uuid4().hex[:8]}"
    
    # Create ticket data
    ticket_data = {
        "ticket_id": ticket_id,
        "customer_id": "test@example.com",
        "description": "This is a test ticket with messages",
        "received_date": datetime.now(),
        "status": "processing"
    }
    
    # Create a state with messages
    state = SupportAgentState()
    state.messages = [
        HumanMessage(content="Hello, I have a problem with my order"),
        AIMessage(content="I'm sorry to hear that. What's the issue?"),
        HumanMessage(content="My order hasn't arrived yet"),
        AIMessage(content="✅ I'll help you track your order and resolve this issue.")
    ]
    state.problems = ["non-delivery", "delayed"]
    state.policy_name = "Non-Delivery Resolution"
    state.policy_desc = "If a customer hasn't received their order within the expected timeframe, we will track the order and provide a resolution."
    state.action_taken = "Track order"
    state.reason = "Order is delayed"
    
    # Get a database session
    db = next(get_db())
    
    try:
        # Save the ticket and state to the database
        save_ticket_state(ticket_data, state, db)
        print(f"Ticket saved: {ticket_id}")
        
        # Query the database to verify the messages were saved
        # Get the ticket ID from the database
        ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
        if not ticket:
            print(f"❌ Ticket not found: {ticket_id}")
            return
            
        # Get the ticket state
        ticket_state = db.query(TicketState).filter(TicketState.ticket_id == ticket.id).first()
        if not ticket_state:
            print(f"❌ Ticket state not found for ticket: {ticket_id}")
            return
            
        print(f"Messages saved: {ticket_state.messages}")
        
        # Verify the messages were saved correctly
        if ticket_state.messages and len(ticket_state.messages) == 4:
            print("✅ Messages were saved correctly!")
            
            # Print each message
            for i, msg in enumerate(ticket_state.messages):
                print(f"Message {i+1}: {msg['type']} - {msg['content']}")
        else:
            print("❌ Messages were not saved correctly!")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    test_messages() 