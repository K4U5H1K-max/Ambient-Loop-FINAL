"""
Test script for the FastAPI server and PostgreSQL integration
"""
import requests
import json
from datetime import datetime
import time
import sys

BASE_URL = "http://localhost:8000"

def test_create_ticket():
    """
    Test creating a new ticket via the API
    """
    print("Testing ticket creation...")
    
    # Generate a unique ticket ID based on timestamp
    ticket_id = f"TICKET-{int(time.time())}"
    
    # Create ticket payload
    payload = {
        "ticket_id": ticket_id,
        "customer_id": "CUST-001",
        "ticket_description": "I ordered product2 in my order #ORD54321, but received product4 instead. What should I do?",
        "received_date": datetime.now().isoformat()
    }
    
    # Send POST request to create ticket
    response = requests.post(f"{BASE_URL}/tickets/", json=payload)
    
    # Print response
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return ticket_id

def test_get_ticket(ticket_id):
    """
    Test retrieving a ticket by ID
    """
    print(f"\nTesting ticket retrieval for {ticket_id}...")
    
    # Wait a bit for processing to complete
    print("Waiting for ticket processing to complete...")
    time.sleep(5)  # Wait 5 seconds
    
    # Send GET request to retrieve ticket
    response = requests.get(f"{BASE_URL}/tickets/{ticket_id}")
    
    # Print response
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # If ticket is still processing, wait and try again
    if response.json().get("status") == "processing":
        print("\nTicket still processing, waiting longer...")
        time.sleep(10)  # Wait 10 more seconds
        
        # Try again
        response = requests.get(f"{BASE_URL}/tickets/{ticket_id}")
        print(f"Status code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_list_tickets():
    """
    Test listing all tickets
    """
    print("\nTesting ticket listing...")
    
    # Send GET request to list tickets
    response = requests.get(f"{BASE_URL}/tickets/")
    
    # Print response
    print(f"Status code: {response.status_code}")
    print(f"Found {len(response.json())} tickets")
    
    # Print first ticket if available
    if response.json():
        print(f"First ticket: {json.dumps(response.json()[0], indent=2)}")

def main():
    """
    Run all tests
    """
    try:
        # Test creating a ticket
        ticket_id = test_create_ticket()
        
        # Test getting the ticket
        test_get_ticket(ticket_id)
        
        # Test listing all tickets
        test_list_tickets()
        
        print("\nAll tests completed!")
        
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to the API server at {BASE_URL}")
        print("Make sure the server is running with: uvicorn api_server:app --reload")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
