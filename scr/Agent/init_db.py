"""
Initialize the PostgreSQL database tables for the support ticket system
"""
from database.ticket_db import create_tables
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """
    Create all database tables
    """
    print("Initializing database tables...")
    create_tables()
    print("Database tables created successfully!")
    print(f"Using database URL: {os.getenv('DATABASE_URL', 'Not set - using default')}")

if __name__ == "__main__":
    main()
