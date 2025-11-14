"""
Initialize the PostgreSQL database tables for the support ticket system
"""
from database.ticket_db import create_tables
from database.seed_policies import seed_policies_from_py
from database.seed_products import seed_products
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
    seed_policies_from_py()
    seed_products()
    print("Database tables created successfully!")
    print(f"Using database URL: {os.getenv('DATABASE_URL', 'Not set - using default')}")

if __name__ == "__main__":
    main()
