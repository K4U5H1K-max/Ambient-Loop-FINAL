"""
Initialize the PostgreSQL database tables for the support ticket system
"""
from database.ticket_db import create_tables, SessionLocal
from database.seed_policies import seed_policies_from_py
from database.seed_products import seed_products
from database.memory import get_policy_memory, get_product_memory, seed_policy_memory, seed_product_memory
from langgraph.store.postgres import PostgresStore
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

STORE_URL = os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/support_tickets")

def main():
    """
    Create all database tables and seed data
    """
    print("🔧 Step 1: Creating database tables...")
    create_tables()
    
    print("\n🔧 Step 2: Initializing LangGraph PostgresStore...")
    try:
        with PostgresStore.from_conn_string(STORE_URL) as store:
            store.setup()
            print("✅ LangGraph store schema initialized")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize store: {e}")
    
    print("\n🌱 Step 3: Seeding policies...")
    seed_policies_from_py()
    
    print("\n🌱 Step 4: Seeding products...")
    seed_products()
    
    print("\n🌱 Step 5: Seeding LangGraph memory store...")
    db = SessionLocal()
    try:
        with get_policy_memory() as policy_store:
            seed_policy_memory(db, policy_store)
        with get_product_memory() as product_store:
            seed_product_memory(db, product_store)
    except Exception as e:
        print(f"⚠️ Warning: Could not seed memory store: {e}")
    finally:
        db.close()
    
    print("\n✅ Database initialization complete!")
    print(f"Using database URL: {os.getenv('DATABASE_URL', 'Not set - using default')}")

if __name__ == "__main__":
    main()
