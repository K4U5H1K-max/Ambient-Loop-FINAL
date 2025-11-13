"""
Script to create the support_tickets database if it doesn't exist
"""
import psycopg2
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def create_database():
    """
    Create the support_tickets database if it doesn't exist
    """
    # Connection parameters for the default postgres database
    conn_params = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
        "database": "postgres"  # Connect to default postgres database first
    }
    
    # Wait for PostgreSQL to be ready
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"Attempting to connect to PostgreSQL (attempt {retry_count + 1}/{max_retries})...")
            conn = psycopg2.connect(**conn_params)
            conn.autocommit = True
            print("Successfully connected to PostgreSQL!")
            break
        except psycopg2.OperationalError as e:
            print(f"Could not connect to PostgreSQL: {e}")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Could not connect to PostgreSQL.")
                raise
    
    # Check if support_tickets database exists
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'support_tickets'")
        exists = cursor.fetchone()
        
        if not exists:
            print("Creating 'support_tickets' database...")
            cursor.execute("CREATE DATABASE support_tickets")
            print("Database 'support_tickets' created successfully!")
        else:
            print("Database 'support_tickets' already exists.")
        
        cursor.close()
        conn.close()
        
        # Test connection to the support_tickets database
        db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/support_tickets")
        print(f"Testing connection to {db_url}...")
        
        # Extract connection parameters from DATABASE_URL
        # This is a simple parser and might need to be adjusted for complex URLs
        if "://" in db_url:
            auth_part = db_url.split("://")[1].split("@")[0]
            user_pass = auth_part.split(":")
            user = user_pass[0]
            password = user_pass[1] if len(user_pass) > 1 else ""
            
            host_part = db_url.split("@")[1].split("/")[0]
            host_port = host_part.split(":")
            host = host_port[0]
            port = host_port[1] if len(host_port) > 1 else "5432"
            
            database = db_url.split("/")[-1].split("?")[0]
            
            # Connect to the support_tickets database
            conn_params = {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "database": database
            }
            
            test_conn = psycopg2.connect(**conn_params)
            test_conn.close()
            print(f"Successfully connected to {database} database!")
            return True
        else:
            print(f"Invalid DATABASE_URL format: {db_url}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if create_database():
        print("Database setup completed successfully!")
    else:
        print("Database setup failed.")
