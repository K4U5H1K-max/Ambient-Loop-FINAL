"""
LangGraph Store integration for policies and products memory.
"""
from langgraph.store.postgres import PostgresStore
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv

load_dotenv()

STORE_URL = os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/support_tickets")

def get_policy_memory():
    """Return a PostgresStore context manager for policy namespace."""
    return PostgresStore.from_conn_string(STORE_URL)

def get_product_memory():
    """Return a PostgresStore context manager for product namespace."""
    return PostgresStore.from_conn_string(STORE_URL)

def get_policies_context(query: str = "", limit: int = 25) -> str:
    """Fetch policies from memory store for LLM context."""
    uri = STORE_URL
    if not uri:
        # Fallback to static policies
        from database.policies import format_policies_for_llm
        return format_policies_for_llm()
    
    lines = []
    try:
        with PostgresStore.from_conn_string(uri) as store:
            # For queries, return all policies since they are general
            items = store.search(("policies",), limit=limit)
            for item in items:
                val = item.value
                lines.append(f"- {val.get('policy_name','?')}: {val.get('description','')} (Problems: {val.get('applicable_problems','[]')})")
        if lines:
            return "\n".join(lines)
        else:
            # Fallback if store is empty
            from database.policies import format_policies_for_llm
            return format_policies_for_llm()
    except Exception as e:
        # Fallback on error
        from database.policies import format_policies_for_llm
        return format_policies_for_llm()

def get_products_context(limit: int = 25) -> str:
    """Fetch products from memory store for LLM context."""
    uri = STORE_URL
    if not uri:
        return "(No product memory available)"
    
    lines = []
    try:
        with PostgresStore.from_conn_string(uri) as store:
            items = store.search(("products",))
            for item in items[:limit]:
                val = item.value
                lines.append(
                    f"- {val.get('id','?')}: {val.get('name','?')} | ${val.get('price','?')} | {val.get('category','?')}"
                )
        return "\n".join(lines) if lines else "(No products found in memory)"
    except Exception as e:
        return f"(Error retrieving products: {str(e)})"

def seed_policy_memory(db: Session, store):
    """Seed LangMem from existing policies stored in the database."""
    from database.ticket_db import Policy
    
    # Check if already seeded
    try:
        existing_items = list(store.search(("policies",), limit=1))
        if existing_items:
            print(f"↩️ Policy memory already seeded. Skipping.")
            return
    except Exception:
        pass  # Store might be empty or not initialized
    
    policies = db.query(Policy).all()
    for policy in policies:
        store.put(
            ("policies",),
            f"policy:{policy.id}",
            {
                "policy_name": policy.policy_name,
                "description": policy.description,
                "when_to_use": policy.when_to_use,
                "applicable_problems": policy.applicable_problems,
            }
        )
    print(f"✅ Seeded {len(policies)} policies into PostgresStore.")

def seed_product_memory(db: Session, store):
    """Load all product metadata into LangMem."""
    from database.ticket_db import Product
    
    # Check if already seeded
    try:
        existing_items = list(store.search(("products",), limit=1))
        if existing_items:
            print(f"↩️ Product memory already seeded. Skipping.")
            return
    except Exception:
        pass  # Store might be empty or not initialized
    
    products = db.query(Product).all()
    for product in products:
        store.put(
            ("products",),
            f"product:{product.id}",
            {
                "id": product.id,
                "name": product.name,
                "description": product.description,
                "price": product.price,
                "category": product.category.value if hasattr(product.category, 'value') else product.category,
                "weight": product.weight,
                "dimensions": product.dimensions,
            }
        )
    print(f"✅ Seeded {len(products)} products into PostgresStore.")