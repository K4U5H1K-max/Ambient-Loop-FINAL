"""Database models and session management."""
from db.models import Ticket, TicketState, Policy, Product, GmailMessage, GmailConfig, Base
from db.session import (
    SessionLocal, engine, get_db, get_session, create_tables,
    AsyncSessionLocal, async_engine, get_async_session,
    async_get_last_history_id, async_set_last_history_id,
    async_claim_message, async_update_gmail_message_status, async_save_ticket_state,
)

__all__ = [
    "Ticket", "TicketState", "Policy", "Product", "GmailMessage", "GmailConfig", "Base",
    "SessionLocal", "engine", "get_db", "get_session", "create_tables",
    "AsyncSessionLocal", "async_engine", "get_async_session",
    "async_get_last_history_id", "async_set_last_history_id",
    "async_claim_message", "async_update_gmail_message_status", "async_save_ticket_state",
]
