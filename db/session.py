"""Database session management - async and sync support."""
import os
import json
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from db.models import Base, Ticket, TicketState, GmailMessage, GmailConfig

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/support_tickets")
# Convert to async URL
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Sync engine (for migrations, scripts)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Async engine
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

logger = logging.getLogger(__name__)


def create_tables():
    """Create all database tables (sync)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency for sync database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session() -> Session:
    """Get a new sync database session."""
    return SessionLocal()


@asynccontextmanager
async def get_async_session():
    """Get an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# =============================================================================
# ASYNC DATABASE OPERATIONS
# =============================================================================

async def async_get_last_history_id(session: AsyncSession) -> Optional[str]:
    """Get last Gmail history ID (async)."""
    from sqlalchemy import select
    result = await session.execute(select(GmailConfig).limit(1))
    config = result.scalar_one_or_none()
    return config.last_history_id if config else None


async def async_set_last_history_id(session: AsyncSession, history_id: str):
    """Set last Gmail history ID (async)."""
    from sqlalchemy import select
    result = await session.execute(select(GmailConfig).limit(1))
    config = result.scalar_one_or_none()
    
    if not config:
        config = GmailConfig(id=1, last_history_id=history_id)
        session.add(config)
    else:
        config.last_history_id = history_id
        config.updated_at = datetime.now()
    
    await session.commit()


async def async_claim_message(session: AsyncSession, msg_id: str, gmail_thread_id: str, sender: str, subject: str) -> bool:
    """Atomically claim a message for processing (async)."""
    from sqlalchemy.dialects.postgresql import insert
    
    stmt = insert(GmailMessage).values(
        gmail_msg_id=msg_id,
        gmail_thread_id=gmail_thread_id,
        sender_email=sender,
        subject=subject,
        status="processing",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    ).on_conflict_do_nothing(index_elements=["gmail_msg_id"])
    
    result = await session.execute(stmt)
    await session.commit()
    
    return result.rowcount > 0


async def async_update_gmail_message_status(session: AsyncSession, gmail_msg_id: str, status: str, **kwargs):
    """Update Gmail message status (async)."""
    from sqlalchemy import select
    
    result = await session.execute(
        select(GmailMessage).where(GmailMessage.gmail_msg_id == gmail_msg_id)
    )
    msg = result.scalar_one_or_none()
    
    if msg:
        msg.status = status
        msg.updated_at = datetime.now()
        for key, value in kwargs.items():
            if hasattr(msg, key):
                setattr(msg, key, value)
        await session.commit()


async def async_save_ticket_state(session: AsyncSession, ticket_data: dict, state_data):
    """Save ticket and its state to the database (async)."""
    from sqlalchemy import select
    
    try:
        result = await session.execute(
            select(Ticket).where(Ticket.ticket_id == ticket_data["ticket_id"])
        )
        ticket = result.scalar_one_or_none()
        
        if ticket:
            ticket.processed_date = datetime.now()
            ticket.status = "resolved"
        else:
            received = ticket_data.get("received_date")
            if isinstance(received, str):
                received = datetime.fromisoformat(received)
            
            ticket = Ticket(
                ticket_id=ticket_data["ticket_id"],
                customer_id=ticket_data.get("customer_id"),
                description=ticket_data.get("description"),
                received_date=received,
                processed_date=datetime.now(),
                status="resolved"
            )
            session.add(ticket)
        
        await session.flush()
        
        # Extract state values
        if hasattr(state_data, 'get'):
            problems = state_data.get('problems', [])
            policy_name = state_data.get('policy_name')
            policy_desc = state_data.get('policy_desc')
            action_taken = state_data.get('action_taken')
            reasoning = state_data.get('reasoning', {})
            thought_process = state_data.get('thought_process', [])
            messages = state_data.get('messages', [])
        else:
            problems = getattr(state_data, 'problems', [])
            policy_name = getattr(state_data, 'policy_name', None)
            policy_desc = getattr(state_data, 'policy_desc', None)
            action_taken = getattr(state_data, 'action_taken', None)
            reasoning = getattr(state_data, 'reasoning', {})
            thought_process = getattr(state_data, 'thought_process', [])
            messages = getattr(state_data, 'messages', [])
        
        # Serialize messages
        serialized_messages = []
        for msg in messages:
            if hasattr(msg, 'to_dict'):
                serialized_messages.append(msg.to_dict())
            elif hasattr(msg, 'content'):
                serialized_messages.append({'content': msg.content, 'type': getattr(msg, 'type', 'unknown')})
            elif isinstance(msg, dict):
                serialized_messages.append(msg)
        
        result = await session.execute(
            select(TicketState).where(TicketState.ticket_id == ticket.id)
        )
        state = result.scalar_one_or_none()
        
        state_values = {
            "messages": serialized_messages,
            "problems": problems,
            "policy_name": policy_name,
            "policy_desc": policy_desc,
            "action_taken": action_taken,
            "reasoning": reasoning,
            "thought_process": json.loads(json.dumps(thought_process, default=str)),
        }
        
        if state:
            for key, value in state_values.items():
                setattr(state, key, value)
        else:
            state = TicketState(ticket_id=ticket.id, **state_values)
            session.add(state)
        
        await session.commit()
        logger.info(f"Saved ticket {ticket_data['ticket_id']}")
        return ticket
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Error saving ticket: {e}")
        raise


# =============================================================================
# SYNC WRAPPERS (for scripts/seeding)
# =============================================================================

def update_gmail_message_status(db: Session, gmail_msg_id: str, status: str, **kwargs):
    """Update Gmail message status (sync)."""
    msg = db.query(GmailMessage).filter(GmailMessage.gmail_msg_id == gmail_msg_id).first()
    if msg:
        msg.status = status
        msg.updated_at = datetime.now()
        for key, value in kwargs.items():
            if hasattr(msg, key):
                setattr(msg, key, value)
        db.commit()


def save_ticket_state(ticket_data: dict, state_data, db: Session):
    """Save ticket and its state to the database (sync)."""
    try:
        ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_data["ticket_id"]).first()
        
        if ticket:
            ticket.processed_date = datetime.now()
            ticket.status = "resolved"
        else:
            received = ticket_data.get("received_date")
            if isinstance(received, str):
                received = datetime.fromisoformat(received)
            
            ticket = Ticket(
                ticket_id=ticket_data["ticket_id"],
                customer_id=ticket_data.get("customer_id"),
                description=ticket_data.get("description"),
                received_date=received,
                processed_date=datetime.now(),
                status="resolved"
            )
            db.add(ticket)
        
        db.flush()
        
        if hasattr(state_data, 'get'):
            problems = state_data.get('problems', [])
            policy_name = state_data.get('policy_name')
            policy_desc = state_data.get('policy_desc')
            action_taken = state_data.get('action_taken')
            reasoning = state_data.get('reasoning', {})
            thought_process = state_data.get('thought_process', [])
            messages = state_data.get('messages', [])
        else:
            problems = getattr(state_data, 'problems', [])
            policy_name = getattr(state_data, 'policy_name', None)
            policy_desc = getattr(state_data, 'policy_desc', None)
            action_taken = getattr(state_data, 'action_taken', None)
            reasoning = getattr(state_data, 'reasoning', {})
            thought_process = getattr(state_data, 'thought_process', [])
            messages = getattr(state_data, 'messages', [])
        
        serialized_messages = []
        for msg in messages:
            if hasattr(msg, 'to_dict'):
                serialized_messages.append(msg.to_dict())
            elif hasattr(msg, 'content'):
                serialized_messages.append({'content': msg.content, 'type': getattr(msg, 'type', 'unknown')})
            elif isinstance(msg, dict):
                serialized_messages.append(msg)
        
        state = db.query(TicketState).filter(TicketState.ticket_id == ticket.id).first()
        
        state_values = {
            "messages": serialized_messages,
            "problems": problems,
            "policy_name": policy_name,
            "policy_desc": policy_desc,
            "action_taken": action_taken,
            "reasoning": reasoning,
            "thought_process": json.loads(json.dumps(thought_process, default=str)),
        }
        
        if state:
            for key, value in state_values.items():
                setattr(state, key, value)
        else:
            state = TicketState(ticket_id=ticket.id, **state_values)
            db.add(state)
        
        db.commit()
        logger.info(f"Saved ticket {ticket_data['ticket_id']}")
        return ticket
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving ticket: {e}")
        raise
