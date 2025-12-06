"""SQLAlchemy database models."""
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, JSON, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class Ticket(Base):
    """Support ticket."""
    __tablename__ = "tickets"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String, unique=True, index=True)
    customer_id = Column(String, index=True)
    description = Column(Text)
    received_date = Column(DateTime, default=datetime.now)
    processed_date = Column(DateTime, nullable=True)
    status = Column(String, default="new")
    
    state_data = relationship("TicketState", back_populates="ticket", uselist=False, cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "ticket_id": self.ticket_id,
            "customer_id": self.customer_id,
            "description": self.description,
            "received_date": self.received_date.isoformat() if self.received_date else None,
            "processed_date": self.processed_date.isoformat() if self.processed_date else None,
            "status": self.status
        }


class TicketState(Base):
    """Ticket processing state."""
    __tablename__ = "ticket_states"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"))
    messages = Column(JSON, nullable=True)
    problems = Column(JSON)
    policy_name = Column(String, nullable=True)
    policy_desc = Column(Text, nullable=True)
    policy_reason = Column(Text, nullable=True)
    action_taken = Column(String, nullable=True)
    reason = Column(Text, nullable=True)
    reasoning = Column(JSON, nullable=True)
    thought_process = Column(JSON, nullable=True)
    interrupt = Column(JSON, nullable=True)
    
    ticket = relationship("Ticket", back_populates="state_data")
    
    def to_dict(self):
        return {
            "id": self.id,
            "ticket_id": self.ticket_id,
            "problems": self.problems,
            "policy_name": self.policy_name,
            "policy_desc": self.policy_desc,
            "action_taken": self.action_taken,
            "messages": self.messages,
            "reasoning": self.reasoning,
            "thought_process": self.thought_process
        }


class Policy(Base):
    """Support policy."""
    __tablename__ = "policies"
    
    id = Column(Integer, primary_key=True, index=True)
    policy_name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=False)
    when_to_use = Column(Text)
    applicable_problems = Column(JSON)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "policy_name": self.policy_name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "applicable_problems": self.applicable_problems,
        }


class Product(Base):
    """Product catalog."""
    __tablename__ = "products"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    price = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    weight = Column(Float)
    dimensions = Column(JSONB)


class GmailMessage(Base):
    """Gmail message tracking."""
    __tablename__ = "gmail_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    gmail_msg_id = Column(String, unique=True, index=True, nullable=False)
    gmail_thread_id = Column(String, index=True)
    langgraph_thread_id = Column(String, nullable=True)
    sender_email = Column(String)
    subject = Column(String)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def to_dict(self):
        return {
            "gmail_msg_id": self.gmail_msg_id,
            "status": self.status,
            "subject": self.subject,
            "sender_email": self.sender_email,
        }


class GmailConfig(Base):
    """Gmail configuration (single row)."""
    __tablename__ = "gmail_config"
    
    id = Column(Integer, primary_key=True, default=1)
    last_history_id = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
