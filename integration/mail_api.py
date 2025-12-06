"""
Gmail + LangGraph integration (fully async).

Flow: Gmail WATCH → Pub/Sub → /gmail/push → process_new_emails() → LangGraph → reply
"""

import os
import pickle
import logging
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
from datetime import datetime
from email.mime.text import MIMEText

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langgraph_sdk import get_client

from db import (
    get_async_session,
    async_get_last_history_id,
    async_set_last_history_id,
    async_claim_message,
    async_update_gmail_message_status,
    async_save_ticket_state,
)

# =============================================================================
# CONFIG
# =============================================================================

LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://127.0.0.1:2024")
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "customer_support_agent")

ROOT = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(ROOT, "token.pickle")
CREDS_PATH = os.path.join(ROOT, "credentials.json")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

client = get_client(url=LANGGRAPH_URL)
_authenticated_email: Optional[str] = None

# Thread pool for Gmail API calls (which are synchronous)
_executor = ThreadPoolExecutor(max_workers=4)


# =============================================================================
# GMAIL HELPERS (sync - run in executor)
# =============================================================================

def get_gmail_service():
    """Authenticate and return Gmail API service."""
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDS_PATH):
                raise RuntimeError(f"credentials.json not found at {CREDS_PATH}")
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "wb") as f:
            pickle.dump(creds, f)
    
    return build("gmail", "v1", credentials=creds)


def _get_authenticated_email(service) -> str:
    """Get authenticated Gmail address (sync)."""
    global _authenticated_email
    if _authenticated_email is None:
        try:
            profile = service.users().getProfile(userId="me").execute()
            _authenticated_email = profile.get("emailAddress", "").lower()
        except Exception:
            _authenticated_email = ""
    return _authenticated_email


def _get_history(service, start_history_id: str) -> dict:
    """Get Gmail history (sync)."""
    return service.users().history().list(userId="me", startHistoryId=start_history_id).execute()


def _get_full_message(service, msg_id: str) -> Dict[str, Any]:
    """Get full Gmail message (sync)."""
    return service.users().messages().get(userId="me", id=msg_id, format="full").execute()


def _send_reply(service, *, to: str, subject: str, body: str, thread_id: str, in_reply_to: str = None) -> bool:
    """Send reply in same Gmail thread (sync)."""
    try:
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        
        msg = MIMEText(body)
        msg["to"] = to
        msg["subject"] = subject
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = in_reply_to
        
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        service.users().messages().send(userId="me", body={"raw": raw, "threadId": thread_id}).execute()
        return True
    except HttpError as e:
        logging.error(f"Failed to send reply: {e}")
        return False


def _mark_as_read(service, msg_id: str):
    """Remove UNREAD label (sync)."""
    try:
        service.users().messages().modify(
            userId="me", id=msg_id, 
            body={"removeLabelIds": ["UNREAD"]}
        ).execute()
    except HttpError:
        pass


def parse_email(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata and body from Gmail message."""
    payload = msg.get("payload", {})
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", []) if h.get("name")}
    
    def extract_body(p):
        if p.get("mimeType") == "text/plain" and "data" in p.get("body", {}):
            return base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8", errors="ignore")
        for part in p.get("parts", []) or []:
            if text := extract_body(part):
                return text
        return ""
    
    return {
        "subject": headers.get("subject", "(no subject)"),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "message_id_header": headers.get("message-id", ""),
        "body": extract_body(payload) or "(no body)",
    }


# =============================================================================
# ASYNC GMAIL WRAPPERS
# =============================================================================

async def get_authenticated_email(service) -> str:
    """Get authenticated Gmail address (async)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _get_authenticated_email, service)


async def get_history(service, start_history_id: str) -> dict:
    """Get Gmail history (async)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _get_history, service, start_history_id)


async def get_full_message(service, msg_id: str) -> Dict[str, Any]:
    """Get full Gmail message (async)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _get_full_message, service, msg_id)


async def send_reply(service, *, to: str, subject: str, body: str, thread_id: str, in_reply_to: str = None) -> bool:
    """Send reply (async)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, 
        lambda: _send_reply(service, to=to, subject=subject, body=body, thread_id=thread_id, in_reply_to=in_reply_to)
    )


async def mark_as_read(service, msg_id: str):
    """Mark message as read (async)."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _mark_as_read, service, msg_id)


# =============================================================================
# LANGGRAPH
# =============================================================================

async def run_agent(email: Dict[str, Any]) -> Dict[str, Any]:
    """Run LangGraph agent on email, return result."""
    thread = await client.threads.create(metadata={"source": "gmail", "from": email["from"]})
    thread_id = thread.get("thread_id") or thread.get("id")
    
    run_input = {
        "messages": [{"role": "human", "content": f"From: {email['from']}\nSubject: {email['subject']}\n\n{email['body']}"}],
        "email_from": email["from"],
        "email_subject": email["subject"],
        "email_body": email["body"],
    }
    
    interrupted = False
    async for chunk in client.runs.stream(thread_id, ASSISTANT_ID, input=run_input, stream_mode="updates"):
        if chunk and chunk.event == "updates" and "__interrupt__" in (chunk.data or {}):
            interrupted = True
            break
    
    state = await client.threads.get_state(thread_id=thread_id)
    values = state.get("values", {})
    
    return {
        "thread_id": thread_id,
        "interrupted": interrupted,
        "email_reply": values.get("email_reply"),
        "action_taken": values.get("action_taken", ""),
        "requires_human_review": values.get("requires_human_review", False),
        "values": values,
    }


# =============================================================================
# MAIN PROCESSING (fully async)
# =============================================================================

async def process_new_emails(service, new_history_id: str):
    """Process new emails from Gmail history. Called by /gmail/push."""
    async with get_async_session() as session:
        try:
            last = await async_get_last_history_id(session)
            if not last:
                await async_set_last_history_id(session, new_history_id)
                logging.info(f"Initialized history_id={new_history_id}")
                return

            logging.info(f"Processing history {last} → {new_history_id}")
            
            resp = await get_history(service, last)
            history = resp.get("history", [])
            
            if not history:
                await async_set_last_history_id(session, new_history_id)
                return

            my_email = await get_authenticated_email(service)

            for h in history:
                for entry in h.get("messagesAdded", []):
                    if msg_id := entry.get("message", {}).get("id"):
                        await process_email(service, session, msg_id, my_email)

            await async_set_last_history_id(session, new_history_id)

        except Exception as e:
            logging.error(f"Error processing emails: {e}")


async def process_email(service, session, msg_id: str, my_email: str):
    """Process single email: claim → run agent → reply → update status."""
    try:
        # Fetch email
        gmail_msg = await get_full_message(service, msg_id)
        gmail_thread_id = gmail_msg.get("threadId", "")
        email = parse_email(gmail_msg)
        
        # Skip self-sent
        if my_email and my_email in email["from"].lower():
            return
        
        # Atomic claim
        if not await async_claim_message(session, msg_id, gmail_thread_id, email["from"], email["subject"]):
            logging.info(f"Skipping {msg_id} - already claimed")
            return
        
        logging.info(f"Processing {msg_id}: {email['subject']}")
        
        # Run agent
        result = await run_agent(email)
        
        # Handle interrupt/human review
        if result["interrupted"] or result["requires_human_review"]:
            logging.info(f"{msg_id} requires human review")
            await async_update_gmail_message_status(session, msg_id, "awaiting_human", langgraph_thread_id=result["thread_id"])
            return
        
        # Handle denied actions
        if "denied" in result["action_taken"].lower():
            await async_update_gmail_message_status(session, msg_id, "action_denied")
            return
        
        # Get reply
        reply = (result.get("email_reply") or "").strip()
        if not reply:
            logging.warning(f"No reply for {msg_id}")
            return
        
        # Save ticket
        try:
            await async_save_ticket_state(
                session,
                {"ticket_id": msg_id, "customer_id": email["from"], "description": email["body"], "received_date": datetime.utcnow().isoformat()},
                result["values"],
            )
        except Exception as e:
            logging.warning(f"Failed to save ticket: {e}")
        
        # Send reply
        sent = await send_reply(service, to=email["from"], subject=email["subject"], body=reply, thread_id=gmail_thread_id, in_reply_to=email["message_id_header"])
        if sent:
            await mark_as_read(service, msg_id)
            await async_update_gmail_message_status(session, msg_id, "completed", langgraph_thread_id=result["thread_id"])
            logging.info(f"✅ Replied to {msg_id}")
        else:
            logging.error(f"❌ Failed to reply to {msg_id}")

    except Exception as e:
        logging.error(f"Error processing {msg_id}: {e}")
