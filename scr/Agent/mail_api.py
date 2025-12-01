"""
Ambient Gmail + LangGraph (Agent Inbox) integration, push Pub/Sub style.

External flow:
    Gmail WATCH -> Google Pub/Sub topic -> push -> /gmail/push (FastAPI)
    /gmail/push -> process_gmail_history() -> incoming_email_queue

Internal flow:
    incoming_email_queue -> email_worker() -> process_email_event()
    process_email_event() -> LangGraph run + save_ticket_state() + Gmail reply
    human_resolution_monitor() -> handles Agent Inbox interrupts

Rules:
- Reply ALWAYS taken from nodes.py → thread_state["values"]["email_reply"]
- No regex. No guessing.
- L1/L2 always reply.
- L3 → interrupt → pending → human_monitor checks → send only if approved.
- If human ignores/denies → message frozen (unread).
"""

import os
import json
import pickle
import logging
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import base64
from email.mime.text import MIMEText

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langgraph_sdk import get_client

# =====================================================================
# CONFIG
# =====================================================================

LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://127.0.0.1:2024")
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "customer_support_agent")  # your assistant id

# Gmail
ROOT = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(ROOT, "token.pickle")
CREDS_PATH = os.path.join(ROOT, "credentials.json")
STATE_PATH = os.path.join(ROOT, "state.json")

# Gmail scopes: read, send, mark-as-read
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

# How often to check for human-resolved threads (seconds)
HUMAN_MONITOR_INTERVAL = int(os.getenv("HUMAN_MONITOR_INTERVAL", "20"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# LangGraph async client
client = get_client(url=LANGGRAPH_URL)

# Pub/sub queue: producer = process_gmail_history (push), consumer = email_worker
incoming_email_queue: asyncio.Queue = asyncio.Queue()


# =====================================================================
# STATE PERSISTENCE (Gmail <-> LangGraph mapping)
# =====================================================================

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"seen_messages": {}, "last_history_id": None}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "seen_messages" not in data:
                data["seen_messages"] = {}
            return data
    except Exception as e:
        logging.error(f"Failed to load state.json: {e}")
        return {"seen_messages": {}, "last_history_id": None}


def save_state(state: Dict[str, Any]) -> None:
    tmp_path = STATE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, STATE_PATH)


def get_last_history_id() -> Optional[str]:
    state = load_state()
    return state.get("last_history_id")


def set_last_history_id(history_id: str) -> None:
    state = load_state()
    state["last_history_id"] = history_id
    save_state(state)


def set_message_status(
    msg_id: str,
    *,
    gmail_thread_id: str,
    langgraph_thread_id: str,
    status: str,
    last_run_id: Optional[str] = None,
) -> None:
    """
    Persist per-message state.

    status:
        - "pending"          : queued for processing
        - "processing"       : currently being handled
        - "awaiting_human"   : interrupted, waiting for Agent Inbox decision
        - "completed"        : final email sent, Gmail marked read
        - "action_denied"    : human denied/ignored critical action, no auto email
    """
    state = load_state()
    seen = state.setdefault("seen_messages", {})
    seen[msg_id] = {
        "gmail_thread_id": gmail_thread_id,
        "langgraph_thread_id": langgraph_thread_id,
        "status": status,
        "last_run_id": last_run_id,
        "updated_at": datetime.utcnow().isoformat(),
    }
    save_state(state)


def get_message_record(msg_id: str) -> Optional[Dict[str, Any]]:
    state = load_state()
    return state.get("seen_messages", {}).get(msg_id)


def iter_messages_by_status(status: str) -> List[Dict[str, Any]]:
    state = load_state()
    result = []
    for mid, rec in state.get("seen_messages", {}).items():
        if rec.get("status") == status:
            rec_copy = rec.copy()
            rec_copy["msg_id"] = mid
            result.append(rec_copy)
    return result


# =====================================================================
# GMAIL HELPERS
# =====================================================================

def get_gmail_service():
    """Authenticate and return a Gmail API service."""
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDS_PATH):
                raise RuntimeError(
                    f"credentials.json not found at {CREDS_PATH}. "
                    f"Download OAuth client credentials from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "wb") as token:
            pickle.dump(creds, token)

    try:
        service = build("gmail", "v1", credentials=creds)
        return service
    except Exception as e:
        logging.error(f"Failed to build Gmail service: {e}")
        raise


def get_full_message(service, msg_id: str) -> Dict[str, Any]:
    try:
        return (
            service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )
    except HttpError as e:
        logging.error(f"Error fetching message {msg_id}: {e}")
        raise


def parse_message_metadata(msg: Dict[str, Any]) -> Dict[str, Any]:
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])

    meta: Dict[str, str] = {}
    for h in headers:
        name = h.get("name")
        value = h.get("value")
        if name and value:
            meta[name.lower()] = value

    subject = meta.get("subject", "(no subject)")
    from_addr = meta.get("from", "")
    to_addr = meta.get("to", "")
    msg_id_header = meta.get("message-id", "")

    body = extract_plain_body(payload)

    return {
        "subject": subject,
        "from": from_addr,
        "to": to_addr,
        "message_id_header": msg_id_header,
        "body": body,
    }


def extract_plain_body(payload: Dict[str, Any]) -> str:
    """
    Traverse payload parts to find a text/plain body.
    """
    def _walk(p):
        if p.get("mimeType") == "text/plain" and "data" in p.get("body", {}):
            data = p["body"]["data"]
            return base64.urlsafe_b64decode(data.encode("utf-8")).decode(
                "utf-8", errors="ignore"
            )
        for part in p.get("parts", []) or []:
            text = _walk(part)
            if text:
                return text
        return ""

    body = _walk(payload)
    if not body:
        body = "(no plain text body found)"
    return body


def build_reply_message(
    to_addr: str,
    original_subject: str,
    reply_text: str,
    thread_id: str,
    in_reply_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a MIME reply, encoded for Gmail API."""
    subject = original_subject
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    msg = MIMEText(reply_text)
    msg["to"] = to_addr
    msg["subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return {
        "raw": raw,
        "threadId": thread_id,
    }


def send_reply(
    service,
    *,
    to_addr: str,
    subject: str,
    reply_text: str,
    gmail_thread_id: str,
    in_reply_to: Optional[str],
) -> bool:
    """Send a reply in the same Gmail thread."""
    try:
        message = build_reply_message(
            to_addr=to_addr,
            original_subject=subject,
            reply_text=reply_text,
            thread_id=gmail_thread_id,
            in_reply_to=in_reply_to,
        )
        sent = (
            service.users()
            .messages()
            .send(userId="me", body=message)
            .execute()
        )
        logging.info(f"Sent reply Gmail message id={sent.get('id')}")
        return True
    except HttpError as e:
        logging.error(f"Failed to send reply email: {e}")
        return False


def mark_message_as_read(service, msg_id: str) -> bool:
    """Remove UNREAD label from a message."""
    try:
        body = {"removeLabelIds": ["UNREAD"], "addLabelIds": []}
        (
            service.users()
            .messages()
            .modify(userId="me", id=msg_id, body=body)
            .execute()
        )
        logging.info(f"Marked message {msg_id} as read")
        return True
    except HttpError as e:
        if e.resp.status == 403 and "insufficientPermissions" in str(e):
            logging.error(
                "Insufficient permissions to modify message. "
                "Ensure 'gmail.modify' scope is included. "
                "Delete token.pickle and re-authenticate."
            )
        else:
            logging.error(f"Failed to mark message {msg_id} as read: {e}")
        return False


# =====================================================================
# LANGGRAPH INTEGRATION
# =====================================================================

def make_langgraph_thread_id(gmail_identifier: str) -> str:
    """
    Stable mapping from a Gmail identifier -> LangGraph thread_id.
    """
    return f"gmail-{gmail_identifier}"


async def ensure_langgraph_thread(
    requested_thread_id: str, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Ensure a LangGraph thread exists for an incoming Gmail thread.
    Returns the actual `thread_id` created or existing on the server.
    """
    try:
        if isinstance(requested_thread_id, str) and requested_thread_id.startswith(
            "gmail-"
        ):
            resp = await client.threads.create(metadata=metadata or {})
            created_thread_id = (
                resp.get("thread_id") or resp.get("id") or resp.get("threadId")
            )
            if not created_thread_id:
                created_thread_id = resp.get("id") if isinstance(resp, dict) else None

            if not created_thread_id:
                logging.error(
                    f"Could not determine created thread_id from response: {resp}"
                )
                raise RuntimeError("Failed to determine LangGraph thread id")

            return created_thread_id

        try:
            await client.threads.get_state(thread_id=requested_thread_id)
            return requested_thread_id
        except Exception:
            logging.info(
                f"Requested LangGraph thread {requested_thread_id} not found; creating new thread"
            )
            resp = await client.threads.create(metadata=metadata or {})
            created_thread_id = (
                resp.get("thread_id") or resp.get("id") or resp.get("threadId")
            )
            if not created_thread_id:
                created_thread_id = resp.get("id") if isinstance(resp, dict) else None
            if not created_thread_id:
                logging.error(
                    f"Could not determine created thread_id from response: {resp}"
                )
                raise RuntimeError("Failed to determine LangGraph thread id")
            return created_thread_id

    except Exception as e:
        logging.error(f"Failed to ensure LangGraph thread {requested_thread_id}: {e}")
        raise


def build_run_input_from_email(email_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Gmail metadata into the graph's expected input.
    """
    from_addr = email_meta["from"]
    subject = email_meta["subject"]
    body = email_meta["body"]

    content = (
        f"New customer email received.\n\n"
        f"From: {from_addr}\n"
        f"Subject: {subject}\n\n"
        f"Body:\n{body}"
    )

    return {
        "messages": [
            {
                "role": "human",
                "content": content,
            }
        ],
        "email_from": from_addr,
        "email_subject": subject,
        "email_body": body,
    }


def thread_state_has_interrupt(thread_state: Dict[str, Any]) -> bool:
    """
    Check if thread state still has pending interrupts.
    """
    interrupts = thread_state.get("interrupts")
    if interrupts:
        return True
    values = thread_state.get("values", {})
    interrupts2 = values.get("interrupts")
    return bool(interrupts2)


def thread_state_has_resolution(thread_state: Dict[str, Any]) -> bool:
    """
    Resolution exists iff nodes.py wrote a non-empty values["email_reply"].
    No regex. No guessing.
    """
    if thread_state_has_interrupt(thread_state):
        return False
    values = thread_state.get("values", thread_state)
    reply = values.get("email_reply")
    return isinstance(reply, str) and bool(reply.strip())


# =====================================================================
# PUSH HANDLER: PROCESS GMAIL HISTORY
# =====================================================================

async def process_gmail_history(service, new_history_id: str):
    """
    Called from FastAPI /gmail/push after decoding Gmail Pub/Sub payload.

    Uses last_history_id from state.json to fetch only new messages via
    Gmail 'history.list', then enqueues them into incoming_email_queue.
    """
    try:
        last = get_last_history_id()
        if last is None:
            set_last_history_id(new_history_id)
            logging.info(
                f"Initialized last_history_id={new_history_id}, skipping first batch."
            )
            return

        logging.info(f"Processing Gmail history from {last} to {new_history_id}")

        resp = (
            service.users()
            .history()
            .list(userId="me", startHistoryId=last)
            .execute()
        )

        history = resp.get("history", [])
        if not history:
            logging.info("No new history records.")
            set_last_history_id(new_history_id)
            return

        for h in history:
            msgs_added = h.get("messagesAdded", [])
            for entry in msgs_added:
                m = entry.get("message", {})
                msg_id = m.get("id")
                if not msg_id:
                    continue

                record = get_message_record(msg_id)
                if record and record.get("status") in (
                    "completed",
                    "awaiting_human",
                    "action_denied",
                ):
                    continue

                gmail_msg = get_full_message(service, msg_id)
                gmail_thread_id = gmail_msg.get("threadId")
                email_meta = parse_message_metadata(gmail_msg)

                if record and record.get("langgraph_thread_id"):
                    langgraph_thread_id = record.get("langgraph_thread_id")
                else:
                    langgraph_thread_id = make_langgraph_thread_id(msg_id)
                    set_message_status(
                        msg_id,
                        gmail_thread_id=gmail_thread_id,
                        langgraph_thread_id=langgraph_thread_id,
                        status="pending",
                        last_run_id=None,
                    )

                event = {
                    "msg_id": msg_id,
                    "gmail_thread_id": gmail_thread_id,
                    "langgraph_thread_id": langgraph_thread_id,
                    "email_meta": email_meta,
                }

                await incoming_email_queue.put(event)
                logging.info(
                    f"[PUSH] Enqueued msg {msg_id} from '{email_meta['from']}' "
                    f"subject='{email_meta['subject']}'"
                )

        set_last_history_id(new_history_id)

    except Exception as e:
        logging.error(f"Error in process_gmail_history: {e}")


# =====================================================================
# WORKERS (PUB/SUB)
# =====================================================================

async def process_email_event(event: Dict[str, Any], service):
    """
    Subscriber handler: processes a single email event.
    Runs in its own Task so one long run doesn't block others.
    """
    msg_id = event["msg_id"]
    gmail_thread_id = event["gmail_thread_id"]
    langgraph_thread_id = event["langgraph_thread_id"]
    email_meta = event["email_meta"]

    # -----------------------------------------------------------------
    # SKIP SELF-SENT EMAILS (Prevents re-processing our own replies)
    # -----------------------------------------------------------------
    SELF_SENDERS = {
        "narasimha112503@gmail.com",
    }
    sender = (email_meta.get("from") or "").strip()
    if any(s in sender for s in SELF_SENDERS):
        logging.info(
            f"[SKIP] Ignoring self-sent email msg_id={msg_id} from {sender}"
        )
        try:
            incoming_email_queue.task_done()
        except Exception:
            pass
        return

    # -----------------------------------------------------------------
    # DUPLICATE PREVENTION: Skip if already completed / frozen
    # -----------------------------------------------------------------
    existing = get_message_record(msg_id)
    if existing and existing.get("status") in ("completed", "action_denied"):
        logging.info(
            f"Skipping msg_id={msg_id}; already handled with status="
            f"{existing.get('status')}."
        )
        try:
            incoming_email_queue.task_done()
        except Exception:
            pass
        return

    logging.info(f"Processing email msg_id={msg_id} thread={gmail_thread_id}")

    try:
        set_message_status(
            msg_id,
            gmail_thread_id=gmail_thread_id,
            langgraph_thread_id=langgraph_thread_id,
            status="processing",
        )

        created_thread_id = await ensure_langgraph_thread(
            langgraph_thread_id,
            metadata={
                "source": "gmail",
                "gmail_thread_id": gmail_thread_id,
                "gmail_msg_id": msg_id,
                "from": email_meta["from"],
                "subject": email_meta["subject"],
            },
        )

        if created_thread_id and created_thread_id != langgraph_thread_id:
            langgraph_thread_id = created_thread_id
            set_message_status(
                msg_id,
                gmail_thread_id=gmail_thread_id,
                langgraph_thread_id=langgraph_thread_id,
                status="processing",
                last_run_id=None,
            )

        run_input = build_run_input_from_email(email_meta)

        interrupt_detected = False
        last_run_id = None

        async for chunk in client.runs.stream(
            langgraph_thread_id,
            ASSISTANT_ID,
            input=run_input,
            stream_mode="updates",
        ):
            if not chunk or not chunk.data:
                continue

            if chunk.event == "metadata":
                last_run_id = chunk.data.get("run_id", last_run_id)

            if chunk.event == "updates":
                data = chunk.data
                if "__interrupt__" in data:
                    interrupt_detected = True
                    logging.info(f"Interrupt detected for msg_id={msg_id}")
                    break

        if interrupt_detected:
            logging.info(
                f"Run interrupted for msg_id={msg_id}; waiting for Agent Inbox "
                f"decision."
            )
            set_message_status(
                msg_id,
                gmail_thread_id=gmail_thread_id,
                langgraph_thread_id=langgraph_thread_id,
                status="awaiting_human",
                last_run_id=last_run_id,
            )
            return

        thread_state = await client.threads.get_state(
            thread_id=langgraph_thread_id
        )

        if not thread_state_has_resolution(thread_state):
            logging.warning(f"No clear resolution yet for msg_id={msg_id}")
            return

        # Strict conditions ONLY for L3 ignore/deny (tier classification or refund/resend)
        values = thread_state.get("values", {})
        requires_human_review = values.get("requires_human_review", False)
        action_taken = values.get("action_taken")
        action_str = str(action_taken).lower() if action_taken else ""

        # Unified Option C:
        # If requires_human_review or action contains 'denied'/'ignore' → freeze
        if (
            requires_human_review
            or "denied" in action_str
            or "ignore" in action_str
        ):
            logging.info(
                f"❌ Human denial/ignore for msg_id={msg_id}. "
                f"Not sending email and freezing message (action_denied)."
            )
            set_message_status(
                msg_id,
                gmail_thread_id=gmail_thread_id,
                langgraph_thread_id=langgraph_thread_id,
                status="action_denied",
                last_run_id=last_run_id,
            )
            # Do NOT mark as read. Leave unread and never process again.
            return

        # L1/L2 (and L3 approved) ALWAYS reply — reply taken ONLY from email_reply
        reply_text = values.get("email_reply")
        if not isinstance(reply_text, str) or not reply_text.strip():
            logging.error(f"No email_reply found in state for msg_id={msg_id}")
            return

        reply_text = reply_text.strip()

        # Save to PostgreSQL
        try:
            from database.ticket_db import SessionLocal, save_ticket_state

            db = SessionLocal()
            try:
                save_ticket_state(
                    {
                        "ticket_id": msg_id,
                        "customer_id": email_meta.get("from"),
                        "description": email_meta.get("body"),
                        "received_date": datetime.utcnow().isoformat(),
                    },
                    values,
                    db,
                )
            finally:
                db.close()
        except Exception as e:
            logging.exception(
                f"Failed to save ticket state for msg_id={msg_id}: {e}"
            )

        success = send_reply(
            service,
            to_addr=email_meta["from"],
            subject=email_meta["subject"],
            reply_text=reply_text,
            gmail_thread_id=gmail_thread_id,
            in_reply_to=email_meta["message_id_header"],
        )

        if success:
            mark_message_as_read(service, msg_id)
            set_message_status(
                msg_id,
                gmail_thread_id=gmail_thread_id,
                langgraph_thread_id=langgraph_thread_id,
                status="completed",
                last_run_id=last_run_id,
            )
            logging.info(f"✅ Email sent and marked read for msg_id={msg_id}")
        else:
            logging.error(f"❌ Failed to send reply for msg_id={msg_id}")

    except Exception as e:
        logging.error(f"Error processing msg_id={msg_id}: {e}")
    finally:
        try:
            incoming_email_queue.task_done()
        except Exception:
            pass


async def email_worker(service):
    """
    Subscriber: reads events from the queue and spawns a Task per message.
    """
    logging.info("Starting email worker (subscriber)...")
    while True:
        event = await incoming_email_queue.get()
        asyncio.create_task(process_email_event(event, service))
        await asyncio.sleep(0)


async def human_resolution_monitor(service):
    """
    Background worker:

    - Scans all messages with status 'awaiting_human'.
    - For each, checks LangGraph thread_state.
    - Once human has accepted & run has completed:
        - Uses values['email_reply'] as final reply,
        - Sends Gmail reply,
        - Marks message as read,
        - Marks status 'completed'.

    If supervisor ignores/denies (tier classification or refund/resend):
        - No reply is sent,
        - Message stays UNREAD,
        - Status becomes 'action_denied' (frozen).
    """
    logging.info("Starting human resolution monitor...")
    while True:
        try:
            awaiting = iter_messages_by_status("awaiting_human")

            for rec in awaiting:
                msg_id = rec["msg_id"]
                lg_thread_id = rec["langgraph_thread_id"]
                gmail_thread_id = rec["gmail_thread_id"]

                logging.info(
                    f"Checking human resolution for msg_id={msg_id}, "
                    f"thread={lg_thread_id}"
                )

                try:
                    thread_state = await client.threads.get_state(
                        thread_id=lg_thread_id
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to get thread_state for {lg_thread_id}: {e}"
                    )
                    continue

                if not thread_state_has_resolution(thread_state):
                    continue

                values = thread_state.get("values", {})
                requires_human_review = values.get(
                    "requires_human_review", False
                )
                action_taken = values.get("action_taken")
                action_str = str(action_taken).lower() if action_taken else ""

                # Unified Option C for human-resolved path as well
                if (
                    requires_human_review
                    or "denied" in action_str
                    or "ignore" in action_str
                ):
                    logging.info(
                        f"IGNORED/DENIED by supervisor for msg_id={msg_id}. "
                        f"Freezing as action_denied (no reply, stays unread)."
                    )
                    set_message_status(
                        msg_id,
                        gmail_thread_id=gmail_thread_id,
                        langgraph_thread_id=lg_thread_id,
                        status="action_denied",
                        last_run_id=rec.get("last_run_id"),
                    )
                    continue

                gmail_msg = get_full_message(service, msg_id)
                email_meta = parse_message_metadata(gmail_msg)

                reply_text = values.get("email_reply")
                if not isinstance(reply_text, str) or not reply_text.strip():
                    logging.warning(
                        f"Resolved thread but no email_reply for msg_id={msg_id}"
                    )
                    continue

                reply_text = reply_text.strip()

                success = send_reply(
                    service,
                    to_addr=email_meta["from"],
                    subject=email_meta["subject"],
                    reply_text=reply_text,
                    gmail_thread_id=gmail_thread_id,
                    in_reply_to=email_meta["message_id_header"],
                )

                if success:
                    mark_message_as_read(service, msg_id)
                    set_message_status(
                        msg_id,
                        gmail_thread_id=gmail_thread_id,
                        langgraph_thread_id=lg_thread_id,
                        status="completed",
                        last_run_id=rec.get("last_run_id"),
                    )
                    logging.info(
                        f"Human-resolved reply sent and message marked read "
                        f"for msg_id={msg_id}"
                    )

        except Exception as e:
            logging.error(f"Human resolution monitor error: {e}")

        await asyncio.sleep(HUMAN_MONITOR_INTERVAL)


# MAIN
async def main():
    service = get_gmail_service()
    logging.info("Gmail + LangGraph ambient agent (push-based) starting up...")

    await asyncio.gather(
        email_worker(service),       # subscriber
        human_resolution_monitor(service),
    )


if __name__ == "__main__":
    asyncio.run(main())