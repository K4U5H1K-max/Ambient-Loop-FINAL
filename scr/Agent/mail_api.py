"""
Ambient Gmail poller + LangGraph (Agent Inbox) integration, pub/sub style.

- Publisher:
    gmail_poller() -> pushes events into incoming_email_queue
- Subscriber:
    email_worker() -> pulls events and calls process_email_event()
- Human resolution worker:
    human_resolution_monitor() -> handles Agent Inbox interrupts

If a run completes with no interrupts:
    - Pull final state from LangGraph
    - Extract reply text
    - Save ticket + state into PostgreSQL
    - Send Gmail reply and mark original message as read
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
import re

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

# Polling interval (seconds)
POLL_INTERVAL = int(os.getenv("GMAIL_POLL_INTERVAL", "30"))

# How often to check for human-resolved threads (seconds)
HUMAN_MONITOR_INTERVAL = int(os.getenv("HUMAN_MONITOR_INTERVAL", "20"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# LangGraph async client
client = get_client(url=LANGGRAPH_URL)

# Pub/sub queue: publisher = gmail_poller, subscriber = email_worker
incoming_email_queue: asyncio.Queue = asyncio.Queue()


# =====================================================================
# STATE PERSISTENCE (Gmail <-> LangGraph mapping)
# =====================================================================

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"seen_messages": {}}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load state.json: {e}")
        return {"seen_messages": {}}


def save_state(state: Dict[str, Any]) -> None:
    tmp_path = STATE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, STATE_PATH)


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


def list_unread_messages(service, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    List unread messages in INBOX.
    """
    try:
        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                labelIds=["INBOX"],
                q="is:unread",
                maxResults=max_results,
            )
            .execute()
        )
        return response.get("messages", [])
    except HttpError as e:
        logging.error(f"Error listing unread messages: {e}")
        return []


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
            return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="ignore")
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


async def ensure_langgraph_thread(requested_thread_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Ensure a LangGraph thread exists for an incoming Gmail thread.
    Returns the actual `thread_id` created or existing on the server.
    """
    try:
        if isinstance(requested_thread_id, str) and requested_thread_id.startswith("gmail-"):
            resp = await client.threads.create(metadata=metadata or {})
            created_thread_id = resp.get("thread_id") or resp.get("id") or resp.get("threadId")
            if not created_thread_id:
                created_thread_id = resp.get("id") if isinstance(resp, dict) else None

            if not created_thread_id:
                logging.error(f"Could not determine created thread_id from response: {resp}")
                raise RuntimeError("Failed to determine LangGraph thread id")

            return created_thread_id

        try:
            await client.threads.get_state(thread_id=requested_thread_id)
            return requested_thread_id
        except Exception:
            logging.info(f"Requested LangGraph thread {requested_thread_id} not found; creating new thread")
            resp = await client.threads.create(metadata=metadata or {})
            created_thread_id = resp.get("thread_id") or resp.get("id") or resp.get("threadId")
            if not created_thread_id:
                created_thread_id = resp.get("id") if isinstance(resp, dict) else None
            if not created_thread_id:
                logging.error(f"Could not determine created thread_id from response: {resp}")
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


def extract_reply_from_thread_state(thread_state: Dict[str, Any]) -> Optional[str]:
    """
    Extract the final reply text from thread_state.
    """
    values = thread_state.get("values", thread_state)

    try:
        logging.debug(f"extract_reply: thread_state keys: {list(thread_state.keys())}")
        logging.debug(f"extract_reply: values (truncated): {json.dumps(values, default=str)[:2000]}")
    except Exception:
        logging.debug("extract_reply: failed to serialize thread_state for debug")

    for key in ("email_reply", "reply", "final_reply", "assistant_reply"):
        v = values.get(key) if isinstance(values, dict) else None
        if isinstance(v, str) and v.strip():
            txt = v.strip()
            m = re.search(r"\bDear\b", txt, re.IGNORECASE)
            return txt[m.start():].strip() if m else txt

    def walk_for_candidates(obj):
        candidates = []

        def walk(o):
            if isinstance(o, dict):
                for k, vv in o.items():
                    lk = k.lower()
                    if isinstance(vv, str):
                        if lk in ("email_reply", "reply", "final_reply", "assistant_reply", "response"):
                            candidates.append(vv)
                    walk(vv)
            elif isinstance(o, list):
                for item in o:
                    walk(item)

        walk(obj)
        return candidates

    candidates = walk_for_candidates(values)
    if candidates:
        candidates = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
        if candidates:
            candidates.sort(key=lambda s: len(s), reverse=True)
            txt = candidates[0]
            m = re.search(r"\bDear\b", txt, re.IGNORECASE)
            return txt[m.start():].strip() if m else txt

    def find_message_in_node(o):
        if isinstance(o, dict):
            for k, vv in o.items():
                lk = k.lower()
                if lk in ("messages", "history", "outputs", "events") and isinstance(vv, list):
                    for entry in reversed(vv):
                        if isinstance(entry, dict):
                            cont = (
                                entry.get("content")
                                or entry.get("text")
                                or entry.get("value")
                                or entry.get("output")
                            )
                            if isinstance(cont, str) and cont.strip():
                                return cont.strip()
                            for subkey in ("content", "text", "value", "output"):
                                sub = entry.get(subkey)
                                if isinstance(sub, str) and sub.strip():
                                    return sub.strip()
                        elif isinstance(entry, str) and entry.strip():
                            return entry.strip()
                else:
                    res = find_message_in_node(vv)
                    if res:
                        return res
        elif isinstance(o, list):
            for item in reversed(o):
                res = find_message_in_node(item)
                if res:
                    return res
        return None

    msg = find_message_in_node(values)
    if msg:
        txt = msg
        m = re.search(r"\bDear\b", txt, re.IGNORECASE)
        return txt[m.start():].strip() if m else txt

    try:
        s = json.dumps(values, default=str)
        m = re.search(r"Resolution Summary:\\s*(Dear[\s\S]{20,2000})", s)
        if m:
            txt = m.group(0)
            m2 = re.search(r"\bDear\b", txt, re.IGNORECASE)
            return txt[m2.start():].strip() if m2 else txt
    except Exception:
        pass

    return None


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
    Heuristic to decide if the agent finished with a resolution.
    """
    if thread_state_has_interrupt(thread_state):
        return False

    values = thread_state.get("values", thread_state)

    if values.get("action_taken"):
        return True

    for key in ("email_reply", "final_reply", "reply", "assistant_reply"):
        v = values.get(key)
        if isinstance(v, str) and v.strip():
            return True

    msgs = values.get("messages") or []

    def looks_like_final_message(s: str) -> bool:
        if not s or not isinstance(s, str):
            return False
        low = s.lower()
        if "resolution summary" in low:
            return True
        if "✅ **resolution**" in s or "✅ resolution" in low:
            return True
        if "dear" in low:
            return True
        return False

    for m in reversed(msgs):
        if isinstance(m, dict):
            cont = m.get("content") or m.get("text") or m.get("value") or m.get("output")
        else:
            cont = m
        if isinstance(cont, str) and looks_like_final_message(cont):
            return True

    return False


# =====================================================================
# WORKERS (PUB/SUB)
# =====================================================================

async def gmail_poller(service):
    """
    Publisher: polls Gmail and publishes new messages to the incoming_email_queue.

    It only enqueues messages that are:
    - Unread
    - Not yet tracked as completed / awaiting_human
    """
    logging.info("Starting Gmail poller (publisher)...")
    while True:
        try:
            messages = list_unread_messages(service)
            if messages:
                logging.info(f"Found {len(messages)} unread messages")

            for msg_stub in messages:
                msg_id = msg_stub["id"]
                record = get_message_record(msg_id)

                if record and record.get("status") in ("completed", "awaiting_human"):
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
                    f"Enqueued message {msg_id} from '{email_meta['from']}' subject='{email_meta['subject']}'"
                )

            await asyncio.sleep(POLL_INTERVAL)

        except Exception as e:
            logging.error(f"Gmail poller error: {e}")
            await asyncio.sleep(POLL_INTERVAL)


async def process_email_event(event: Dict[str, Any], service):
    """
    Subscriber handler: processes a single email event.
    Runs in its own Task so one long run doesn't block others.
    """
    msg_id = event["msg_id"]
    gmail_thread_id = event["gmail_thread_id"]
    langgraph_thread_id = event["langgraph_thread_id"]
    email_meta = event["email_meta"]

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
            logging.info(
                f"Using LangGraph thread id {created_thread_id} (mapped from requested {langgraph_thread_id})"
            )
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
                    logging.info(
                        f"Interrupt detected for msg_id={msg_id}: {data['__interrupt__']}"
                    )
                    try:
                        thread_state = await client.threads.get_state(thread_id=langgraph_thread_id)
                        try:
                            dump = json.dumps(thread_state, default=str)
                            logging.info(
                                f"Full thread_state (truncated 8k) for interrupted msg_id={msg_id}: {dump[:8192]}"
                            )
                        except Exception:
                            logging.info(
                                f"Interrupted thread_state for msg_id={msg_id} (non-serializable)"
                            )
                    except Exception as e:
                        logging.info(
                            f"Failed to fetch thread_state for interrupted msg_id={msg_id}: {e}"
                        )
                    break

        record = get_message_record(msg_id) or {}
        set_message_status(
            msg_id,
            gmail_thread_id=gmail_thread_id,
            langgraph_thread_id=langgraph_thread_id,
            status="awaiting_human" if interrupt_detected else record.get("status", "processing"),
            last_run_id=last_run_id,
        )

        if interrupt_detected:
            logging.info(
                f"Run interrupted for msg_id={msg_id}; waiting for Agent Inbox decision."
            )
            set_message_status(
                msg_id,
                gmail_thread_id=gmail_thread_id,
                langgraph_thread_id=langgraph_thread_id,
                status="awaiting_human",
                last_run_id=last_run_id,
            )
            return

        # Final state, non-interrupted
        thread_state = await client.threads.get_state(thread_id=langgraph_thread_id)

        if not thread_state_has_resolution(thread_state):
            logging.warning(
                f"No clear resolution yet for msg_id={msg_id} "
                f"(but no interrupts). Skipping reply this cycle."
            )
            return

        reply_text = extract_reply_from_thread_state(thread_state)
        if not reply_text:
            logging.warning(
                f"Could not extract reply text from thread_state for msg_id={msg_id}"
            )
            return

        # Save into PostgreSQL (reusing ticket_db logic)
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
                        "interrupt": None,
                    },
                    thread_state.get("values", {}),
                    db,
                )
            finally:
                db.close()
        except Exception as e:
            logging.exception(f"Failed to save ticket state for msg_id={msg_id}: {e}")

        # Send Gmail reply
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
        else:
            logging.error(f"Failed to send reply for msg_id={msg_id}")

    except Exception as e:
        logging.error(f"Error while processing msg_id={msg_id}: {e}")

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
        - Extracts final reply,
        - Sends Gmail reply,
        - Marks message as read,
        - Marks status 'completed'.
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
                    f"Checking human resolution for msg_id={msg_id}, thread={lg_thread_id}"
                )

                try:
                    thread_state = await client.threads.get_state(thread_id=lg_thread_id)
                except Exception as e:
                    logging.error(f"Failed to get thread_state for {lg_thread_id}: {e}")
                    continue

                if not thread_state_has_resolution(thread_state):
                    continue

                try:
                    tool_data = {}
                    for key in ("tools", "tool_calls", "tool_responses", "calls"):
                        if key in thread_state:
                            tool_data[key] = thread_state.get(key)
                        else:
                            vals = thread_state.get("values", {}) or {}
                            if key in vals:
                                tool_data[key] = vals.get(key)
                    if tool_data:
                        try:
                            logging.info(
                                f"Found tool-related data for msg_id={msg_id} "
                                f"(truncated 8k): {json.dumps(tool_data, default=str)[:8192]}"
                            )
                        except Exception:
                            logging.info(
                                f"Found tool-related data for msg_id={msg_id} (non-serializable)"
                            )
                except Exception as e:
                    logging.debug(f"Error while inspecting tool data for msg_id={msg_id}: {e}")

                gmail_msg = get_full_message(service, msg_id)
                email_meta = parse_message_metadata(gmail_msg)

                reply_text = extract_reply_from_thread_state(thread_state)
                if not reply_text:
                    logging.warning(
                        f"Resolved thread but no reply text for msg_id={msg_id}"
                    )
                    continue

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
                        f"Human-resolved reply sent and message marked read for msg_id={msg_id}"
                    )

        except Exception as e:
            logging.error(f"Human resolution monitor error: {e}")

        await asyncio.sleep(HUMAN_MONITOR_INTERVAL)


# =====================================================================
# MAIN
# =====================================================================

async def main():
    service = get_gmail_service()
    logging.info("Gmail + LangGraph ambient agent starting up...")

    await asyncio.gather(
        gmail_poller(service),       # publisher
        email_worker(service),       # subscriber
        human_resolution_monitor(service),
    )


if __name__ == "__main__":
    asyncio.run(main())