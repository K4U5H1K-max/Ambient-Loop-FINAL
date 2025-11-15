"""
Simple Gmail poller that notifies an ambient agent when new unread messages arrive.

How it works (simple polling implementation):
 - Uses OAuth2 installed app flow; expects `credentials.json` (from Google Cloud Console) in project root.
 - Stores user token in `token.pickle` after first auth.
 - Polls Gmail for unread messages every `poll_interval_seconds` from `config.json`.
 - For each unseen message, fetches metadata (From, Subject) and either prints or POSTs to configured webhook.

Notes:
 - This is a simple approach for development. For production, consider Gmail push notifications via Cloud Pub/Sub + verified endpoints.
"""
import os
import time
import json
import pickle
import logging
from typing import Dict, List
import requests
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request 
from googleapiclient.discovery import build 
from googleapiclient.errors import HttpError 
from graph import graph_app
from langchain_core.messages import HumanMessage
import base64
from email.mime.text import MIMEText


# SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify"  # Required to mark messages as read
]

ROOT = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(ROOT, "token.pickle")
CREDS_PATH = os.path.join(ROOT, "credentials.json")
STATE_PATH = os.path.join(ROOT, "state.json")
CONFIG_PATH = os.path.join(ROOT, "config.json")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def load_config() -> Dict:
    if not os.path.exists(CONFIG_PATH):
        return {"poll_interval_seconds": 30, "webhook_url": "", "notify_via_webhook": False}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDS_PATH):
                raise FileNotFoundError(
                    "credentials.json not found. Create OAuth 2.0 Client ID credentials and save as credentials.json"
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_PATH, "wb") as token:
            pickle.dump(creds, token)

    service = build("gmail", "v1", credentials=creds)
    return service


def load_state() -> Dict:
    if not os.path.exists(STATE_PATH):
        return {"seen_ids": []}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# def notify_agent(payload: Dict, config: Dict):
#     if config.get("notify_via_webhook") and config.get("webhook_url"):
#         try:
#             r = requests.post(config["webhook_url"], json=payload, timeout=5)
#             r.raise_for_status()
#             logging.info("Notified webhook: %s", config["webhook_url"])
#         except Exception as e:
#             logging.warning("Failed to notify webhook: %s", e)
#     else:
#         print(json.dumps(payload, ensure_ascii=False))
def notify_agent(payload: Dict, config: Dict):
    """
    Process incoming email through the LangGraph Support Agent and reply if it's a support case.
    """
    try:
        # Build email text
        email_text = (
            f"From: {payload.get('from', '')}\n"
            f"Subject: {payload.get('subject', '')}\n\n"
            f"{payload.get('body', '') or ''}"
        )

        # Run through LangGraph workflow
        final_state = graph_app.invoke({"messages": [HumanMessage(content=email_text)]})

        # Extract state values - handle both dict-like and object-like states
        if hasattr(final_state, 'get'):
            is_support_ticket = final_state.get('is_support_ticket', False)
            problems = final_state.get('problems', [])
            policy_name = final_state.get('policy_name', 'N/A')
            action_taken = final_state.get('action_taken', 'N/A')
            messages = final_state.get("messages", [])
        else:
            is_support_ticket = getattr(final_state, 'is_support_ticket', False)
            problems = getattr(final_state, 'problems', [])
            policy_name = getattr(final_state, 'policy_name', 'N/A')
            action_taken = getattr(final_state, 'action_taken', 'N/A')
            messages = getattr(final_state, 'messages', [])

        # Basic logging
        print("\n=== NEW EMAIL PROCESSED ===")
        print(f"From: {payload.get('from')}")
        print(f"Subject: {payload.get('subject')}")
        print(f"Support Ticket: {is_support_ticket}")
        print(f"Problems: {problems}")
        print(f"Policy: {policy_name}")
        print(f"Action: {action_taken}")

        # Get the last AI response (resolution message if it's a support ticket)
        ai_response = None

        if messages:
            # Get the last message content
            last_message = messages[-1]
            print(last_message)
            if hasattr(last_message, 'content'):
                ai_response = last_message.content
            elif isinstance(last_message, dict):
                ai_response = last_message.get('content', '')

        # Send auto-reply only if it's a support ticket
        reply_sent = False
        if is_support_ticket:
            if ai_response:
                print(f"\n--- AI Response ---\n{ai_response}\n")
                
                # Extract only the customer email part from AI response
                # The AI response contains: Resolution part + Customer email part
                # We need to extract only the "Dear ..." email section
                import re
                
                # Pattern to extract customer email starting with "Dear" and ending before any metadata
                customer_email = ai_response
                
                # Method 1: Extract content after "Resolution Summary:" or "Dear"
                if "Resolution Summary:" in ai_response:
                    customer_email = ai_response.split("Resolution Summary:", 1)[1].strip()
                elif re.search(r'Dear [^,\n]+,', ai_response):
                    # Find the position of "Dear ..." and extract from there
                    match = re.search(r'(Dear [^,\n]+,.*)', ai_response, re.DOTALL)
                    if match:
                        customer_email = match.group(1).strip()
                
                # Method 2: Remove resolution metadata (lines starting with ✅, ###, etc.)
                lines = customer_email.split('\n')
                cleaned_lines = []
                skip_section = False
                
                for line in lines:
                    # Skip resolution metadata lines
                    if line.strip().startswith('✅') or line.strip().startswith('###') or \
                       line.strip().startswith('**Resolution**') or line.strip().startswith('**Extract') or \
                       line.strip().startswith('1. **') or line.strip().startswith('2. **') or \
                       line.strip().startswith('3. **') or line.strip().startswith('4. **') or \
                       line.strip().startswith('5. **'):
                        skip_section = True
                        continue
                    
                    # Start including lines when we hit "Dear" or customer-facing content
                    if line.strip().startswith('Dear') or (skip_section and line.strip() and not line.strip().startswith('#')):
                        skip_section = False
                    
                    if not skip_section:
                        cleaned_lines.append(line)
                
                customer_email = '\n'.join(cleaned_lines).strip()
                
                # Fallback: If extraction failed, use the whole response
                if not customer_email or len(customer_email) < 50:
                    customer_email = ai_response
                
                print(f"\n--- Customer Email (Cleaned) ---\n{customer_email}\n")
                
                reply_to = payload.get("from") or "anikakarampuri.test@gmail.com"
                subject = f"Re: {payload.get('subject', 'Support Response')}"
                try:
                    send_email(get_gmail_service(), reply_to, subject, customer_email)
                    reply_sent = True
                    logging.info(f"✅ Support email sent to {reply_to}")
                except Exception as e:
                    logging.error(f"❌ Failed to send email: {e}")
                    raise  # Re-raise to prevent marking as read on failure
            else:
                logging.warning("⚠️ Support ticket detected but no response message found")
        else:
            logging.info("ℹ️ Not a support ticket — no reply sent")

        return {
            "status": "processed" if reply_sent or not is_support_ticket else "error",
            "is_support": is_support_ticket,
            "reply_sent": reply_sent
        }

    except Exception as e:
        logging.error(f"❌ notify_agent failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

   

def extract_body(payload):
    """Extract text body from Gmail message payload."""
    body = ""
    if "parts" in payload:
        # Multipart message
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part.get("body", {}).get("data")
                if data:
                    body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                    break
            elif part["mimeType"] == "text/html" and not body:
                # Fallback to HTML if no plain text
                data = part.get("body", {}).get("data")
                if data:
                    body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    else:
        # Single part message
        if payload.get("mimeType") == "text/plain":
            data = payload.get("body", {}).get("data")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    return body


def is_message_unread(service, msg_id: str) -> bool:
    """
    Check if a Gmail message is unread by checking its labelIds.
    Returns True if message is unread, False if read.
    """
    try:
        msg = service.users().messages().get(userId="me", id=msg_id, format="metadata", metadataHeaders=[]).execute()
        label_ids = msg.get("labelIds", [])
        # UNREAD label indicates the message is unread
        return "UNREAD" in label_ids
    except HttpError as e:
        logging.error("Error checking read status for message %s: %s", msg_id, e)
        # On error, assume unread to be safe (allows retry)
        return True


def mark_message_as_read(service, msg_id: str) -> bool:
    """
    Mark a Gmail message as read by removing the UNREAD label.
    Returns True on success, False on failure.
    """
    try:
        service.users().messages().modify(
            userId="me",
            id=msg_id,
            body={"removeLabelIds": ["UNREAD"]}
        ).execute()
        logging.info(f"✅ Marked message {msg_id} as read")
        return True
    except HttpError as e:
        logging.error(f"❌ Failed to mark message {msg_id} as read: {e}")
        return False


def get_message_meta(service, msg_id: str) -> Dict:
    try:
        # Get full message to extract body
        msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        meta = {h["name"]: h["value"] for h in headers}
        
        # Extract body
        body = extract_body(payload)
        
        return {
            "id": msg_id,
            "threadId": msg.get("threadId"),
            "headers": meta,
            "body": body
        }
    except HttpError as e:
        logging.error("Error fetching message %s: %s", msg_id, e)
        return {"id": msg_id, "headers": {}, "body": ""}


def poll_loop():
    config = load_config()
    interval = int(config.get("poll_interval_seconds", 30))
    logging.info("Starting Gmail poller (interval %ss)", interval)
    service = get_gmail_service()
    state = load_state()
    seen = set(state.get("seen_ids", []))

    try:
        while True:
            try:
                # response = service.users().messages().list(userId="me", labelIds=["INBOX"], q="is:unread", maxResults=100).execute()
                response = service.users().messages().list(
                    userId="me",
                    labelIds=["INBOX"],
                    q="is:unread newer_than:1d",   # <- only last 1 day
                    maxResults=10
                ).execute()

                messages = response.get("messages", [])
                new_ids: List[str] = []
                if messages:
                    for m in messages:
                        mid = m.get("id")
                        if not mid:
                            continue
                        
                        # Skip already-read messages (idempotent: re-running won't re-process)
                        if not is_message_unread(service, mid):
                            logging.debug(f"Skipping already-read message {mid}")
                            continue
                        
                        # Skip messages we've already seen in this session
                        if mid in seen:
                            continue
                        
                        # Process the unread message
                        meta = get_message_meta(service, mid)
                        payload = {
                            "id": mid,
                            "from": meta.get("headers", {}).get("From"),
                            "subject": meta.get("headers", {}).get("Subject"),
                            "date": meta.get("headers", {}).get("Date"),
                            "body": meta.get("body", ""),
                        }
                        
                        # Process email and attempt to send reply if it's a support ticket
                        result = notify_agent(payload, config)
                        
                        # Only mark as read after successful reply (or if not a support ticket)
                        # This ensures failed sends can be retried on next poll
                        if result.get("status") == "processed":
                            if mark_message_as_read(service, mid):
                                new_ids.append(mid)
                            else:
                                logging.warning(f"Failed to mark message {mid} as read, will retry")
                        else:
                            logging.warning(f"Message {mid} processing failed, not marking as read to allow retry")
                
                # update seen set and persist (only for successfully processed messages)
                if new_ids:
                    seen.update(new_ids)
                    state["seen_ids"] = list(seen)
                    save_state(state)
                time.sleep(interval)
            except HttpError as e:
                logging.error("Gmail API error: %s", e)
                time.sleep(interval)
            except Exception as e:
                logging.exception("Unexpected error in poll loop: %s", e)
                time.sleep(interval)
    except KeyboardInterrupt:
        logging.info("Exiting poller")


def send_email(service, to_email: str, subject: str, body_text: str):
    """Send an email through Gmail API. Raises exception on failure."""
    message = MIMEText(body_text)
    message["to"] = to_email
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    try:
        send_message = (
            service.users()
            .messages()
            .send(userId="me", body={"raw": raw})
            .execute()
        )
        logging.info(f"Email sent successfully to {to_email}. Message ID: {send_message['id']}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        raise  # Re-raise to allow caller to handle failure


if __name__ == "__main__":
    poll_loop()
