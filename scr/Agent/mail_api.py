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
    "https://www.googleapis.com/auth/gmail.send"
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

        # Basic logging
        print("\n=== NEW EMAIL PROCESSED ===")
        print(f"From: {payload.get('from')}")
        print(f"Subject: {payload.get('subject')}")
        print(f"Support Ticket: {final_state.get('is_support_ticket', False)}")
        print(f"Problems: {final_state.get('problems', [])}")
        print(f"Policy: {final_state.get('policy_name', 'N/A')}")
        print(f"Action: {final_state.get('action_taken', 'N/A')}")

        # Get the last AI response (if any)
        messages = final_state.get("messages", [])
        ai_response = getattr(messages[-1], "content", None) if messages else None

        if ai_response:
            print(f"\n--- AI Response ---\n{ai_response}\n")

            # Send auto-reply only if it's a support ticket
            if final_state.get("is_support_ticket"):
                reply_to = payload.get("from") or "anikakarampuri.test@gmail.com"
                subject = f"Re: {payload.get('subject', 'Support Response')}"
                send_email(get_gmail_service(), reply_to, subject, ai_response)
                logging.info(f"✅ Support email sent to {reply_to}")
            else:
                logging.info("ℹ️ Not a support ticket — no reply sent")

        return {"status": "processed", "is_support": final_state.get("is_support_ticket", False)}

    except Exception as e:
        logging.error(f"❌ notify_agent failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

   

def get_message_meta(service, msg_id: str) -> Dict:
    try:
        msg = service.users().messages().get(userId="me", id=msg_id, format="metadata", metadataHeaders=["From", "Subject", "Date"]).execute()
        headers = msg.get("payload", {}).get("headers", [])
        meta = {h["name"]: h["value"] for h in headers}
        return {"id": msg_id, "threadId": msg.get("threadId"), "headers": meta}
    except HttpError as e:
        logging.error("Error fetching message %s: %s", msg_id, e)
        return {"id": msg_id, "headers": {}}


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
                        if mid and mid not in seen:
                            meta = get_message_meta(service, mid)
                            payload = {
                                "id": mid,
                                "from": meta.get("headers", {}).get("From"),
                                "subject": meta.get("headers", {}).get("Subject"),
                                "date": meta.get("headers", {}).get("Date"),
                            }
                            notify_agent(payload, config)
                            new_ids.append(mid)
                # update seen set and persist
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
    """Send an email through Gmail API."""
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


if __name__ == "__main__":
    poll_loop()
