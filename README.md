# Ambient Agent — Gmail trigger (polling)

This small project polls the Gmail API for unread messages and notifies an ambient agent when a new message arrives.

Files created
- `mail_api.py` — Gmail poller and notifier
- `requirements.txt` — Python dependencies
- `config.json` — Poll interval and webhook config (sample)
- `.gitignore` — ignores local tokens and secrets

Quick start

1. Set up Google OAuth credentials
   - Go to https://console.cloud.google.com/apis/credentials
   - Create an OAuth 2.0 Client ID (Application type: Desktop app)
   - Download the JSON and save it as `credentials.json` in the project root (`AmbientAgent`)

2. Install dependencies
   - It's recommended to use a virtualenv. In PowerShell:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Configure
   - Edit `config.json`:
     - `poll_interval_seconds` — seconds between polls (default 30)
     - `notify_via_webhook` — set to `true` to POST JSON notifications
     - `webhook_url` — your ambient agent endpoint (e.g. `http://localhost:8000/notify`)

4. Run

```powershell
python main.py
```

On first run a browser window will open to authorize the Gmail account. A token will be stored in `token.pickle`.
