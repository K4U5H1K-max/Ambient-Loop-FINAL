# Ambient Loop — Gmail‑Driven Customer Support Agent

This project is a LangGraph-based customer support agent that reacts to Gmail messages, triages tickets with an LLM, and records state in a PostgreSQL database. It is designed for local development with LangGraph Studio and a FastAPI backend.

## Project Structure

- `agent/` – LangGraph agent implementation
   - `graph.py` – builds and exposes the `customer_support_agent` graph (`graph_app`)
   - `nodes.py` – node functions (validate, classify, policy selection, resolve, etc.)
   - `state.py` – `SupportAgentState` definition and graph state helpers
   - `tools.py` – tool functions the graph can call (ERP, stock checks, resend/refund initializers, etc.)
   - `prompts.py` – all core system prompts and instruction templates

- `api/` – HTTP API
   - `server.py` – FastAPI app for ticket operations and (optionally) Gmail push endpoints

- `config/` – LangGraph configuration
   - `langgraph.json` – maps the graph id `customer_support_agent` to `./graph.py:graph_app`
   - `graph.py` – thin wrapper that adjusts `sys.path` and re-exports `graph_app` from `agent.graph` for `langgraph dev`

- `data/` – persistence and domain data
   - `ticket_db.py` – SQLAlchemy models, session, and helpers for ticket state
   - `data.py` – in-memory product/order data used by tools and tests
   - `service.py` – ERP-style service wrapper over `data.py`
   - `policies.py` – policy definitions / rules used during resolution
   - `memory.py` – helpers for loading and persisting conversational context

- `integration/`
   - `mail_api.py` – Gmail integration (watch/history processing, unread detection, agent notification)

- `tests/`
   - `test_api.py` – API behavior
   - `test_erp_integration.py` – ERP and tool layer
   - `test_mail_read_status.py` – Gmail read/unread helpers
   - `test_messages.py` – ticket + message state interactions

- Root
   - `README.md` – this file
   - `requirements.txt` – Python dependencies

## Prerequisites

- Python 3.10+
- A PostgreSQL instance (local or remote)
- Google Cloud project with Gmail API enabled and OAuth credentials
- LangGraph CLI installed (`pip install langgraph-cli`)

## Setup

1. Create and activate a virtual environment

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Configure environment

- Copy your OpenAI and LangSmith keys into environment variables if needed (e.g. `OPENAI_API_KEY`, `LANGCHAIN_API_KEY`).
- Ensure your PostgreSQL connection URL is configured wherever `data/ticket_db.py` expects it (commonly via an environment variable or a DSN string inside that module).

4. Configure Gmail

- Create OAuth credentials in the Google Cloud Console (Desktop application).
- Download the JSON and save it as `credentials.json` in `scr/Agent` (or wherever `integration/mail_api.py` expects it in your current setup).
- On first run of the Gmail flow, a browser window will open; consent will be stored in a local token file.

> Note: This repo has evolved from a simple polling script into a modular LangGraph + FastAPI application. Some legacy references to polling may remain in comments; the active integration is via the modules listed above.

## Running the LangGraph Dev Server

From the `config` directory:

```powershell
cd config
langgraph dev
```

This starts a local dev server that:

- Registers the `customer_support_agent` graph from `config/graph.py`.
- Opens LangGraph Studio pointing at `http://127.0.0.1:2024`.

You can now create threads and runs in Studio and watch the graph execute nodes like `validate`, `tier_classification`, `query_issue_classification`, `classify`, `policy`, and `resolve`.

## Running the FastAPI Server

From the project root:

```powershell
uvicorn api.server:app --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive docs. The API can be used for ticket creation, status checks, and as a receiver for Gmail-triggered events depending on how `integration/mail_api.py` is wired.

## Tests

Run tests from the project root:

```powershell
pytest
```

This exercises the core agent state, ERP integration, mail utilities, and API wiring.

## Notes

- All new code should import from the modular packages (`agent.*`, `data.*`, `integration.*`, `api.*`) rather than the old `scr/Agent` or `database` paths.
- If you add new tools or nodes, keep prompts centralized in `agent/prompts.py` and data/logic in `data/` where appropriate.
