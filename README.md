# Ambient Loop

**Gmail-Driven Customer Support Agent with Human-in-the-Loop Approval**

---

## Table of Contents

- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Agent Lifecycle & Execution](#agent-lifecycle--execution)
- [Installation & Setup](#installation--setup)
- [Running the Project](#running-the-project)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [License](#license)

---

## Overview

Ambient Loop automates customer support ticket triage, classification, and resolution using AI while maintaining human oversight for critical actions. The system integrates directly with Gmail to process incoming support requests, intelligently routes them through multi-tier classification, and applies policy-driven resolutions with automated stock checking and smart resend/refund logic.

**Why it exists:** Manual customer support is time-consuming, inconsistent, and expensive. Support teams face repetitive queries, inconsistent policy application, delays in escalation, and lack of automated decision-making for inventory-based resolutions.

**Value proposition:** Reduce support costs and response times while ensuring consistent policy application and maintaining human control over critical financial actions (refunds, resends).

---

## Key Capabilities

- **Intelligent Ticket Validation**: Validates incoming messages as support tickets, extracts order IDs (format: ORD#####), and pre-loads product catalog context using structured LLM outputs
- **Multi-Tier Classification**: Routes tickets through L1 (automated), L2 (agent-assisted), or L3 (requires human approval) based on complexity and business impact
- **Problem Type Classification**: Identifies categories (damaged, missing, wrong item, late delivery, defective) with detailed reasoning
- **Policy-Driven Resolution**: Matches problems to company policies (e.g., "Damaged Item Policy", "Customer Satisfaction Guarantee") with fuzzy matching
- **Automated Stock Intelligence**: Checks real-time inventory levels and automatically suggests resend if in stock, refund if out of stock
- **Human-in-the-Loop Approval**: Critical actions (`initialize_refund`, `initialize_resend`) pause execution and require explicit human approval via LangGraph interrupts
- **Gmail Push Integration**: Native Gmail API integration with watch notifications and history processing for instant ticket creation
- **PostgreSQL Persistence**: Stores ticket state, messages, resolution history, and approval audit trail
- **Professional Email Generation**: Generates customer-facing email responses with order details, resolution steps, and timelines
- **FastAPI REST API**: Endpoints for ticket creation, status checks, and Gmail webhook integration with interactive Swagger docs

---

## System Architecture

Ambient Loop follows a **modular monolith architecture** with an event-driven agent workflow built on LangGraph.

### High-Level Architecture

```
┌─────────────────┐      ┌──────────────────┐
│   Gmail API     │─────>│  FastAPI Server  │
│  (Push/Watch)   │      │   (api/server)   │
└─────────────────┘      └────────┬─────────┘
                                 │
                                 ▼
                        ┌────────────────────┐
                        │  LangGraph Agent   │
                        │   (agent/graph)    │
                        └────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
   ┌──────────┐          ┌──────────────┐       ┌──────────────┐
   │  Nodes   │          │    Tools     │       │  PostgreSQL  │
   │ (validate│          │ (ERP, Stock) │       │  (Tickets)   │
   │ classify │          └──────────────┘       └──────────────┘
   │  policy  │
   │ resolve) │
   └──────────┘
         │
         ▼
   ┌────────────────┐
   │  Human Review  │
   │  (Interrupts)  │
   └────────────────┘
```

### Agent Workflow

The LangGraph state machine processes tickets through the following nodes:

1. **Validate & Load Context**: Validates message as support ticket, extracts order ID, pre-loads product catalog
2. **Tier Classification**: Analyzes complexity and assigns L1/L2/L3 tier
3. **Query/Issue Classification**: Distinguishes between support tickets and general inquiries
4. **Problem Classification**: Identifies problem types (damaged, delayed, wrong-item, etc.)
5. **Policy Selection**: Matches problems to applicable policies with reasoning
6. **Resolve Issue**: Executes tools (check stock, track order), makes resend/refund decisions, generates customer email
7. **Human Approval** (if L3 or critical action): Pauses execution for human review via LangGraph Studio

### Agent Inbox (state.messages)

The agent's message inbox is implemented as `state.messages` in `SupportAgentState`, defined in `agent/state.py`. This append-only message history serves as the complete conversation context for the agent.

**Message Types:**
- **HumanMessage**: Customer input (initial ticket description, follow-up questions)
- **AIMessage**: Agent reasoning, classifications, tool call requests, and generated responses
- **ToolMessage**: Tool execution results (e.g., stock levels, order status, tracking information)

**Message Ordering:**
- Messages are append-only and maintained in chronological order
- Messages are never reordered or deleted
- The full message history is preserved across all node transitions

**Visibility:**
- The agent sees the complete message history on every node invocation
- All messages are passed to the LLM on each node execution, providing full context
- This ensures the agent maintains awareness of previous decisions and tool results

**Persistence:**
- Messages persist across:
  - Node transitions within the workflow
  - Interrupt pauses (human approval requests)
  - Resume actions after human approval
  - Graph execution checkpoints

**Explicit Limitations:**
- The system does NOT use message prioritization
- The system does NOT deduplicate messages
- The system does NOT support agent-to-agent messaging (single agent architecture)

### Human-in-the-Loop Design

Critical actions trigger interrupts that pause the workflow:
- **L3 Tier Classification**: Requires approval before proceeding
- **Refund Requests**: Human must approve before processing
- **Resend Requests**: Human must approve before processing

If approved, workflow continues. If rejected, workflow stops gracefully and requires supervisor review.

#### Interrupt Handling Flow

**Where Interrupts Occur:**
- **L3 tier classification approval** (`tier_classifier` node in `agent/nodes.py`)
- **Critical tool approvals** in `resolve_issue` node:
  - `initialize_refund` tool execution
  - `initialize_resend` tool execution

**How Interrupts Work:**
1. `interrupt()` function is called, pausing graph execution
2. State is checkpointed before the pause (full `SupportAgentState` is saved)
3. Human reviewer responds via LangGraph Studio interface
4. Execution resumes from the exact paused node after human response

**Response Handling Logic:**
- **Accept**: Execution continues normally from the point of interruption
- **Reject or Ignore**: 
  - Graph terminates early
  - State returns with `requires_human_review = True`
  - No rollback is performed (state changes before interrupt remain)

**Explicit Limitations:**
- No automatic retries after rejection
- No state rollback on rejection (previous state changes persist)
- No interrupt timeout handling (waits indefinitely for human response)
- System/LLM failures are not yet interrupt-driven (handled via standard error paths)

### Design Decisions

- **LangGraph State Machine**: Provides visual debugging, checkpointing, and explicit control flow
- **Structured LLM Outputs**: Uses Pydantic models for type safety and validation
- **Modular Package Structure**: Separates agent logic, data persistence, API interface, and external integrations
- **Centralized Prompts**: All system prompts in `agent/prompts.py` for easy versioning and testing
- **Tool-Loop Pattern**: Supports multi-step reasoning with iterative tool calls

### Project Structure

```
Ambient-Loop-FINAL/
├── agent/                    # Core agent logic
│   ├── graph.py             # LangGraph state machine definition
│   ├── nodes.py             # Node functions (validate, classify, resolve, etc.)
│   ├── state.py             # SupportAgentState TypedDict
│   ├── tools.py             # Tool functions (check_stock, initialize_refund, etc.)
│   └── prompts.py           # Centralized system prompts
│
├── api/                      # HTTP API layer
│   └── server.py            # FastAPI application (ticket endpoints, Gmail push)
│
├── config/                   # Configuration files
│   ├── langgraph.json       # Graph registration config
│   ├── config.json          # Application configuration
│   ├── docker-compose.yml   # Docker orchestration
│   └── state.json           # Runtime state persistence
│
├── data/                     # Data layer
│   ├── ticket_db.py         # SQLAlchemy models + session management
│   ├── data.py              # In-memory product/order data (mock ERP)
│   ├── service.py           # ERP service wrapper
│   ├── policies.py           # Policy definitions and retrieval
│   ├── memory.py            # Context loading helpers (products, policies)
│   ├── models.py            # Data models (Order, Product, Shipment, etc.)
│   ├── create_db.py         # Database creation utility
│   ├── seed_policies.py     # Policy seeding script
│   └── seed_products.py     # Product seeding script
│
├── integration/              # External integrations
│   └── mail_api.py          # Gmail API (watch, history, mark read)
│
├── scripts/                  # Utility scripts
│   └── init_db.py           # Database initialization
│
├── tests/                    # Test suite
│   ├── test_api.py
│   ├── test_erp_integration.py
│   ├── test_mail_read_status.py
│   ├── test_messages.py
│   └── test_scenarios.json  # Test case definitions
│
├── .env.example
├── README.md
└── requirements.txt
```

---

## Tech Stack

### Language
- **Python 3.10+**

### Frameworks & Libraries
- **LangGraph**: Agent orchestration and state management
- **LangChain**: LLM framework and tool integration
- **FastAPI**: Asynchronous REST API
- **OpenAI GPT-4o-mini**: Large language model for reasoning
- **Pydantic**: Data validation and structured outputs
- **SQLAlchemy**: ORM for PostgreSQL
- **Google API Client**: Gmail API integration

### Database
- **PostgreSQL**: Ticket persistence, message history, approval audit trail
- **SQLAlchemy ORM**: Object-relational mapping for database operations, providing type-safe database interactions, session management, and declarative model definitions. SQLAlchemy handles connection pooling, transaction management, and query building, abstracting away raw SQL while maintaining performance and flexibility.

### Infrastructure
- **Currently**: Local development with `langgraph dev`
- **Future**: LangSmith deployment for production observability and monitoring

### Tools & Services
- **LangGraph CLI**: Development server (`langgraph dev`)
- **LangSmith**: LLM observability and tracing (optional)
- **Gmail API**: Email ingestion and processing
- **pytest**: Testing framework
- **python-dotenv**: Environment variable management

---

## How It Works

### End-to-End Flow

1. **Gmail Push Notification**
   - Customer sends email to support inbox
   - Gmail API sends push notification to FastAPI `/gmail/push` endpoint
   - FastAPI decodes notification and extracts history ID

2. **Email Processing**
   - FastAPI calls `process_gmail_history()` to fetch new messages
   - Email metadata (from, subject, body) is extracted
   - Message is enqueued for processing

3. **Agent Workflow Execution**
   - LangGraph agent receives email content as initial state
   - **Validate Node**: Confirms support ticket, extracts order ID, loads product context
   - **Tier Classification Node**: Analyzes complexity (L1/L2/L3)
   - **Query/Issue Classification Node**: Distinguishes ticket vs. inquiry
   - **Problem Classification Node**: Identifies problem types with reasoning
   - **Policy Selection Node**: Matches problems to policies
   - **Resolve Node**: Executes tools (check stock, track order), makes decisions

4. **Tool Execution**
   - `check_order_status`: Retrieves order details from ERP
   - `track_order`: Gets shipment tracking information
   - `check_stock`: Checks inventory levels
   - `initialize_resend`: Creates resend shipment (if stock available)
   - `initialize_refund`: Processes refund request (if stock unavailable)

5. **Human Approval (if needed)**
   - L3 classification or critical action triggers interrupt
   - Workflow pauses, waiting for human approval in LangGraph Studio
   - Human reviews request and approves/rejects
   - If approved: workflow continues
   - If rejected: workflow stops, requires supervisor review

6. **Email Response Generation**
   - LLM generates professional customer email with resolution details
   - Email includes order number, timeline, next steps, tracking info

7. **Response Delivery**
   - Gmail API sends reply email to customer
   - Original message marked as read
   - Ticket state saved to PostgreSQL

8. **Audit Trail**
   - All decisions, approvals, and reasoning stored in database
   - Full message history and thought process preserved

---

## Agent Lifecycle & Execution

### Tool Execution Loop

The agent uses an iterative tool execution loop within the `resolve_issue` node (defined in `agent/nodes.py`). This loop enables multi-step reasoning and decision-making.

**Decision Process:**
The agent decides whether to:
- **Call a tool**: When additional information is needed (e.g., check stock, verify order status)
- **Ask a clarification question**: When customer input is ambiguous or incomplete
- **Generate a customer-facing response**: When sufficient information is gathered to resolve the issue

**Tool Call Execution:**
- Tool calls are executed sequentially within the loop
- Each tool call is appended to `state.messages` as an AIMessage → ToolMessage pair:
  - **AIMessage**: Contains the tool call request with parameters
  - **ToolMessage**: Contains the tool execution result
- Tool results persist in `state.messages` and are available to subsequent LLM invocations
- Tools are designed to be idempotent (e.g., checking stock multiple times does not create duplicate refunds)

**Available Tools:**
- `check_order_status`: Retrieves order details from ERP system
- `track_order`: Gets shipment tracking information
- `check_stock`: Checks inventory levels for products
- `initialize_resend`: Creates resend shipment (requires human approval if triggered)
- `initialize_refund`: Processes refund request (requires human approval if triggered)

**Loop Termination:**
The tool execution loop ends when:
- A resolution is generated (customer email response is created), OR
- Human approval is rejected (workflow terminates early), OR
- The issue is escalated for review (requires supervisor intervention)

### State Management & Persistence

The agent uses LangGraph's checkpoint-based persistence to maintain state across node transitions and interrupt pauses.

**State Structure:**
The full `SupportAgentState` (defined in `agent/state.py`) is persisted between nodes and includes:
- **messages**: Complete conversation history (HumanMessage, AIMessage, ToolMessage)
- **tier_level**: Classification tier (L1, L2, or L3)
- **approved**: Human approval status for L3 or critical actions
- **problems**: List of identified problem types
- **policy_name**: Selected policy for resolution
- **policy_desc**: Policy description and reasoning
- **action_taken**: Final resolution action (e.g., "Refund issued", "Resend item")
- **reason**: Reasoning for the action taken
- **email_reply**: Generated customer-facing email response
- **reasoning**: Dictionary of reasoning steps at each node
- **thought_process**: List of agent decision-making steps
- **order_id**: Extracted order identifier
- **has_order_id**: Boolean flag for order ID presence
- **products_cache**: Pre-loaded product catalog context
- **is_support_ticket**: Validation flag
- **query_issue**: Classification (query vs. issue)

**State Persistence:**
- State is checkpointed automatically by LangGraph between node transitions
- State survives interrupt pauses (full state saved before `interrupt()` call)
- State is restored exactly when execution resumes after human approval
- State changes are cumulative (no rollback mechanism)

**Persistence Limitations:**
- Rollback is NOT implemented (state changes before interrupt remain even if rejected)
- State is not versioned (no history of state changes)
- State persistence relies on LangGraph's checkpoint system (in-memory for `langgraph dev`, configurable for production)

---

## Installation & Setup

### Prerequisites

- **Python 3.10+**
- **PostgreSQL** (local or remote instance)
- **Google Cloud Project** with Gmail API enabled
- **OAuth 2.0 Credentials** (Desktop application type)
- **LangGraph CLI**: `pip install langgraph-cli`
- **OpenAI API Key**
- **LangSmith API Key** (optional, for observability)

### Installation Steps

**1. Clone the repository**
```bash
git clone https://github.com/K4U5H1K-max/Ambient-Loop-FINAL.git
cd Ambient-Loop-FINAL
```

**2. Create and activate virtual environment**
```powershell
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up PostgreSQL**
```bash
# Create database
psql -U postgres
CREATE DATABASE ambient_loop;
\q
```

**5. Configure environment variables**

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit the `.env` file in the project root with your configuration:

```env
# Required
OPENAI_API_KEY=sk-proj-...
DATABASE_URL=postgresql://user:password@localhost:5432/ambient_loop

# Optional
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ambient-loop
```

The `.env` file is located in the project root and contains all required environment variables. See `.env.example` for a template with all available configuration options.

**6. Set up Gmail OAuth**

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create OAuth 2.0 Client ID (Application type: Desktop app)
3. Download JSON and save as `credentials.json` in project root
4. On first run, browser will open for OAuth consent

**7. Initialize database**

```bash
python scripts/init_db.py
```

This creates tables and seeds initial policies and products.

---

## Running the Project

### Option 1: Docker Compose (Recommended for Containerized Setup)

Build and run the entire stack using Docker Compose:

```bash
docker-compose -f config/docker-compose.yml up --build
```

This command:
- Builds the Docker image for the application
- Starts PostgreSQL database container
- Runs the application with all dependencies
- Automatically uses environment variables from `.env` file in the project root

To run in detached mode:
```bash
docker-compose -f config/docker-compose.yml up -d --build
```

To stop the containers:
```bash
docker-compose -f config/docker-compose.yml down
```

### Option 2: LangGraph Dev Server (Recommended for Development)

```bash
cd config
langgraph dev
```

- Opens LangGraph Studio at http://127.0.0.1:2024
- Visual graph debugging and manual run control
- Interactive thread management

### Option 3: FastAPI Server

```bash
uvicorn api.server:app --reload
```

- REST API at http://127.0.0.1:8000
- Interactive docs at http://127.0.0.1:8000/docs
- Gmail push endpoint at http://127.0.0.1:8000/gmail/push

### Option 4: Both (Separate Terminals)

```bash
# Terminal 1: LangGraph Dev
cd config
langgraph dev

# Terminal 2: FastAPI Server
uvicorn api.server:app --reload
```

This setup allows full end-to-end testing with Gmail integration.

---

## Usage

### LangGraph Studio (Development/Debugging)

1. Start LangGraph dev server: `cd config && langgraph dev`
2. Open http://127.0.0.1:2024 in browser
3. Create a new thread
4. Send message: `"I received my order #ORD12345 yesterday, but the Premium Wireless Headphones are damaged."`
5. Watch the graph execute nodes visually
6. Approve/reject critical actions when interrupted

### FastAPI REST API

**Create Ticket**
```bash
curl -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-001",
    "customer_id": "customer@example.com",
    "ticket_description": "My order #ORD12345 arrived damaged.",
    "received_date": "2025-01-12T10:00:00"
  }'
```

**Get Ticket Status**
```bash
curl http://localhost:8000/tickets/TICKET-001
```

**List All Tickets**
```bash
curl http://localhost:8000/tickets
```

### Gmail-Based Workflow (End-to-End)

1. Customer sends email to support inbox
2. Gmail push notification triggers FastAPI webhook
3. LangGraph agent processes ticket automatically
4. Human approves critical actions in LangGraph Studio (if needed)
5. Customer receives resolution email automatically

### Expected Inputs

- **Customer message**: Natural language describing issue
- **Order ID**: Format `ORD#####` (5 digits)
- **Product mentions**: Product names or IDs from catalog

### Expected Outputs

- **Classification**: Tier (L1/L2/L3), problem types
- **Policy**: Selected policy with reasoning
- **Resolution**: Action (Resend/Refund), professional email response
- **State**: Updated ticket state in PostgreSQL

---

## Configuration

### langgraph.json

Located in `config/langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "customer_support_agent": "./graph.py:graph_app"
  },
  "env": ".env"
}
```

Maps graph ID to Python module path. The `graph.py` file is a thin wrapper that adds the project root to `sys.path` and re-exports `graph_app` from `agent.graph`.

### LLM Model Configuration

Edit `agent/nodes.py` to change the LLM model:

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

Available options:
- `gpt-4o-mini` (default, cost-effective)
- `gpt-4o` (higher quality, more expensive)
- `gpt-4-turbo` (best quality, highest cost)

### Interrupt Behavior

To disable L3 approval (not recommended for production):

Edit `agent/nodes.py`, in `tier_classifier()` function:
```python
if tier_level == "L3":
    # Comment out interrupt block for auto-approval
    pass
```

### Database Configuration

Edit `data/ticket_db.py` to change database connection:

```python
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/ambient_loop")
```

### Environment Variables

The `.env` file is located in the project root and contains all required environment variables. A template file `.env.example` is provided with all available configuration options. Copy `.env.example` to `.env` and fill in your values.

See [Installation & Setup](#installation--setup) for detailed environment variable configuration.

---

## Testing

### Types of Tests

- **Unit Tests**: Tool functions, ERP service, data models
- **Integration Tests**: Mail API, database operations
- **End-to-End Tests**: Full ticket resolution workflows with test scenarios

### Running Tests

**Run all tests**
```bash
pytest
```

**Run specific test file**
```bash
pytest tests/test_erp_integration.py -v
```

**Run with coverage**
```bash
pytest --cov=agent --cov=data --cov=api
```

**Run specific scenario**
```bash
# Test scenarios defined in tests/test_scenarios.json
pytest tests/ -k "damaged_product"
```

### Testing Tools

- **pytest**: Test framework
- **unittest.mock**: Mocking Gmail API, LLM responses
- **pytest-asyncio**: Async test support

### Coverage Expectations

- Target: 80% code coverage
- Focus areas: Agent nodes, tool functions, API endpoints

---

## Deployment

### Current Status

**Development Only**: The system is currently configured for local development. Production deployment requires additional work.

### Planned Deployment Options

**Option 1: AWS ECS/Fargate**
- Container on ECS with RDS PostgreSQL
- API Gateway for FastAPI endpoints
- EventBridge for scheduled Gmail checks
- CloudWatch for monitoring

**Option 2: Azure Container Apps**
- FastAPI on Container Apps
- Azure Database for PostgreSQL
- Azure Functions for Gmail webhooks
- Application Insights for monitoring

**Option 3: Railway/Render**
- One-click deployment for MVP
- Managed PostgreSQL included
- Automatic HTTPS and domain

### Supported Environments

- **Development**: `langgraph dev` on localhost ✅
- **Staging**: Container on cloud with test database (TODO)
- **Production**: Multi-region deployment with HA PostgreSQL (TODO)

---

## License

If MIT License is chosen:

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Legal Considerations

1. **OpenAI Terms**: Ensure compliance with OpenAI API Terms of Service
2. **Gmail API**: Follow Google's API Terms (no scraping, respect rate limits)
3. **Data Privacy**: If handling EU customers, consider GDPR compliance
4. **Customer Data**: Store customer emails/orders securely (encryption at rest/transit)
5. **Third-Party Licenses**: All dependencies in `requirements.txt` have permissive licenses (MIT, Apache 2.0, BSD)

---

**Repository**: [GitHub - Ambient-Loop-FINAL](https://github.com/K4U5H1K-max/Ambient-Loop-FINAL)
