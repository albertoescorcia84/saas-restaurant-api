"""
SaaS Restaurant Multi-Tenant Chat API
======================================
Architecture: FastAPI + Groq (LLaMA) + PostgreSQL + In-Memory FSM

Conversation flow (enforced server-side, NOT by the LLM):
─────────────────────────────────────────────────────────
  STATE_ORDER   → LLM takes the order using query_vector_database
       ↓           (transition triggered by tool call: order_ready)
  STATE_CHECKOUT → LLM collects missing customer data field-by-field
       ↓           (server captures each field, LLM just asks one question)
  STATE_CONFIRM  → LLM presents full summary, waits for YES / NO
       ↓           (YES detected server-side, not by LLM)
  STATE_SAVING   → Server calls manage_customer_data directly (no LLM guessing)
       ↓           (success guaranteed, then LLM generates closing message)
  STATE_DONE     → Session marked completed
"""

import os
import re
import uuid
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("restaurant_api")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,     # Recover from dropped DB connections
    pool_size=10,
    max_overflow=20,
)

app = FastAPI(title="SaaS Restaurant Multi-Tenant API", version="3.0.0")


# ─────────────────────────────────────────────────────────────────────────────
# LLM Constants
# ─────────────────────────────────────────────────────────────────────────────
TEMPERATURE    = 0.1    # Near-deterministic; LLaMA follows instructions reliably
MAX_TOKENS     = 300    # Enough for a confirmation message, not an essay
SEED           = 42     # Reproducibility
MAX_TOOL_ITERS = 5      # Safety cap on the agentic tool loop


# ─────────────────────────────────────────────────────────────────────────────
# FSM — States
# ─────────────────────────────────────────────────────────────────────────────
class State(str, Enum):
    ORDER    = "taking_order"       # LLM takes food order (RAG)
    CHECKOUT = "collecting_data"    # Server collects name / address / email
    CONFIRM  = "awaiting_confirm"   # Customer reviews summary → YES / NO
    SAVING   = "saving"             # Server persists data (bypasses LLM)
    DONE     = "completed"          # Session closed


# Which checkout field to collect next
class CheckoutField(str, Enum):
    ADDRESS = "address"
    NAME    = "full_name"
    EMAIL   = "email"
    DONE    = "done"


# ─────────────────────────────────────────────────────────────────────────────
# Session data model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CustomerData:
    full_name:     str = ""
    address:       str = ""
    email:         str = ""
    order_summary: str = ""
    customer_id:   Optional[str] = None   # UUID from DB after save


@dataclass
class Session:
    session_id:      str
    tenant_id:       str
    status:          State              = State.ORDER
    checkout_field:  CheckoutField      = CheckoutField.ADDRESS
    collected:       CustomerData       = field(default_factory=CustomerData)
    messages:        list               = field(default_factory=list)
    # Customer profile flags
    is_global_customer:  bool = False   # exists in `customers` table
    is_tenant_customer:  bool = False   # exists in `tenant_customers` for this tenant
    had_address:         bool = False   # had a default address on file


# In-memory store  key = from_number (customer phone)
_sessions: dict[str, Session] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    to_number:   str   # Restaurant's WhatsApp number  → tenant lookup
    from_number: str   # Customer's WhatsApp number   → session key
    message:     str


# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1 tools: ordering
# NOTE: order_ready is intentionally REMOVED — the server detects order confirmation
# from the conversation context. LLaMA was printing it as plain text instead of
# calling it as a tool, so we moved that responsibility server-side.
TOOLS_ORDER = [
    {
        "type": "function",
        "function": {
            "name": "query_vector_database",
            "description": (
                "Search the restaurant menu. "
                "Call for any question about dishes, prices, or ingredients."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User's question about the menu."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Phase 2 tools: saving (LLM receives this ONLY as a structured instruction;
# the actual SQL execution is always done server-side as a safety guarantee)
TOOLS_SAVE = [
    {
        "type": "function",
        "function": {
            "name": "manage_customer_data",
            "description": (
                "Persist the confirmed customer data and order. "
                "Call ONLY once all fields are confirmed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name":     {"type": "string", "description": "Customer full name."},
                    "address":       {"type": "string", "description": "Delivery address."},
                    "email":         {"type": "string", "description": "Email (optional)."},
                    "order_summary": {"type": "string", "description": "Confirmed order items."}
                },
                "required": ["full_name", "address", "order_summary"]
            }
        }
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt engineering
# ─────────────────────────────────────────────────────────────────────────────

# Strip any leftover directives from DB-stored prompts (legacy compatibility)
_LEGACY_RE = re.compile(
    r"\[ORDER_FINALIZED\]"
    r"|manage_customer_data\b"
    r"|query_vector_database\b"
    r"|STRICT PROTOCOL[\s\S]*",
    re.IGNORECASE,
)

def _clean_base_prompt(raw: str, brand: str) -> str:
    """
    Sanitize and format the DB-stored base prompt.

    Removes legacy directives, fills {restaurant_name}, and strips
    {menu_context} entirely (menu data comes from query_vector_database,
    not from an inline placeholder).

    Falls back to a safe minimal prompt if the DB value is empty or broken.
    """
    cleaned = _LEGACY_RE.sub("", raw).strip()

    try:
        cleaned = cleaned.replace("{restaurant_name}", brand)
        cleaned = cleaned.replace("{menu_context}", "")
        cleaned = re.sub(r"Menu Context:[^\n]*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\d+\.\s*(CRITICAL|ORDER FLOW|STRICT)[^\n]*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
    except Exception:
        pass

    # Hard minimum — if after all cleaning we have less than 20 chars, use fallback
    if not cleaned or len(cleaned) < 20:
        cleaned = f"You are a professional ordering assistant for {brand}."

    return cleaned


def build_system_prompt(session: Session, brand: str, raw_prompt: str, menu_text: str = "") -> str:
    """
    Build a conversational, state-specific system prompt.

    Golden rule: never use uppercase headers like "CURRENT TASK —" or numbered
    lists in the instructions — LLaMA 3 leaks them verbatim into the reply.
    Write instructions as if privately coaching a human support agent.
    """
    base = _clean_base_prompt(raw_prompt, brand)
    c    = session.collected

    if session.status == State.ORDER:
        if session.is_global_customer and c.full_name:
            customer_ctx = f"You are chatting with {c.full_name}, a returning customer. Greet them by name."
        else:
            customer_ctx = "You are chatting with a new customer."

        menu_section = f"\n\nMENU:\n{menu_text}" if menu_text else ""

        return (
            f"{base}"
            f"{menu_section}\n\n"
            f"{customer_ctx} "
            f"You are a friendly and natural restaurant assistant — speak like a real person, not a robot. "
            f"Help the customer order from the menu above. "
            f"If the customer greets you, greet them back warmly and ask what they would like. "
            f"If they name a dish, confirm it and suggest a side dish from the menu if they have not picked one. "
            f"If they decline the side, accept it gracefully and confirm the final order. "
            f"Keep replies short — two sentences maximum. "
            f"Do not ask for name, address, or email. "
            f"Once the customer confirms everything they want, write ORDER_CONFIRMED: followed by "
            f"a short summary of all confirmed items, then say you will collect their delivery info. "
            f"The ORDER_CONFIRMED: tag is required — the system uses it to move to the next step."
        )

    if session.status == State.CHECKOUT:
        o = c.order_summary

        if session.checkout_field == CheckoutField.ADDRESS:
            next_ask = "delivery address"
            extra    = ""
        elif session.checkout_field == CheckoutField.NAME:
            next_ask = "full name"
            extra    = "You already have their address. "
        elif session.checkout_field == CheckoutField.EMAIL:
            next_ask = "email address"
            extra    = "Let them know it is optional. "
        else:
            next_ask = "any remaining delivery information"
            extra    = ""

        return (
            f"{base}\n\n"
            f"The food order is confirmed: {o}. "
            f"You are now collecting delivery information. "
            f"{extra}"
            f"Your only job in this message is to ask the customer for their {next_ask}. "
            f"Do not add commentary, repeat the order, or ask for anything else. "
            f"Keep it to one short, friendly sentence."
        )

    if session.status == State.CONFIRM:
        email_line = f"Email: {c.email}. " if c.email else ""
        return (
            f"{base}\n\n"
            f"Read back the order details to the customer and ask them to confirm. "
            f"The details are: "
            f"Name: {c.full_name}. "
            f"Delivery address: {c.address}. "
            f"{email_line}"
            f"Order: {c.order_summary}. "
            f"Speak naturally, as if on a phone call. "
            f"Ask whether the details are correct and whether they want to proceed. "
            f"Do not call any tool. Do not save anything yet."
        )

    if session.status == State.DONE:
        return (
            f"{base}\n\n"
            f"The order has been placed successfully. "
            f"Write a short, warm closing message to {c.full_name}. "
            f"Confirm their order of {c.order_summary} will be delivered to {c.address}. "
            f"Reference number: {c.customer_id}. "
            f"Keep it to two sentences. Be genuine and friendly."
        )

    return base  # fallback




# ─────────────────────────────────────────────────────────────────────────────
# Database operations
# ─────────────────────────────────────────────────────────────────────────────

def db_get_tenant(to_number: str) -> Optional[dict]:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                t.id           AS tenant_id,
                t.brand_name,
                t.status,
                s.system_prompt,
                m.model_name,
                m.api_key
            FROM tenants t
            JOIN tenant_ai_settings s ON t.id = s.tenant_id
            JOIN llm_models m          ON s.model_id = m.id
            WHERE t.phone_number = :ph
        """), {"ph": to_number}).mappings().first()
    return dict(row) if row else None


def db_get_customer(from_number: str, tenant_id: str) -> dict:
    """
    Returns a dict with:
      full_name, email, address_line_1,
      is_global (bool), is_tenant (bool)
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                c.id               AS customer_id,
                c.full_name,
                c.email,
                tc.id              AS tc_id,
                a.address_line_1
            FROM customers c
            LEFT JOIN tenant_customers tc
                ON c.id = tc.customer_id AND tc.tenant_id = :tid
            LEFT JOIN tenant_customer_addresses a
                ON tc.id = a.tenant_customer_id AND a.is_default = true
            WHERE c.phone_number = :ph
        """), {"tid": tenant_id, "ph": from_number}).mappings().first()

    if not row:
        return {"is_global": False, "is_tenant": False,
                "full_name": "", "email": "", "address_line_1": "", "customer_id": None}

    return {
        "is_global":     True,
        "is_tenant":     row["tc_id"] is not None,
        "full_name":     row["full_name"] or "",
        "email":         row["email"] or "",
        "address_line_1": row["address_line_1"] or "",
        "customer_id":   str(row["customer_id"]),
    }


def db_save_customer(
    user_phone: str,
    tenant_id:  str,
    full_name:  str,
    address:    str,
    email:      Optional[str],
    order_summary: str,
) -> str:
    """
    Upsert customer → tenant link → address.
    Returns the customer UUID as string.
    Raises on DB error (caller handles).
    """
    with engine.begin() as conn:
        # 1. Upsert global customer
        row = conn.execute(text("""
            INSERT INTO customers (phone_number, full_name, email)
            VALUES (:ph, :name, :em)
            ON CONFLICT (phone_number) DO UPDATE
                SET full_name = EXCLUDED.full_name,
                    email     = COALESCE(EXCLUDED.email, customers.email)
            RETURNING id
        """), {"ph": user_phone, "name": full_name, "em": email or None}).fetchone()
        customer_id = row[0]

        # 2. Ensure tenant ↔ customer relationship
        tc_row = conn.execute(text("""
            INSERT INTO tenant_customers (tenant_id, customer_id)
            VALUES (:tid, :cid)
            ON CONFLICT (tenant_id, customer_id) DO NOTHING
            RETURNING id
        """), {"tid": tenant_id, "cid": customer_id}).fetchone()

        if not tc_row:
            tc_row = conn.execute(text("""
                SELECT id FROM tenant_customers
                WHERE tenant_id = :tid AND customer_id = :cid
            """), {"tid": tenant_id, "cid": customer_id}).fetchone()
        tc_id = tc_row[0]

        # 3. Upsert default delivery address
        conn.execute(text("""
            INSERT INTO tenant_customer_addresses
                (tenant_customer_id, address_line_1, is_default, city, state, country)
            VALUES (:tcid, :addr, true, 'Toronto', 'ON', 'Canada')
            ON CONFLICT (tenant_customer_id, is_default) DO UPDATE
                SET address_line_1 = EXCLUDED.address_line_1
        """), {"tcid": tc_id, "addr": address})

        # 4. Insert order record (uncomment when orders table exists)
        # conn.execute(text("""
        #     INSERT INTO orders (tenant_customer_id, summary, status, created_at)
        #     VALUES (:tcid, :summary, 'pending', NOW())
        # """), {"tcid": tc_id, "summary": order_summary})

    logger.info(f"[db_save_customer] Saved customer_id={customer_id}")
    return str(customer_id)


# ─────────────────────────────────────────────────────────────────────────────
# Vector DB stub  (replace with your real implementation)
# ─────────────────────────────────────────────────────────────────────────────

def query_vector_database(query: str, tenant_id: str) -> str:
    """
    Fetch menu content from menu_vectors table for the given tenant.
    Since embeddings are NULL, we do a simple full-text fetch of all
    content rows for this tenant and return them directly.
    """
    logger.info(f"[vector_db] tenant={tenant_id} query='{query}'")
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT content FROM menu_vectors
                WHERE tenant_id = :tid
                ORDER BY id
            """), {"tid": tenant_id}).fetchall()

        if not rows:
            logger.warning(f"[vector_db] No menu found for tenant_id={tenant_id}")
            return "Menu information is not available."

        return "\n".join(r[0] for r in rows)

    except Exception as e:
        logger.error(f"[vector_db] DB error: {e}")
        return "Menu information could not be retrieved."


def get_full_menu(tenant_id: str) -> str:
    """Fetch the complete menu text for a tenant — used to inject into system prompt."""
    return query_vector_database("full menu", tenant_id)


# ─────────────────────────────────────────────────────────────────────────────
# LLM call wrapper
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    client:     Groq,
    model:      str,
    messages:   list,
    tools:      Optional[list] = None,
    force_tool: Optional[str]  = None,
):
    """Single LLM call. Returns the raw message object from Groq."""
    kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEED,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = (
            {"type": "function", "function": {"name": force_tool}}
            if force_tool else "auto"
        )
    return client.chat.completions.create(**kwargs).choices[0].message


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination guard
# ─────────────────────────────────────────────────────────────────────────────

_HALLUC_RE = re.compile(
    r"\[ORDER_FINALIZED\]"
    r"|manage_customer_data\s*[>\({].*"
    r"|<function[_\s].*"
    r"|order_ready\s*[>\({].*",
    re.IGNORECASE | re.DOTALL,
)

_ORDER_TAG_RE = re.compile(r"ORDER_CONFIRMED:[^.\n]*", re.IGNORECASE)

def strip_hallucinations(text: str) -> str:
    cleaned = _HALLUC_RE.sub("", text)
    cleaned = _ORDER_TAG_RE.sub("", cleaned)
    return cleaned.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Input parsers — extract clean values from natural-language customer replies
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_SKIP_EMAIL = {"no", "skip", "none", "n/a", "-", "sin email", "no tengo",
               "no email", "no tengo email", "omitir", "saltar"}

def _extract_email(raw: str) -> str:
    """
    Extract a valid email from a free-text reply like:
      "yes, it is romel123@gmail.com"   → "romel123@gmail.com"
      "my email is foo@bar.com"         → "foo@bar.com"
      "no" / "skip"                     → ""  (treated as skipped)
    """
    stripped = raw.strip()
    if stripped.lower() in _SKIP_EMAIL:
        return ""
    match = _EMAIL_RE.search(stripped)
    return match.group(0) if match else ""


# ── Order confirmed signal ────────────────────────────────────────────────────
# The LLM is instructed to include "ORDER_CONFIRMED:" followed by the items
# when the customer says yes. The server detects this exact tag — no fuzzy matching.
_ORDER_CONFIRMED_RE = re.compile(r"ORDER_CONFIRMED:\s*(.+?)(?:\.|$)", re.IGNORECASE)


def _detect_order_confirmation(llm_reply: str) -> tuple[bool, str]:
    """
    Detect ORDER_CONFIRMED:<items> tag in the LLM reply.
    Must be called on the RAW reply BEFORE strip_hallucinations runs.
    """
    match = _ORDER_CONFIRMED_RE.search(llm_reply)
    if match:
        summary = match.group(1).strip()
        if summary and len(summary) > 2:
            return True, summary
    return False, ""


def _extract_order_from_text(text: str) -> str:
    """Legacy helper — kept for potential future use."""
    qty_item = re.findall(r"\d+\s*x?\s+[A-Za-z][a-z ]{2,25}", text)
    if qty_item:
        return ", ".join(q.strip() for q in qty_item)
    return ""


def _is_affirmative(raw: str) -> bool:
    """
    Detect a YES from natural language — handles:
      "yes", "si", "correct", "that's right", "yep, go ahead", etc.
    Returns False for anything that doesn't clearly signal agreement.
    """
    lowered = raw.strip().lower()
    # Exact single-word matches
    _YES_WORDS = {"yes", "si", "sí", "yep", "yeah", "correct", "ok", "okay",
                  "sure", "confirm", "confirmed", "adelante", "procede",
                  "dale", "claro", "yup", "affirmative", "perfecto", "listo"}
    if lowered in _YES_WORDS:
        return True
    # Phrase-level: starts with or contains a YES word
    _YES_PHRASES = ("yes,", "yes.", "yes!", "si,", "si.", "sí,",
                    "that's correct", "that is correct", "looks good",
                    "all good", "go ahead", "proceed", "everything is correct",
                    "todo bien", "todo correcto", "está bien", "esta bien")
    for phrase in _YES_PHRASES:
        if phrase in lowered:
            return True
    return False


def _extract_name(raw: str) -> str:
    """
    Strip common prefixes from name replies like:
      "my name is John Doe"  → "John Doe"
      "I'm Maria Lopez"      → "Maria Lopez"
      "John Doe"             → "John Doe"
    """
    stripped = raw.strip()
    for prefix in ("my name is ", "i am ", "i'm ", "soy ", "me llamo ",
                   "mi nombre es ", "it's ", "its "):
        lower = stripped.lower()
        if lower.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return stripped.strip().title()


# ─────────────────────────────────────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────────────────────────────────────

def _refresh_system_prompt(session: Session, brand: str, raw_prompt: str, menu_text: str = "") -> None:
    """Replace (or insert) the system message at index 0 of the message list."""
    sp = {"role": "system", "content": build_system_prompt(session, brand, raw_prompt, menu_text)}
    if session.messages and session.messages[0]["role"] == "system":
        session.messages[0] = sp
    else:
        session.messages.insert(0, sp)


def _determine_first_checkout_field(session: Session) -> CheckoutField:
    """
    Decide which field to ask for first based on what we already know.
    Order of collection: address → name → email
    """
    c = session.collected
    # Always ask for address first (even returning customers may want a new one)
    if not c.address:
        return CheckoutField.ADDRESS
    if not c.full_name:
        return CheckoutField.NAME
    if not c.email:
        return CheckoutField.EMAIL
    return CheckoutField.DONE


def _next_checkout_field(session: Session) -> CheckoutField:
    """Advance to the next missing field after the current one was collected."""
    c = session.collected
    current = session.checkout_field

    if current == CheckoutField.ADDRESS:
        return CheckoutField.NAME if not c.full_name else CheckoutField.EMAIL
    if current == CheckoutField.NAME:
        return CheckoutField.EMAIL
    # EMAIL is always last
    return CheckoutField.DONE


# ─────────────────────────────────────────────────────────────────────────────
# Main endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_phone = request.from_number

    # ── 1. Resolve tenant ─────────────────────────────────────────────────────
    tenant = db_get_tenant(request.to_number)
    if not tenant:
        raise HTTPException(404, "Restaurant not found.")
    if tenant["status"] != "Active":
        raise HTTPException(403, "Restaurant is currently inactive.")

    tenant_id  = tenant["tenant_id"]
    brand      = tenant["brand_name"]
    raw_prompt = tenant["system_prompt"]
    client     = Groq(api_key=tenant["api_key"])

    # Fetch menu once per request — injected into every system prompt
    menu_text = get_full_menu(tenant_id)
    logger.info(f"[menu] tenant={tenant_id} chars={len(menu_text)}")

    # ── 2. Resolve / init session ─────────────────────────────────────────────
    session = _sessions.get(user_phone)

    # Force a fresh session if:
    #  - no session exists
    #  - previous session is completed
    #  - session belongs to a different tenant (customer moved to another restaurant)
    _GREETINGS = {"hello","hi","hola","hey","buenos dias","buenas","good morning",
                  "good afternoon","good evening","start","restart","nuevo","nueva"}
    is_greeting = request.message.strip().lower() in _GREETINGS

    need_new_session = (
        not session
        or session.status == State.DONE
        or session.tenant_id != tenant_id
        or is_greeting   # always start fresh on a greeting
    )

    if need_new_session:
        cust = db_get_customer(user_phone, tenant_id)

        collected = CustomerData(
            full_name   = cust["full_name"],
            email       = cust["email"],
            address     = cust["address_line_1"],
            customer_id = cust["customer_id"],
        )

        session = Session(
            session_id         = str(uuid.uuid4()),
            tenant_id          = tenant_id,
            status             = State.ORDER,
            collected          = collected,
            is_global_customer = cust["is_global"],
            is_tenant_customer = cust["is_tenant"],
            had_address        = bool(cust["address_line_1"]),
        )
        _sessions[user_phone] = session
        logger.info(f"[session] New session {session.session_id} for {user_phone}")

    # ── 3. Refresh system prompt and append user message ─────────────────────
    _refresh_system_prompt(session, brand, raw_prompt, menu_text)

    # Trim history to last 20 messages (10 exchanges) to prevent role drift.
    # Always keep index 0 (system prompt).
    if len(session.messages) > 21:
        session.messages = [session.messages[0]] + session.messages[-20:]

    session.messages.append({"role": "user", "content": request.message})
    final_reply = ""

    # ═══════════════════════════════════════════════════════════════════════════
    # FSM dispatcher
    # ═══════════════════════════════════════════════════════════════════════════

    # ── STATE: CHECKOUT (field-by-field collection) ───────────────────────────
    if session.status == State.CHECKOUT:
        user_input = request.message.strip()
        field      = session.checkout_field

        # Capture the current field's value
        if field == CheckoutField.ADDRESS:
            session.collected.address = user_input

        elif field == CheckoutField.NAME:
            session.collected.full_name = _extract_name(user_input)

        elif field == CheckoutField.EMAIL:
            session.collected.email = _extract_email(user_input)

        # Advance to next missing field (or go to CONFIRM)
        next_field = _next_checkout_field(session)

        if next_field == CheckoutField.DONE:
            session.status = State.CONFIRM
        else:
            session.checkout_field = next_field

        # LLM generates the next question — with hardcoded fallback per field
        _refresh_system_prompt(session, brand, raw_prompt, menu_text)
        msg = call_llm(client, tenant["model_name"], session.messages)
        final_reply = (msg.content or "").strip()

        # Fallback: if LLM returns empty, use a simple direct question
        if not final_reply:
            _FALLBACKS = {
                CheckoutField.ADDRESS: "What is your delivery address?",
                CheckoutField.NAME:    "What is your full name?",
                CheckoutField.EMAIL:   "What is your email address? (optional — you can skip this)",
            }
            # Use the CURRENT state's field (after advancing)
            current_field = (
                session.checkout_field
                if session.status == State.CHECKOUT
                else CheckoutField.ADDRESS
            )
            final_reply = _FALLBACKS.get(current_field, "Could you provide the missing information?")

            if session.status == State.CONFIRM:
                c = session.collected
                final_reply = (
                    f"Just to confirm your order: {c.order_summary}, "
                    f"delivering to {c.address} for {c.full_name}. "
                    f"Is everything correct?"
                )

    # ── STATE: CONFIRM (explicit YES / NO detection) ──────────────────────────
    elif session.status == State.CONFIRM:
        confirmed = _is_affirmative(request.message)

        if confirmed:
            # ── Go straight to DB save — do NOT trust the LLM to call the tool
            # ── The LLM is only used to generate the closing message afterward
            session.status = State.SAVING
            c = session.collected

            try:
                customer_id = db_save_customer(
                    user_phone    = user_phone,
                    tenant_id     = tenant_id,
                    full_name     = c.full_name,
                    address       = c.address,
                    email         = c.email or None,
                    order_summary = c.order_summary,
                )
                session.collected.customer_id = customer_id
                session.status = State.DONE
                logger.info(f"[save] customer_id={customer_id} order='{c.order_summary}'")

            except Exception as db_err:
                logger.error(f"[save] DB error: {db_err}")
                # Don't crash the conversation — tell the customer and stay in CONFIRM
                session.status = State.CONFIRM
                final_reply = (
                    "I'm sorry, there was a technical issue saving your order. "
                    "Please reply YES again to retry."
                )
                session.messages.append({"role": "assistant", "content": final_reply})
                return _response(session, final_reply)

            # Generate closing message with LLM
            _refresh_system_prompt(session, brand, raw_prompt, menu_text)
            # Add a tool result message so the LLM understands the save succeeded
            session.messages.append({
                "role":    "assistant",
                "content": None,
                "tool_calls": [{
                    "id":   "direct_save_001",
                    "type": "function",
                    "function": {
                        "name":      "manage_customer_data",
                        "arguments": json.dumps({
                            "full_name":     c.full_name,
                            "address":       c.address,
                            "email":         c.email,
                            "order_summary": c.order_summary,
                        })
                    }
                }]
            })
            session.messages.append({
                "role":         "tool",
                "tool_call_id": "direct_save_001",
                "name":         "manage_customer_data",
                "content":      f"SUCCESS. customer_id={customer_id}. Order saved.",
            })

            msg = call_llm(client, tenant["model_name"], session.messages)
            final_reply = msg.content or (
                f"Thank you {c.full_name}! Your order ({c.order_summary}) "
                f"has been confirmed and will be delivered to {c.address}. "
                f"Your reference ID is {customer_id}. Enjoy your meal!"
            )

        else:
            # Customer said NO — restart from order taking
            session.status         = State.ORDER
            session.checkout_field = CheckoutField.ADDRESS
            session.collected.order_summary = ""
            session.collected.address       = session.collected.address  # keep known address
            _refresh_system_prompt(session, brand, raw_prompt, menu_text)
            msg = call_llm(client, tenant["model_name"], session.messages)
            final_reply = msg.content or "No problem! What would you like to order?"

    # ── STATE: ORDER ──────────────────────────────────────────────────────────
    elif session.status == State.ORDER:
        for iteration in range(MAX_TOOL_ITERS):
            msg = call_llm(
                client, tenant["model_name"], session.messages,
                tools=TOOLS_ORDER,
            )

            # ── Tool call → execute and loop back for text reply ──────────
            if msg.tool_calls:
                session.messages.append(msg)
                for tc in msg.tool_calls:
                    try:
                        f_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        f_args = {}
                    if tc.function.name == "query_vector_database":
                        tool_result = query_vector_database(f_args.get("query", ""), tenant_id)
                    else:
                        tool_result = "Tool not available at this stage."
                    session.messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "name": tc.function.name, "content": tool_result,
                    })
                continue

            # ── Plain text reply ───────────────────────────────────────────
            raw_reply = msg.content or ""

            # IMPORTANT: detect ORDER_CONFIRMED BEFORE stripping it
            order_confirmed, order_summary = _detect_order_confirmation(raw_reply)

            # Now strip internal tags from the customer-facing reply
            final_reply = strip_hallucinations(raw_reply)
            if not final_reply.strip():
                final_reply = "What would you like to order?"

            if order_confirmed and order_summary:
                session.collected.order_summary = order_summary
                session.checkout_field = _determine_first_checkout_field(session)
                session.status = (
                    State.CONFIRM if session.checkout_field == CheckoutField.DONE
                    else State.CHECKOUT
                )
                logger.info(f"[order] confirmed='{order_summary}' next={session.status.value}")
                _refresh_system_prompt(session, brand, raw_prompt, menu_text)
                session.messages.append({"role": "assistant", "content": final_reply})
                msg2 = call_llm(client, tenant["model_name"], session.messages)
                final_reply = (msg2.content or "").strip()

                # Hardcoded fallback — LLM sometimes returns empty on state transition
                if not final_reply:
                    _FIELD_Q = {
                        CheckoutField.ADDRESS: "What is your delivery address?",
                        CheckoutField.NAME:    "What is your full name?",
                        CheckoutField.EMAIL:   "What is your email? (optional, you can skip)",
                    }
                    final_reply = _FIELD_Q.get(
                        session.checkout_field,
                        "Could you please provide your delivery address?"
                    )

                session.messages.append({"role": "assistant", "content": final_reply})
                return _response(session, final_reply)

            break

        else:
            final_reply = "Sorry, I'm having trouble. What would you like to order?"
            logger.error("[order_loop] Exhausted MAX_TOOL_ITERS")

    # ── STATE: DONE (shouldn't normally receive messages, but handle gracefully)
    else:
        final_reply = (
            f"Your order has already been placed, {session.collected.full_name}! "
            "Is there anything else I can help you with?"
        )

    # ── Store assistant reply ─────────────────────────────────────────────────
    session.messages.append({"role": "assistant", "content": final_reply})

    return _response(session, final_reply)


# ─────────────────────────────────────────────────────────────────────────────
# Response helper
# ─────────────────────────────────────────────────────────────────────────────

def _response(session: Session, reply: str) -> dict:
    return {
        "reply":      reply,
        "session_id": session.session_id,
        "status":     session.status.value,
        # ── Remove `debug` block before production ───────────────────────────
        "debug": {
            "checkout_field": session.checkout_field.value,
            "collected":      {
                "full_name":     session.collected.full_name,
                "address":       session.collected.address,
                "email":         session.collected.email,
                "order_summary": session.collected.order_summary,
                "customer_id":   session.collected.customer_id,
            },
            "is_global_customer": session.is_global_customer,
            "is_tenant_customer": session.is_tenant_customer,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Debug endpoint — REMOVE IN PRODUCTION
# Call this to inspect what the DB is returning for a given restaurant number
# GET /debug/tenant?to_number=+14165550001
# ─────────────────────────────────────────────────────────────────────────────
from fastapi.responses import JSONResponse

@app.delete("/session/{from_number}")
async def reset_session(from_number: str):
    """Force-clear a session for a given customer phone number."""
    phone = from_number.replace("-", "+")  # allow URL-safe format
    if phone in _sessions:
        del _sessions[phone]
        return {"cleared": True, "from_number": phone}
    return {"cleared": False, "from_number": phone, "reason": "no active session"}


@app.get("/session/{from_number}")
async def get_session(from_number: str):
    """Inspect the current session state for a customer. Debug only."""
    phone = from_number.replace("-", "+")
    session = _sessions.get(phone)
    if not session:
        return {"session": None}
    return {
        "session_id":   session.session_id,
        "status":       session.status.value,
        "checkout_field": session.checkout_field.value,
        "collected":    {
            "full_name":     session.collected.full_name,
            "address":       session.collected.address,
            "email":         session.collected.email,
            "order_summary": session.collected.order_summary,
        },
        "message_count": len(session.messages),
        "last_messages": [
            {"role": m["role"], "content": (m.get("content") or "")[:120]}
            for m in session.messages[-6:]
        ],
    }


@app.get("/debug/tenant")
async def debug_tenant(to_number: str):
    tenant = db_get_tenant(to_number)
    if not tenant:
        return JSONResponse({"error": "not found"}, status_code=404)
    raw_prompt = tenant["system_prompt"]
    cleaned    = _clean_base_prompt(raw_prompt, tenant["brand_name"])
    return {
        "brand_name":    tenant["brand_name"],
        "model_name":    tenant["model_name"],
        "status":        tenant["status"],
        "raw_prompt":    raw_prompt,
        "cleaned_prompt": cleaned,
        "prompt_length": len(cleaned),
        "looks_ok":      len(cleaned) > 20 and "?" not in cleaned[:50],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)