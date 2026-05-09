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
from routers.notifications import router as notifications_router

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
app.include_router(notifications_router)

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
            f"You are a warm, human restaurant assistant taking a phone order. "
            f"Speak naturally and conversationally — short sentences, friendly tone. "
            f"When the customer says what they want, repeat it back clearly with the price "
            f"and ask if they also want a side dish. "
            f"If they decline the side, acknowledge it warmly and read back their complete order "
            f"with the total, then ask them to confirm. "
            f"Once they say yes or confirm, tell them great and that you will now get their delivery details. "
            f"Do not ask for name, address, or email at this stage."
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
            f"You are now collecting delivery information by phone. "
            f"{extra}"
            f"Ask the customer for their {next_ask} in a natural, conversational way. "
            f"One sentence only. Sound like a real person on the phone."
        )

    if session.status == State.CONFIRM:
        email_line = f"Email: {c.email}. " if c.email else ""
        return (
            f"{base}\n\n"
            f"You are confirming an order over the phone. Read these details back naturally:\n"
            f"Name: {c.full_name}\n"
            f"Address: {c.address}\n"
            f"Order: {c.order_summary}\n"
            f"{email_line}\n"
            f"Sound warm and human. Ask if everything is correct. "
            f"Do not call any tool or save anything yet."
        )

    if session.status == State.DONE:
        return (
            f"{base}\n\n"
            f"The order is placed. Thank {c.full_name} warmly. "
            f"Confirm {c.order_summary} will be delivered to {c.address}. "
            f"Give them reference number {c.customer_id}. "
            f"Two sentences max. Sound like a real person wrapping up a phone call."
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
    Schema facts:
      - customers.id          : uuid, no default → must pass gen_random_uuid()
      - tenant_customers.id   : uuid, default gen_random_uuid() → can omit
      - tenant_customers col  : tenant_specific_status (not 'status'), default 'Active'
      - tenant_customer_addresses: no unique(tenant_customer_id, is_default) →
        use UPDATE then INSERT pattern instead of ON CONFLICT
    """
    with engine.begin() as conn:

        # ── 1. Upsert global customer ─────────────────────────────────────────
        row = conn.execute(text("""
            INSERT INTO customers (id, phone_number, full_name, email)
            VALUES (gen_random_uuid(), :ph, :name, :em)
            ON CONFLICT (phone_number) DO UPDATE
                SET full_name = EXCLUDED.full_name,
                    email     = COALESCE(EXCLUDED.email, customers.email)
            RETURNING id
        """), {"ph": user_phone, "name": full_name, "em": email or None}).fetchone()
        customer_id = row[0]
        logger.info(f"[db] customer upserted id={customer_id}")

        # ── 2. Ensure tenant ↔ customer link ──────────────────────────────────
        # Column is tenant_specific_status with default 'Active' (capital A)
        tc_row = conn.execute(text("""
            INSERT INTO tenant_customers (tenant_id, customer_id, tenant_specific_status)
            VALUES (:tid, :cid, 'Active')
            ON CONFLICT (tenant_id, customer_id) DO NOTHING
            RETURNING id
        """), {"tid": tenant_id, "cid": customer_id}).fetchone()

        if not tc_row:
            tc_row = conn.execute(text("""
                SELECT id FROM tenant_customers
                WHERE tenant_id = :tid AND customer_id = :cid
            """), {"tid": tenant_id, "cid": customer_id}).fetchone()
        tc_id = tc_row[0]
        logger.info(f"[db] tenant_customer id={tc_id}")

        # ── 3. Address — no unique constraint on (tenant_customer_id, is_default)
        # Pattern: clear old default → insert new one
        conn.execute(text("""
            UPDATE tenant_customer_addresses
               SET is_default = false
             WHERE tenant_customer_id = :tcid
        """), {"tcid": tc_id})

        conn.execute(text("""
            INSERT INTO tenant_customer_addresses
                (id, tenant_customer_id, address_line_1, is_default, city, state)
            VALUES (gen_random_uuid(), :tcid, :addr, true, 'Toronto', 'ON')
        """), {"tcid": tc_id, "addr": address})
        logger.info(f"[db] address saved for tc_id={tc_id}")

    logger.info(f"[db_save_customer] Done — customer_id={customer_id}")
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


# ── Order confirmation detection ─────────────────────────────────────────────
# Strategy: the SERVER detects when the customer has confirmed.
# We do NOT rely on the LLM emitting a tag — that was unreliable.
#
# Detection works in two passes:
#   Pass A: LLM reply contains a summary + customer says YES → extract from LLM reply
#   Pass B: No tag needed — if customer is affirmative AND the conversation
#           has a pending order summary in the last assistant message, use that

_ORDER_CONFIRMED_RE = re.compile(r"ORDER_CONFIRMED:\s*([^\n.]+)", re.IGNORECASE)

# Phrases the LLM uses when it has summarized the order and is asking to confirm
_LLM_SUMMARY_TRIGGERS = (
    "just the ", "so that", "your order", "i'll confirm", "confirming",
    "shall i place", "ready to place", "to confirm", "so you",
)

def _extract_order_summary_from_llm(text: str) -> str:
    """
    Extract what the LLM said the order was.
    Looks for sentences containing price markers or known item patterns.
    """
    # Try ORDER_CONFIRMED tag first
    m = _ORDER_CONFIRMED_RE.search(text)
    if m:
        return m.group(1).strip()

    # Find sentence with $ price — most likely the order summary sentence
    for sentence in re.split(r"[.!?]", text):
        if "$" in sentence and len(sentence.strip()) > 5:
            # Clean it up
            clean = re.sub(r"(so |just |that's |I'll confirm |your order is )", "", sentence, flags=re.IGNORECASE)
            return clean.strip().strip(",").strip()

    return ""


def _detect_order_confirmation(llm_reply: str, customer_msg: str = "", conversation: list = []) -> tuple[bool, str]:
    """
    Returns (confirmed, order_summary).

    Confirmed when:
      - LLM reply contains ORDER_CONFIRMED: tag, OR
      - Customer message is affirmative AND the LLM reply summarizes the order
        (contains a price or trigger phrase), OR
      - Customer message is affirmative AND last assistant message had a summary
    """
    # Pass A: tag in LLM reply
    m = _ORDER_CONFIRMED_RE.search(llm_reply)
    if m:
        summary = m.group(1).strip()
        if summary and len(summary) > 3:
            return True, summary

    # Pass B: customer said yes + LLM reply summarizes the order
    if customer_msg and _is_affirmative(customer_msg):
        summary = _extract_order_summary_from_llm(llm_reply)
        if summary:
            return True, summary

        # Pass C: look at the previous assistant message for a summary
        for msg in reversed(conversation):
            if msg.get("role") == "assistant" and msg.get("content"):
                prev_summary = _extract_order_summary_from_llm(msg["content"])
                if prev_summary:
                    return True, prev_summary
                # Stop after checking the most recent assistant message
                break

    return False, ""


def _extract_order_from_text(text: str) -> str:
    """Legacy — kept for compatibility."""
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

        # If we just reached CONFIRM — generate summary directly, no LLM
        if session.status == State.CONFIRM:
            c = session.collected
            email_line = f", email {c.email}" if c.email else ""
            final_reply = (
                f"Perfect! Let me confirm your order: {c.order_summary}, "
                f"delivering to {c.address} for {c.full_name}{email_line}. "
                f"Does everything look correct?"
            )
        else:
            # Still in CHECKOUT — hardcoded questions, no LLM (prevents improvisation)
            _QUESTIONS = {
                CheckoutField.ADDRESS: "Got it! What's your delivery address?",
                CheckoutField.NAME:    "Perfect! And your full name for the order?",
                CheckoutField.EMAIL:   "Almost done! What's your email address? Feel free to skip if you prefer.",
            }
            final_reply = _QUESTIONS.get(
                session.checkout_field,
                "Could you provide the remaining delivery information?"
            )

    # ── STATE: CONFIRM (explicit YES / NO detection) ──────────────────────────
    # The server generates the confirmation message directly — no LLM.
    # This prevents the LLM from generating a premature closing message.
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
                import traceback
                tb = traceback.format_exc()
                logger.error(f"[save] DB error: {db_err}\nTraceback:\n{tb}")
                logger.error(f"[save] Data attempted: phone={user_phone} tenant={tenant_id} "
                             f"name={c.full_name!r} address={c.address!r} email={c.email!r}")
                session.status = State.CONFIRM
                # Return the actual error in debug so we can see it
                final_reply = (
                    "I'm sorry, there was a technical issue saving your order. "
                    "Please reply YES again to retry."
                )
                session.messages.append({"role": "assistant", "content": final_reply})
                resp = _response(session, final_reply)
                resp["_db_error"] = str(db_err)  # visible in response for debugging
                return resp

            # Hardcoded closing — never use LLM here to prevent premature completion
            final_reply = (
                f"You're all set, {c.full_name}! Your order of {c.order_summary} "
                f"will be delivered to {c.address}. "
                f"Your reference number is {customer_id}. Thank you and enjoy your meal!"
            )

        else:
            # Customer said NO — restart order, keep address if we had one
            session.status         = State.ORDER
            session.checkout_field = CheckoutField.ADDRESS
            session.collected.order_summary = ""
            final_reply = "No problem at all! What would you like to order?"

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

            # Detect BEFORE stripping — pass customer message and history for Pass B/C
            order_confirmed, order_summary = _detect_order_confirmation(
                llm_reply=raw_reply,
                customer_msg=request.message,
                conversation=session.messages,
            )

            # Now strip internal tags from the customer-facing reply
            final_reply = strip_hallucinations(raw_reply)
            if not final_reply.strip():
                final_reply = "What would you like to order?"

            if order_confirmed and order_summary:
                # Clean order_summary — extract only "Item $price" patterns
                import re as _re
                # Try to find "Nx Item $price" or "Item $price" patterns first
                clean_summary = order_summary.strip()
                # Strip common filler phrases the LLM prepends
                _FILLERS = (
                    "your total comes out to be ", "you've ordered ", "you ordered ",
                    "you're good with the ", "you're good with ",
                    "so just the ", "so just ", "that's ", "just the ", "just ",
                    "i'll confirm ", "your order is ", "so you want ", "you want ",
                    "you'd like ", "i have ", "we have ",
                )
                lower = clean_summary.lower()
                for filler in _FILLERS:
                    if lower.startswith(filler):
                        clean_summary = clean_summary[len(filler):]
                        lower = clean_summary.lower()
                        break
                # Strip trailing filler
                for suffix in (", no side", ", no sides", " for delivery", " for you"):
                    if clean_summary.lower().endswith(suffix):
                        clean_summary = clean_summary[:-len(suffix)]
                        break
                session.collected.order_summary = clean_summary.strip().rstrip(".,")
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