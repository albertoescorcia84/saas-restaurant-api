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
    Uses safe_substitute-style approach: only fills known keys,
    ignores unknown {placeholders} that may exist in the raw prompt.
    Falls back to a minimal default if the prompt is empty or broken.
    """
    cleaned = _LEGACY_RE.sub("", raw).strip()

    # Replace only the known placeholders — don't crash on unknown ones
    try:
        cleaned = cleaned.replace("{restaurant_name}", brand)
        cleaned = cleaned.replace("{menu_context}", "our menu")
    except Exception:
        pass

    if not cleaned or len(cleaned) < 10:
        # Minimal safe fallback if DB prompt is empty or completely stripped
        cleaned = f"You are a helpful ordering assistant for {brand}."

    return cleaned


def build_system_prompt(session: Session, brand: str, raw_prompt: str) -> str:
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
            greeting_instruction = (
                f"The person messaging is a returning customer named {c.full_name}. "
                f"Start by greeting them by name."
            )
        else:
            greeting_instruction = "Start with a friendly greeting. Do not ask for the customer's name."

        return (
            f"{base}\n\n"
            f"{greeting_instruction} "
            f"You are a friendly restaurant assistant helping the customer order food. "
            f"Only talk about the menu — do not ask for delivery address, name, or email yet. "
            f"If the customer asks about dishes, prices, or ingredients, "
            f"use the query_vector_database tool to look it up — never guess. "
            f"Once the customer decides what they want, confirm it back naturally, for example: "
            f"'Perfect! So that's 1x Roast Chicken — shall I go ahead with that?' "
            f"Wait for the customer to say yes before moving forward. "
            f"Keep every reply short — one or two sentences maximum."
        )

    if session.status == State.CHECKOUT:
        order = c.order_summary

        if session.checkout_field == CheckoutField.ADDRESS:
            task = (
                f"The customer just ordered: {order}. "
                f"Your next message should warmly ask for their delivery address — "
                f"nothing else. One natural sentence, like you would say it on the phone."
            )
        elif session.checkout_field == CheckoutField.NAME:
            task = (
                f"You are collecting delivery info for the order: {order}. "
                f"You already have the address. "
                f"Ask the customer for their full name in a natural, friendly way. "
                f"One sentence only — do not ask for anything else."
            )
        elif session.checkout_field == CheckoutField.EMAIL:
            task = (
                f"You are almost done collecting info for the order: {order}. "
                f"You have the name and address. "
                f"Politely ask for their email. Make clear it's optional and they can skip it. "
                f"One sentence only."
            )
        else:
            task = "Politely ask the customer for any remaining missing information."

        return f"{base}\n\n{task}"

    if session.status == State.CONFIRM:
        email_part = f", email {c.email}" if c.email else ""
        return (
            f"{base}\n\n"
            f"You are about to confirm an order. Tell the customer in a warm, human tone:\n"
            f"- Their name: {c.full_name}\n"
            f"- Delivery address: {c.address}\n"
            f"- Order: {c.order_summary}{email_part}\n\n"
            f"After reading those back, ask something like: "
            f"'Does everything look right?' "
            f"Write it as natural speech — no technical labels, no bullet points in your reply. "
            f"Do not save or call any tool yet."
        )

    if session.status == State.DONE:
        return (
            f"{base}\n\n"
            f"The order is confirmed and saved. Send a short closing message to {c.full_name} "
            f"confirming that {c.order_summary} will be delivered to {c.address}. "
            f"You can mention reference ID {c.customer_id} if you like. "
            f"Two sentences max. Be warm and genuine — sound like a real person, not a robot."
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
    Replace this stub with your actual vector similarity search.
    Example using pgvector:
        embedding = embed(query)
        rows = conn.execute("SELECT item, price FROM menu_vectors
                             WHERE tenant_id=:tid
                             ORDER BY embedding <-> :emb LIMIT 5", ...)
        return format_results(rows)
    """
    logger.info(f"[vector_db] tenant={tenant_id} query='{query}'")
    # ── STUB — replace below ──────────────────────────────────────────────────
    return (
        "MENU RESULTS:\n"
        "- Roast Chicken: $20.00 (gluten-free, served with rice)\n"
        "- Yuca Frita: $4.00\n"
        "- Garden Salad: $3.00 (vegan)\n"
        "- Grilled Salmon: $24.00\n"
        "Source: menu vector index."
    )


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

def strip_hallucinations(text: str) -> str:
    return _HALLUC_RE.sub("", text).strip()


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


def _detect_order_confirmation(
    llm_reply: str,
    customer_message: str,
    conversation: list,
) -> tuple[bool, str]:
    """
    Determine whether the order has been confirmed by the customer.

    Two-phase detection:
    ─────────────────────────────────────────────────────────────────
    PHASE A — LLM asked a confirmation question AND customer said YES.
      The LLM reply contains a confirmation question pattern
      (e.g. "just to confirm, you'd like X, right?") AND the current
      customer message is affirmative. We extract the order summary
      from the LLM's confirmation question.

    PHASE B — Customer message itself explicitly confirms with items.
      e.g. "yes, I want 1 roast chicken" or "confirm: roast chicken"
      We extract the items from the customer's own message.

    Returns (confirmed: bool, order_summary: str).
    order_summary is empty string when confirmed=False.
    """
    reply_lower   = llm_reply.lower()
    customer_lower = customer_message.strip().lower()

    # ── Phase A: LLM asked a confirm question, customer replied YES ────────────
    _CONFIRM_TRIGGERS = (
        "just to confirm",
        "to confirm your order",
        "confirm —",
        "confirming your order",
        "so you'd like",
        "you'd like",
        "you want",
        "your order is",
        "is that right",
        "is that correct",
        "shall i proceed",
        "ready to place",
        "place your order",
    )
    llm_asked_confirmation = any(t in reply_lower for t in _CONFIRM_TRIGGERS)

    if llm_asked_confirmation and _is_affirmative(customer_message):
        # Extract the order summary from the LLM's confirmation sentence.
        # Look for the last assistant message that contained a confirmation question.
        order_summary = _extract_order_from_text(llm_reply)
        if order_summary:
            return True, order_summary

        # Fallback: search the last few assistant messages for one with items
        for msg in reversed(conversation):
            if msg.get("role") == "assistant" and msg.get("content"):
                summary = _extract_order_from_text(msg["content"])
                if summary:
                    return True, summary

    # ── Phase B: Customer message itself contains items + confirmation ─────────
    _CUSTOMER_ORDER_PREFIXES = (
        "yes,", "yes ", "si,", "sí,",
        "confirm:", "order:", "i want", "i'd like",
        "quiero", "me das", "ponme",
    )
    if any(customer_lower.startswith(p) for p in _CUSTOMER_ORDER_PREFIXES):
        summary = _extract_order_from_text(customer_message)
        if summary:
            return True, summary

    return False, ""


# Menu item keywords used by _extract_order_from_text.
# Extend this list to match your real menu items from the vector DB.
_MENU_ITEM_RE = re.compile(
    r"(\d+\s*x?\s*)?"                          # optional qty: "2x" or "2 "
    r"("
    r"roast chicken|chicken|pollo"
    r"|yuca|yuca frita"
    r"|salad|ensalada"
    r"|salmon|salmón"
    r"|[a-z ]{3,30}"                             # fallback: any 3-30 char word sequence
    r")",
    re.IGNORECASE,
)

_ITEM_STOP_WORDS = {
    "the", "a", "an", "your", "our", "their", "its", "this", "that",
    "my", "your", "right", "correct", "order", "like", "want", "would",
    "you", "i", "is", "are", "was", "to", "for", "of", "with", "and",
    "just", "confirm", "confirming", "confirmed", "please", "okay", "ok",
    "so", "as", "in", "at", "on", "up", "no", "not", "yes", "si",
}


def _extract_order_from_text(text: str) -> str:
    """
    Extract a cleaned order summary from a sentence.
    Looks for quantity + item patterns like "1x Roast Chicken, 2x Yuca".
    Returns empty string if nothing recognisable is found.
    """
    # Prefer explicit "Nx item" patterns
    qty_item = re.findall(r"\d+\s*x?\s+[A-Za-z][a-z ]{2,25}", text)
    if qty_item:
        return ", ".join(q.strip() for q in qty_item)

    # Fallback: extract capitalised / known food nouns after a colon or "like"
    after_colon = re.search(r"(?:like|order is|you'd like|you want|confirm[^:]*:)\s*(.+?)(?:\.|,|\?|$)",
                            text, re.IGNORECASE)
    if after_colon:
        candidate = after_colon.group(1).strip()
        words = [w for w in candidate.split() if w.lower() not in _ITEM_STOP_WORDS]
        if words:
            return " ".join(words)

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

def _refresh_system_prompt(session: Session, brand: str, raw_prompt: str) -> None:
    """Replace (or insert) the system message at index 0 of the message list."""
    sp = {"role": "system", "content": build_system_prompt(session, brand, raw_prompt)}
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

    # ── 2. Resolve / init session ─────────────────────────────────────────────
    session = _sessions.get(user_phone)

    if not session or session.status == State.DONE:
        # Fresh session — query the DB for customer context
        cust = db_get_customer(user_phone, tenant_id)

        collected = CustomerData(
            full_name     = cust["full_name"],
            email         = cust["email"],
            address       = cust["address_line_1"],   # pre-fill if known
            customer_id   = cust["customer_id"],
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

    # ── 3. Append user message + refresh system prompt ────────────────────────
    _refresh_system_prompt(session, brand, raw_prompt)
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

        # LLM generates the next question (or the confirmation summary)
        _refresh_system_prompt(session, brand, raw_prompt)
        msg = call_llm(client, tenant["model_name"], session.messages)
        final_reply = msg.content or ""

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
            _refresh_system_prompt(session, brand, raw_prompt)
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
            _refresh_system_prompt(session, brand, raw_prompt)
            msg = call_llm(client, tenant["model_name"], session.messages)
            final_reply = msg.content or "No problem! What would you like to order?"

    # ── STATE: ORDER ──────────────────────────────────────────────────────────
    # Design decision: order_ready is NOT a tool anymore.
    # LLaMA kept printing "<function=order_ready...>" as plain text instead of
    # emitting a proper tool call. Server now owns the confirmation detection:
    #   1. LLM chats freely, only uses query_vector_database for menu lookups.
    #   2. Server inspects each LLM reply for a confirmation pattern.
    #   3. If found → server extracts the order summary and advances FSM.
    #   4. If not   → reply goes to customer as-is.
    elif session.status == State.ORDER:
        for iteration in range(MAX_TOOL_ITERS):
            msg = call_llm(
                client, tenant["model_name"], session.messages,
                tools=TOOLS_ORDER,
            )

            # ── Tool call (only query_vector_database expected here) ───────
            if msg.tool_calls:
                session.messages.append(msg)

                for tc in msg.tool_calls:
                    f_name = tc.function.name
                    try:
                        f_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        f_args = {}
                        logger.warning(f"[tool] Bad JSON args for {f_name}")

                    if f_name == "query_vector_database":
                        tool_result = query_vector_database(f_args.get("query", ""), tenant_id)
                    else:
                        tool_result = f"ERROR: tool '{f_name}' is not available at this stage."
                        logger.warning(f"[tool] Unexpected tool in ORDER state: {f_name}")

                    session.messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "name":         f_name,
                        "content":      tool_result,
                    })
                continue  # loop back so LLM generates a text reply after tool result

            # ── Plain text reply ───────────────────────────────────────────
            raw_reply   = msg.content or ""
            final_reply = strip_hallucinations(raw_reply)

            if not final_reply or len(final_reply) < 8:
                final_reply = "What would you like to order?"
                break

            # ── Server-side order confirmation detection ───────────────────
            # We look for the LLM asking "just to confirm — you'd like X, right?"
            # or the customer explicitly confirming in this same message.
            # Strategy: if the reply contains a confirmation question AND the
            # last customer message is affirmative, extract the order and advance.
            order_confirmed, order_summary = _detect_order_confirmation(
                llm_reply=final_reply,
                customer_message=request.message,
                conversation=session.messages,
            )

            if order_confirmed and order_summary:
                session.collected.order_summary = order_summary
                session.checkout_field = _determine_first_checkout_field(session)
                session.status = (
                    State.CONFIRM if session.checkout_field == CheckoutField.DONE
                    else State.CHECKOUT
                )
                logger.info(f"[order_confirmed] summary='{order_summary}' → {session.status.value}")

                # Strip the confirmation question from the reply — the LLM will
                # generate the first checkout question fresh with the new prompt.
                _refresh_system_prompt(session, brand, raw_prompt)
                session.messages.append({"role": "assistant", "content": final_reply})
                msg2 = call_llm(client, tenant["model_name"], session.messages)
                final_reply = msg2.content or ""
                # Prevent double-append below
                session.messages.append({"role": "assistant", "content": final_reply})
                return _response(session, final_reply)

            break  # plain reply, no confirmation → send to customer

        else:
            final_reply = "I'm having trouble. Could you please describe your order again?"
            logger.error("[tool_loop] Exhausted MAX_TOOL_ITERS in STATE_ORDER")

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
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)