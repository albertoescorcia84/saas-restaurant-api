"""
SaaS Restaurant Multi-Tenant Chat API — v4.12
==============================================
Changes from v4.11:
  - ARCHITECTURE: Separate system-level classifier provider.
    * New env vars: CLASSIFIER_PROVIDER, CLASSIFIER_API_KEY, CLASSIFIER_MODEL.
    * All classifiers (_extract_order_action, _classify_intent,
      _classify_confirmation, _resolve_timezone_from_address) now use the
      system classifier LLM (Groq recommended for speed & cost).
    * Tenant's own LLM is reserved for conversational replies (where the
      brand voice matters). Falls back to tenant LLM if env vars not set.
  - PARSER ROBUSTNESS: _extract_order_action now retries once when the LLM
    returns an empty markdown fence (``` followed by nothing). This was the
    root cause of the "carne asada never added but bot lied" bug.
  - ANTI-HALLUCINATION: Order-state system prompt now hard-states that the
    cart is the source of truth and the LLM must never claim it added,
    modified, or removed items that don't appear in the CURRENT CART block.
  - CUSTOMER DEFENSE: db_get_customer now ignores phone-like full_name
    values (10+ digits) coming from legacy corrupted records. The chat
    flow will treat such customers as needing name collection.
  - MAPBOX UNIT PRESERVATION: Address validation now extracts unit/apt/suite
    numbers from the input before geocoding and re-attaches them to the
    canonical result so the courier knows which unit to deliver to.
"""

import os
import re
import uuid
import json
import math
import logging
import httpx
import zoneinfo as zi
from datetime import datetime, time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from routers.notifications import router as notifications_router
from providers.factory import get_provider
from providers.base    import LLMProvider

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
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")

# ── System-level classifier provider (optional; falls back to tenant LLM) ──
# Set these env vars to dedicate a small/fast model to classification tasks.
# Recommended: Groq Llama 3.1 8B for ~10x speed and ~60x cost reduction.
CLASSIFIER_PROVIDER = os.getenv("CLASSIFIER_PROVIDER", "").strip().lower()
CLASSIFIER_API_KEY  = os.getenv("CLASSIFIER_API_KEY", "").strip()
CLASSIFIER_MODEL    = os.getenv("CLASSIFIER_MODEL", "").strip()

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
app    = FastAPI(title="SaaS Restaurant Multi-Tenant API", version="4.12.0")
app.include_router(notifications_router)

# Singleton instance of the system classifier LLM (None if not configured)
_CLASSIFIER_LLM: Optional[LLMProvider] = None


def _init_classifier_llm() -> None:
    """Initialize the system-wide classifier LLM if env vars are set."""
    global _CLASSIFIER_LLM
    if CLASSIFIER_PROVIDER and CLASSIFIER_API_KEY and CLASSIFIER_MODEL:
        try:
            _CLASSIFIER_LLM = get_provider(
                CLASSIFIER_PROVIDER, CLASSIFIER_API_KEY, CLASSIFIER_MODEL
            )
            logger.info(
                f"[classifier_llm] initialized: provider={CLASSIFIER_PROVIDER} "
                f"model={CLASSIFIER_MODEL}"
            )
        except Exception as e:
            logger.error(f"[classifier_llm] init failed, falling back to tenant LLM: {e}")
            _CLASSIFIER_LLM = None
    else:
        logger.info(
            "[classifier_llm] not configured (CLASSIFIER_PROVIDER/API_KEY/MODEL not set); "
            "using tenant LLM for classification"
        )


def _classifier_or_tenant(tenant_llm: LLMProvider) -> LLMProvider:
    """
    Returns the system classifier LLM if configured, otherwise the tenant's.
    This lets every tenant get the speed/cost benefit of a small classifier
    while still using their chosen model for conversational replies.
    """
    return _CLASSIFIER_LLM if _CLASSIFIER_LLM is not None else tenant_llm

TEMPERATURE       = 0.1
MAX_TOKENS        = 400
SEED              = 42
CLASSIFIER_TOKENS = 400
CLASSIFIER_TEMP   = 0.0
MAPBOX_THRESHOLD  = 0.5

_DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
_DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# Cache of geocoded tenant proximity coords: tenant_id -> "lng,lat"
_TENANT_PROXIMITY: dict[str, str] = {}

# Initialize the system-wide classifier LLM at import time (if configured)
_init_classifier_llm()


# ─────────────────────────────────────────────────────────────────────────────
# FSM States
# ─────────────────────────────────────────────────────────────────────────────
class State(str, Enum):
    ORDER          = "taking_order"
    SERVICE_SELECT = "selecting_service"
    CHECKOUT       = "collecting_data"
    FINAL_CONFIRM  = "final_confirmation"
    SAVING         = "saving"
    DONE           = "completed"


class CheckoutField(str, Enum):
    ADDRESS = "address"
    NAME    = "full_name"
    EMAIL   = "email"
    DONE    = "done"


class ServiceType(str, Enum):
    PICKUP   = "pickup"
    DELIVERY = "delivery"
    DINE_IN  = "dine_in"


# ─────────────────────────────────────────────────────────────────────────────
# Order Cart
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderItem:
    name:              str
    quantity:          int
    unit_price:        float
    prep_notes:        str   = ""
    item_code:         str   = ""
    name_plural:       str   = ""        # Optional plural form for display
    is_tax_exempt:     bool  = False
    tax_rate_override: Optional[float] = None   # None → use tenant default

    @property
    def line_total(self) -> float:
        return round(self.quantity * self.unit_price, 2)

    def effective_tax_rate(self, tenant_default_rate: float) -> float:
        """Returns the tax rate that actually applies to this line."""
        if self.is_tax_exempt:
            return 0.0
        if self.tax_rate_override is not None:
            return float(self.tax_rate_override)
        return float(tenant_default_rate or 0.0)

    def display_name(self, lang_plural: str = "") -> str:
        """
        Pick the right form for the customer-facing display.
        - For quantity == 1: use singular `name`.
        - For quantity > 1: use language-aware plural if available, else
          fall back to base name (no language-specific guessing).
        """
        if self.quantity <= 1:
            return self.name
        if lang_plural:
            return lang_plural
        if self.name_plural:
            return self.name_plural
        return self.name

    def to_display(self) -> str:
        note = f" ({self.prep_notes})" if self.prep_notes else ""
        qty  = f"{self.quantity}x " if self.quantity > 1 else ""
        return f"{qty}{self.display_name()}{note} — ${self.line_total:.2f}"

    def to_dict(self) -> dict:
        return {
            "item_code":         self.item_code,
            "name":              self.name,
            "name_plural":       self.name_plural,
            "quantity":          self.quantity,
            "unit_price":        self.unit_price,
            "prep_notes":        self.prep_notes,
            "line_total":        self.line_total,
            "is_tax_exempt":     self.is_tax_exempt,
            "tax_rate_override": self.tax_rate_override,
        }


@dataclass
class OrderCart:
    items:               list[OrderItem] = field(default_factory=list)
    tenant_default_tax:  float           = 0.0
    delivery_fee:        float           = 0.0

    @property
    def subtotal(self) -> float:
        """Sum of line totals (pre-tax)."""
        return round(sum(i.line_total for i in self.items), 2)

    @property
    def tax_total(self) -> float:
        """Tax computed per-item using each item's effective rate."""
        total = 0.0
        for i in self.items:
            rate = i.effective_tax_rate(self.tenant_default_tax)
            if rate > 0:
                total += i.line_total * rate
        return round(total, 2)

    @property
    def grand_total(self) -> float:
        return round(self.subtotal + self.tax_total + self.delivery_fee, 2)

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    def add_item(self, name: str, quantity: int, unit_price: float,
                 prep_notes: str = "", item_code: str = "",
                 name_plural: str = "", is_tax_exempt: bool = False,
                 tax_rate_override: Optional[float] = None) -> None:
        for item in self.items:
            if item.name.lower() == name.lower():
                item.quantity += quantity
                if prep_notes:
                    item.prep_notes = prep_notes
                if item_code:
                    item.item_code = item_code
                logger.info(f"[cart] updated {name} qty={item.quantity}")
                return
        self.items.append(OrderItem(
            name              = name,
            quantity          = quantity,
            unit_price        = unit_price,
            prep_notes        = prep_notes,
            item_code         = item_code,
            name_plural       = name_plural,
            is_tax_exempt     = is_tax_exempt,
            tax_rate_override = tax_rate_override,
        ))
        logger.info(
            f"[cart] added {name} x{quantity} @ ${unit_price} code={item_code} "
            f"tax_exempt={is_tax_exempt} override={tax_rate_override}"
        )

    def remove_item(self, name: str) -> bool:
        before = len(self.items)
        self.items = [i for i in self.items if name.lower() not in i.name.lower()]
        removed = len(self.items) < before
        logger.info(f"[cart] remove '{name}' — removed={removed}")
        return removed

    def update_prep_notes(self, name: str, notes: str) -> bool:
        for item in self.items:
            if name.lower() in item.name.lower():
                item.prep_notes = notes
                return True
        return False

    def update_quantity(self, name: str, quantity: int) -> bool:
        for item in self.items:
            if name.lower() in item.name.lower():
                item.quantity = quantity
                return True
        return False

    def to_summary_string(self) -> str:
        return " | ".join(
            f"{i.name} (${i.unit_price:.2f})" + (f" [{i.prep_notes}]" if i.prep_notes else "")
            for i in self.items
        )

    def to_display(self, lang: str = "en") -> str:
        """
        Customer-facing cart display during ORDER state.
        Shows items + subtotal ONLY. Tax is intentionally NOT shown here —
        it's revealed at FINAL_CONFIRM.
        Delivery fee shown only if > 0 (it's known after service selection).
        """
        lines = [i.to_display() for i in self.items]
        lines.append(f"\nSubtotal: ${self.subtotal:.2f}")
        if self.delivery_fee > 0:
            lines.append(f"Delivery: ${self.delivery_fee:.2f}")
        return "\n".join(lines)

    def to_list(self) -> list[dict]:
        return [i.to_dict() for i in self.items]


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ServiceInfo:
    service_type:       str
    is_active:          bool
    fee_type:           str
    fee_amount:         float
    is_open_now:        bool  = False
    hours_by_day:       dict  = field(default_factory=dict)
    delivery_radius_km: Optional[float] = None   # only meaningful for delivery


@dataclass
class TenantContext:
    tenant_id:           str
    brand_name:          str
    status:              str
    system_prompt:       str
    model_name:          str
    api_key:             str
    timezone:            str
    physical_address:    str
    city:                str
    state:               str
    country:             str
    provider:            str   = "groq"
    primary_language:    str   = "en"
    supported_languages: list[str] = field(default_factory=lambda: ["en"])
    services:            dict[str, ServiceInfo] = field(default_factory=dict)
    menu_text:           str   = ""
    menu_categories:     list[str] = field(default_factory=list)
    default_tax_rate:    float = 0.0   # e.g. 0.13 for Ontario HST


@dataclass
class CustomerData:
    full_name:         str = ""
    address:           str = ""
    address_validated: bool = False
    email:             str = ""
    service_type:      str = ""
    delivery_fee:      float = 0.0
    order_number:      str = ""
    customer_id:       Optional[str] = None


@dataclass
class Session:
    session_id:         str
    tenant_id:          str
    tenant:             TenantContext
    cart:               OrderCart     = field(default_factory=OrderCart)
    status:             State         = State.ORDER
    checkout_field:     CheckoutField = CheckoutField.ADDRESS
    collected:          CustomerData  = field(default_factory=CustomerData)
    messages:           list          = field(default_factory=list)
    language:           str           = "en"
    channel:            str           = "chat"   # "chat" | "voice"
    is_global_customer: bool          = False
    is_tenant_customer: bool          = False
    had_address:        bool          = False
    address_attempts:   int           = 0


_sessions: dict[str, Session] = {}


class ChatRequest(BaseModel):
    to_number:   str
    from_number: str
    message:     str
    channel:     str = "chat"   # "chat" | "voice"


# ─────────────────────────────────────────────────────────────────────────────
# Channel-aware text formatting
# ─────────────────────────────────────────────────────────────────────────────

# Matches most emoji ranges + decorative symbols. Conservative — keeps text/punctuation.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # symbols & pictographs
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"   # misc symbols (☀ etc.)
    "\U00002700-\U000027BF"   # dingbats
    "\u2705\u2611\u2714"      # check marks
    "]+",
    flags=re.UNICODE,
)

def _voice_safe(text: str, channel: str) -> str:
    """
    If channel is 'voice', strip emojis, asterisks, markdown bullets/headers,
    and collapse extra whitespace so TTS reads cleanly.
    On 'chat', returns text unchanged.
    """
    if channel != "voice":
        return text
    # Strip emojis
    out = _EMOJI_RE.sub("", text)
    # Strip markdown markers
    out = re.sub(r"\*+", "", out)            # *bold* → bold
    out = re.sub(r"^#+\s*", "", out, flags=re.MULTILINE)   # ## headers
    out = re.sub(r"^\s*[•\-]\s*", "", out, flags=re.MULTILINE)  # bullets → plain lines
    # Collapse multiple blank lines and trim
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]+", " ", out)
    return out.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Database — Tenant
# ─────────────────────────────────────────────────────────────────────────────

def db_get_tenant_context(to_number: str) -> Optional[TenantContext]:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                t.id, t.brand_name, t.status,
                t.physical_address, t.city, t.state, t.country, t.timezone,
                COALESCE(t.default_tax_rate, 0) AS default_tax_rate,
                s.system_prompt, s.primary_language, s.supported_languages,
                m.model_name, m.api_key, m.provider
            FROM tenants t
            JOIN tenant_ai_settings s ON t.id = s.tenant_id
            JOIN llm_models m          ON s.model_id = m.id
            WHERE t.phone_number = :ph
        """), {"ph": to_number}).mappings().first()
        if not row:
            return None

        svc_rows = conn.execute(text("""
            SELECT service_type, is_active, fee_type, fee_amount,
                   delivery_radius_km,
                   monday_open, monday_close,
                   tuesday_open, tuesday_close,
                   wednesday_open, wednesday_close,
                   thursday_open, thursday_close,
                   friday_open, friday_close,
                   saturday_open, saturday_close,
                   sunday_open, sunday_close
            FROM tenant_services WHERE tenant_id = :tid
        """), {"tid": row["id"]}).mappings().all()

    ctx = TenantContext(
        tenant_id           = row["id"],
        brand_name          = row["brand_name"],
        status              = row["status"],
        system_prompt       = row["system_prompt"],
        model_name          = row["model_name"],
        api_key             = row["api_key"],
        provider            = row["provider"] or "groq",
        timezone            = row["timezone"] or "",
        physical_address    = row["physical_address"] or "",
        city                = row["city"] or "",
        state               = row["state"] or "",
        country             = row["country"] or "",
        primary_language    = row["primary_language"] or "en",
        supported_languages = [l.strip() for l in (row["supported_languages"] or "en").split(",")],
        default_tax_rate    = float(row["default_tax_rate"] or 0),
    )
    for s in svc_rows:
        hours = {
            day: {"open": s[f"{day}_open"], "close": s[f"{day}_close"]}
            for day in _DAYS
        }
        ctx.services[s["service_type"]] = ServiceInfo(
            service_type       = s["service_type"],
            is_active          = s["is_active"],
            fee_type           = s["fee_type"],
            fee_amount         = float(s["fee_amount"] or 0),
            hours_by_day       = hours,
            delivery_radius_km = float(s["delivery_radius_km"]) if s.get("delivery_radius_km") is not None else None,
        )
    return ctx


def db_get_customer(from_number: str, tenant_id: str) -> dict:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT c.id AS customer_id, c.full_name, c.email,
                   tc.id AS tc_id, a.address_line_1
            FROM customers c
            LEFT JOIN tenant_customers tc ON c.id = tc.customer_id AND tc.tenant_id = :tid
            LEFT JOIN tenant_customer_addresses a ON tc.id = a.tenant_customer_id AND a.is_default = true
            WHERE c.phone_number = :ph
        """), {"tid": tenant_id, "ph": from_number}).mappings().first()
    if not row:
        return {"is_global": False, "is_tenant": False,
                "full_name": "", "email": "", "address_line_1": "", "customer_id": None}

    # Defense: ignore full_name that looks like a phone number (legacy bug from
    # v4.7 where the bot incorrectly asked for phone and stored it as name).
    full_name = row["full_name"] or ""
    if re.match(r"^\+?\d{7,}$", full_name.strip()):
        logger.warning(
            f"[db_get_customer] ignoring phone-like full_name='{full_name}' "
            f"for phone={from_number} (data hygiene)"
        )
        full_name = ""

    return {
        "is_global":      True,
        "is_tenant":      row["tc_id"] is not None,
        "full_name":      full_name,
        "email":          row["email"] or "",
        "address_line_1": row["address_line_1"] or "",
        "customer_id":    str(row["customer_id"]),
    }


def db_save_customer(user_phone, tenant_id, full_name, address, email, order_summary) -> str:
    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO customers (id, phone_number, full_name, email)
            VALUES (gen_random_uuid(), :ph, :name, :em)
            ON CONFLICT (phone_number) DO UPDATE
                SET full_name = EXCLUDED.full_name,
                    email     = COALESCE(EXCLUDED.email, customers.email)
            RETURNING id
        """), {"ph": user_phone, "name": full_name, "em": email or None}).fetchone()
        customer_id = row[0]

        tc_row = conn.execute(text("""
            INSERT INTO tenant_customers (tenant_id, customer_id, tenant_specific_status)
            VALUES (:tid, :cid, 'Active')
            ON CONFLICT (tenant_id, customer_id) DO NOTHING
            RETURNING id
        """), {"tid": tenant_id, "cid": customer_id}).fetchone()
        if not tc_row:
            tc_row = conn.execute(text("""
                SELECT id FROM tenant_customers WHERE tenant_id = :tid AND customer_id = :cid
            """), {"tid": tenant_id, "cid": customer_id}).fetchone()
        tc_id = tc_row[0]

        conn.execute(text("""
            UPDATE tenant_customer_addresses SET is_default = false
            WHERE tenant_customer_id = :tcid
        """), {"tcid": tc_id})
        conn.execute(text("""
            INSERT INTO tenant_customer_addresses
                (id, tenant_customer_id, address_line_1, is_default, city, state)
            VALUES (gen_random_uuid(), :tcid, :addr, true, 'Toronto', 'ON')
        """), {"tcid": tc_id, "addr": address})
    logger.info(f"[db_save] customer_id={customer_id}")
    return str(customer_id)


def _generate_order_number(tenant_id: str) -> str:
    prefix   = tenant_id.replace("-", "")[:3].upper()
    date_str = datetime.now().strftime("%Y%m%d")
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) FROM orders
            WHERE tenant_id = :tid AND DATE(order_date) = CURRENT_DATE
        """), {"tid": tenant_id}).fetchone()
    seq = str((row[0] or 0) + 1).zfill(4)
    return f"{prefix}-{date_str}-{seq}"


def db_save_order(session_id, tenant_id, customer_id, customer_name,
                  customer_phone, customer_email, service_type,
                  delivery_address, cart: OrderCart, notes=None) -> str:
    """
    Persists the order with full audit trail:
      - orders.status starts at 'confirmed' (customer accepted summary).
      - orders.payment_status starts at 'pending' (payment not yet processed).
      - Each order_items row carries its own tax_rate/tax_amount when applicable.
      - The aggregated 'tax' line is kept for backward compatibility.
    """
    order_number = _generate_order_number(tenant_id)
    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO orders (
                order_number, tenant_id, customer_id,
                customer_name, customer_phone, customer_email,
                service_type, delivery_address,
                subtotal, tax_total, tip_total, delivery_total, grand_total,
                notes, session_id, status, payment_status
            ) VALUES (
                :num, :tid, :cid, :name, :phone, :email,
                :stype, :addr, :sub, :tax, 0, :del, :grand, :notes, :sid,
                'confirmed', 'pending'
            ) RETURNING id
        """), {
            "num": order_number, "tid": tenant_id, "cid": customer_id,
            "name": customer_name, "phone": customer_phone,
            "email": customer_email or None, "stype": service_type,
            "addr": delivery_address or "",
            "sub": cart.subtotal, "tax": cart.tax_total,
            "del": cart.delivery_fee, "grand": cart.grand_total,
            "notes": notes, "sid": session_id,
        }).fetchone()
        order_id = row[0]

        # Per-item rows with effective tax info
        for item in cart.items:
            eff_rate   = item.effective_tax_rate(cart.tenant_default_tax)
            tax_amount = round(item.line_total * eff_rate, 2) if eff_rate > 0 else 0.0
            conn.execute(text("""
                INSERT INTO order_items (
                    order_id, line_type, item_code, item_name,
                    quantity, unit_price, line_total, prep_notes,
                    is_tax_exempt, tax_rate, tax_amount
                ) VALUES (
                    :oid, 'item', :code, :name,
                    :qty, :price, :total, :notes,
                    :exempt, :rate, :tamt
                )
            """), {
                "oid": order_id, "code": item.item_code or None,
                "name": item.name, "qty": item.quantity,
                "price": item.unit_price, "total": item.line_total,
                "notes": item.prep_notes or None,
                "exempt": item.is_tax_exempt,
                "rate":   eff_rate if eff_rate > 0 else None,
                "tamt":   tax_amount if tax_amount > 0 else None,
            })

        # Delivery line (kept as before — convenient for invoice rendering)
        if cart.delivery_fee > 0:
            conn.execute(text("""
                INSERT INTO order_items (order_id, line_type, item_name, quantity, unit_price, line_total)
                VALUES (:oid, 'delivery', 'Delivery fee', 1, :fee, :fee)
            """), {"oid": order_id, "fee": cart.delivery_fee})

        # Aggregated tax line (kept for backward compatibility with existing reports)
        if cart.tax_total > 0:
            conn.execute(text("""
                INSERT INTO order_items (order_id, line_type, item_name, quantity, unit_price, line_total)
                VALUES (:oid, 'tax', 'Tax', 1, :tax, :tax)
            """), {"oid": order_id, "tax": cart.tax_total})

    logger.info(
        f"[db_save_order] order_number={order_number} "
        f"subtotal=${cart.subtotal} tax=${cart.tax_total} "
        f"delivery=${cart.delivery_fee} grand_total=${cart.grand_total} "
        f"items={len(cart.items)}"
    )
    return order_number


# ─────────────────────────────────────────────────────────────────────────────
# Menu — exclusively from menu_items DB table
# ─────────────────────────────────────────────────────────────────────────────

def get_menu_for_language(tenant_id: str, language: str) -> str:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    tmc.serve_order,
                    COALESCE(mct_lang.name, mct_en.name, tmc.category_code) AS category_name,
                    mi.item_code,
                    COALESCE(mit_lang.name, mi.name)              AS item_name,
                    COALESCE(mit_lang.description, mi.description) AS item_desc,
                    mi.price,
                    mi.sort_order
                FROM tenant_menu_categories tmc
                LEFT JOIN menu_category_translations mct_lang
                    ON tmc.category_code = mct_lang.category_code
                    AND mct_lang.language_code = :lang
                LEFT JOIN menu_category_translations mct_en
                    ON tmc.category_code = mct_en.category_code
                    AND mct_en.language_code = 'en'
                JOIN menu_items mi
                    ON tmc.tenant_id = mi.tenant_id
                    AND tmc.category_code = mi.category_code
                LEFT JOIN menu_item_translations mit_lang
                    ON mi.tenant_id = mit_lang.tenant_id
                    AND mi.item_code = mit_lang.item_code
                    AND mit_lang.language_code = :lang
                WHERE tmc.tenant_id = :tid
                    AND tmc.is_active = true
                    AND mi.is_available = true
                ORDER BY tmc.serve_order, mi.sort_order
            """), {"tid": tenant_id, "lang": language}).mappings().all()

        if not rows:
            logger.warning(f"[menu] No items found for tenant {tenant_id}")
            return "Menu information is not available."

        current_category = None
        lines = []
        for row in rows:
            if row["category_name"] != current_category:
                current_category = row["category_name"]
                lines.append(f"\n## {current_category}")
            desc = f" — {row['item_desc']}" if row["item_desc"] else ""
            lines.append(f"[{row['item_code']}] {row['item_name']} — ${row['price']:.2f}{desc}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[menu] DB error: {e}")
        return "Menu information could not be retrieved."


def lookup_translated_plural(tenant_id: str, item_code: str, language: str) -> str:
    """
    Returns the language-specific plural form if available, else empty string.
    Falls back to menu_items.name_plural is handled by OrderItem.display_name().
    """
    if not item_code or not language:
        return ""
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT name_plural
                FROM menu_item_translations
                WHERE tenant_id = :tid AND item_code = :code AND language_code = :lang
                LIMIT 1
            """), {"tid": tenant_id, "code": item_code, "lang": language}).fetchone()
        return (row[0] or "") if row else ""
    except Exception as e:
        logger.error(f"[lookup_translated_plural] error: {e}")
        return ""


def get_menu_categories(tenant_id: str) -> list[str]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT COALESCE(mct.name, tmc.category_code) AS name
                FROM tenant_menu_categories tmc
                LEFT JOIN menu_category_translations mct
                    ON tmc.category_code = mct.category_code
                    AND mct.language_code = 'en'
                WHERE tmc.tenant_id = :tid AND tmc.is_active = true
                ORDER BY tmc.serve_order
            """), {"tid": tenant_id}).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        logger.error(f"[categories] error: {e}")
        return []


def lookup_menu_item(tenant_id: str, item_name: str) -> Optional[dict]:
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT item_code, name, name_plural, price,
                       is_tax_exempt, tax_rate_override
                FROM menu_items
                WHERE tenant_id = :tid
                    AND is_available = true
                    AND (
                        LOWER(name) LIKE LOWER(:search)
                        OR LOWER(:exact) LIKE LOWER('%' || name || '%')
                    )
                ORDER BY
                    CASE WHEN LOWER(name) = LOWER(:exact) THEN 0 ELSE 1 END,
                    LENGTH(name)
                LIMIT 1
            """), {"tid": tenant_id, "search": f"%{item_name}%", "exact": item_name}).mappings().first()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"[lookup_item] error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Timezone & Service availability (day-of-week aware)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_timezone_from_address(llm: LLMProvider, ctx: TenantContext) -> str:
    if ctx.timezone and "/" in ctx.timezone:
        logger.info(f"[tz] using DB timezone: {ctx.timezone}")
        return ctx.timezone
    classifier = _classifier_or_tenant(llm)
    location = f"{ctx.city}, {ctx.state}, {ctx.country}"
    try:
        tz = classifier.classify(
            f"What is the IANA timezone identifier for: {location}? "
            f"Reply with ONLY the timezone string, e.g. America/Toronto. No explanation.",
            max_tokens=30,
        ).strip().strip('"').strip("'")
        tz = "/".join(part.capitalize() for part in tz.split("/"))
        if "/" in tz and len(tz) < 50:
            logger.info(f"[tz] resolved {location} → {tz}")
            return tz
    except Exception as e:
        logger.error(f"[tz] error: {e}")
    logger.warning(f"[tz] could not resolve timezone — defaulting to UTC")
    return "UTC"


def _check_service_availability(ctx: TenantContext) -> None:
    try:
        tz     = zi.ZoneInfo(ctx.timezone)
        now_dt = datetime.now(tz)
        now    = now_dt.time()
        today  = _DAYS[now_dt.weekday()]
    except Exception:
        now_dt = datetime.utcnow()
        now    = now_dt.time()
        today  = _DAYS[now_dt.weekday()]

    # Check tenant_closures
    closed_today = False
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT id FROM tenant_closures
                WHERE tenant_id = :tid AND closed_date = CURRENT_DATE
                LIMIT 1
            """), {"tid": ctx.tenant_id}).fetchone()
            closed_today = row is not None
    except Exception as e:
        logger.error(f"[closures] error: {e}")

    for svc in ctx.services.values():
        if not svc.is_active:
            svc.is_open_now = False
            continue

        if closed_today:
            svc.is_open_now = False
            logger.info(f"[svc] {svc.service_type}: closed today (closure record)")
            continue

        day_hours = svc.hours_by_day.get(today, {})
        day_open  = day_hours.get("open")
        day_close = day_hours.get("close")

        if day_open is None or day_close is None:
            svc.is_open_now = False
            logger.info(f"[svc] {svc.service_type}: closed on {today} (no hours)")
            continue

        if day_close > day_open:
            svc.is_open_now = day_open <= now <= day_close
        else:
            svc.is_open_now = now >= day_open or now <= day_close

        logger.info(f"[svc] {svc.service_type}: open={svc.is_open_now} "
                    f"({today} {day_open}-{day_close} now={now.strftime('%H:%M')})")


def _get_available_services(ctx: TenantContext) -> list[ServiceInfo]:
    return [s for s in ctx.services.values() if s.is_active and s.is_open_now]


def _format_service_options(services: list[ServiceInfo], lang: str) -> str:
    options = []
    for s in services:
        if s.service_type == ServiceType.PICKUP:
            options.append(
                "🏃 *Pickup* — puedes recoger tu pedido sin costo adicional" if lang == "es"
                else "🏃 *Pickup* — pick up your order at no extra charge"
            )
        elif s.service_type == ServiceType.DELIVERY:
            if s.fee_type == "free" or s.fee_amount == 0:
                fee_str = "gratis" if lang == "es" else "free"
            elif s.fee_type == "fixed":
                fee_str = f"${s.fee_amount:.2f}"
            else:
                fee_str = "calculado según tu dirección" if lang == "es" else "calculated based on your address"
            options.append(
                f"🛵 *Delivery* — te lo llevamos a domicilio ({fee_str})" if lang == "es"
                else f"🛵 *Delivery* — delivered to your door ({fee_str})"
            )
    return "\n".join(options)


def _build_hours_note(ctx: TenantContext) -> tuple[str, str, str]:
    """
    Returns (today_name, today_time, hours_note) built from tenant_services hours_by_day.
    Uses pickup or dine_in service as reference for restaurant hours.
    """
    today_name = ""
    today_time = ""
    hours_note = ""
    try:
        tz        = zi.ZoneInfo(ctx.timezone)
        now_local = datetime.now(tz)
        today_key = _DAYS[now_local.weekday()]
        today_name = _DAY_NAMES[now_local.weekday()]
        today_time = now_local.strftime("%I:%M %p").lstrip("0")

        ref_svc = next(
            (s for s in ctx.services.values() if s.service_type in ("pickup","dine_in")),
            None
        )
        if ref_svc:
            hours_lines = []
            for i, day_key in enumerate(_DAYS):
                dh = ref_svc.hours_by_day.get(day_key, {})
                o  = dh.get("open")
                c  = dh.get("close")
                if o and c:
                    def fmt(t):
                        return datetime.combine(now_local.date(), t).strftime("%I:%M %p").lstrip("0")
                    hours_lines.append(f"{_DAY_NAMES[i]}: {fmt(o)} – {fmt(c)}")
                else:
                    hours_lines.append(f"{_DAY_NAMES[i]}: Closed")

            hours_str  = "\n".join(hours_lines)
            hours_note = (
                f"\n\nToday is {today_name} and the current local time is {today_time}.\n"
                f"Our weekly hours are:\n{hours_str}\n"
                f"Use ONLY these hours if the customer asks. Do NOT guess or invent hours."
            )
    except Exception as e:
        logger.error(f"[hours_note] error: {e}")

    return today_name, today_time, hours_note


# ─────────────────────────────────────────────────────────────────────────────
# Mapbox — dynamic per-tenant proximity, configurable threshold
# ─────────────────────────────────────────────────────────────────────────────

async def _geocode_tenant_proximity(ctx: TenantContext) -> str:
    """
    Geocodes the tenant's physical address and returns "lng,lat" for Mapbox
    proximity bias. Cached per tenant_id. Falls back to country-default if
    geocoding fails.
    """
    if ctx.tenant_id in _TENANT_PROXIMITY:
        return _TENANT_PROXIMITY[ctx.tenant_id]

    if not MAPBOX_TOKEN:
        return ""

    parts = [p for p in [ctx.physical_address, ctx.city, ctx.state, ctx.country] if p]
    query = ", ".join(parts)
    if not query:
        return ""

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json",
                params={"access_token": MAPBOX_TOKEN, "limit": 1}
            )
        features = resp.json().get("features", [])
        if features and "center" in features[0]:
            lng, lat = features[0]["center"]
            proximity = f"{lng},{lat}"
            _TENANT_PROXIMITY[ctx.tenant_id] = proximity
            logger.info(f"[mapbox_proximity] tenant={ctx.tenant_id} → {proximity}")
            return proximity
    except Exception as e:
        logger.error(f"[mapbox_proximity] error: {e}")

    # Fallback by country
    fallback_by_country = {
        "Canada":   "-79.3832,43.6532",   # Toronto
        "USA":      "-74.0060,40.7128",   # NYC
        "Colombia": "-74.7813,10.9685",   # Barranquilla
        "Mexico":   "-99.1332,19.4326",   # CDMX
    }
    proximity = fallback_by_country.get(ctx.country, "")
    _TENANT_PROXIMITY[ctx.tenant_id] = proximity
    return proximity


def _haversine_km(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    """Great-circle distance in kilometers between two lng/lat points."""
    R = 6371.0  # Earth radius in km
    phi1   = math.radians(lat1)
    phi2   = math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlamb  = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlamb/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _country_to_mapbox_code(country: str) -> str:
    """Convert country name to comma-separated Mapbox country codes."""
    mapping = {
        "Canada":   "ca",
        "USA":      "us",
        "United States": "us",
        "Mexico":   "mx",
        "Colombia": "co",
    }
    code = mapping.get(country, "")
    if code == "ca": return "ca,us"
    if code == "us": return "us,ca"
    if code: return code
    return "ca,us"


_UNIT_PATTERNS = [
    # "unit 1710", "unit #1710", "apt 5B", "apartment 12", "suite 200", "# 1710"
    re.compile(r"\b(?:unit|apt|apartment|suite|ste)\.?\s*#?\s*([A-Za-z0-9\-]+)\b", re.IGNORECASE),
    # Standalone leading "#1710" before a comma (must be followed by comma or end)
    re.compile(r"^#\s*([A-Za-z0-9\-]+)\s*(?=,)", re.IGNORECASE),
    # "1710-273 Pharmacy" pattern (unit-streetnum prefix)
    re.compile(r"^(\d{2,5})\s*[-–]\s*(?=\d)", re.IGNORECASE),
]

def _extract_unit(address: str) -> tuple[str, str]:
    """
    If the address mentions a unit/apt/suite, extract it and return
    (clean_address_without_unit, unit_string).
    If no unit detected, returns (address_as_is, '').
    """
    for pattern in _UNIT_PATTERNS:
        m = pattern.search(address)
        if m:
            unit = m.group(1).strip()
            # Remove the matched portion, plus any surrounding commas/whitespace
            clean = pattern.sub("", address, count=1)
            clean = re.sub(r"^\s*,\s*", "", clean)         # leading comma
            clean = re.sub(r",\s*,", ",", clean)           # double commas
            clean = re.sub(r"\s+", " ", clean).strip(", ")
            if unit:
                return clean, unit
    return address, ""


async def validate_address_mapbox(address: str, ctx: TenantContext) -> dict:
    """
    Validates an address using Mapbox geocoding, biased to the tenant's location.
    Threshold is MAPBOX_THRESHOLD. If a delivery_radius_km is configured on
    the delivery service, also checks that the address is within that radius.

    Preserves unit/apt numbers across geocoding (Mapbox strips them).

    Returns dict with: valid, canonical, suggestions, relevance,
    out_of_zone (bool), distance_km (float|None), unit (str).
    """
    if not MAPBOX_TOKEN:
        return {"valid": True, "canonical": address, "suggestions": [],
                "relevance": 1.0, "out_of_zone": False, "distance_km": None, "unit": ""}

    # Extract unit/apt before geocoding (Mapbox doesn't handle these well)
    base_address, unit = _extract_unit(address)
    if unit:
        logger.info(f"[mapbox] extracted unit='{unit}' from address; base='{base_address}'")

    parts = [base_address]
    if ctx.city:    parts.append(ctx.city)
    if ctx.state:   parts.append(ctx.state)
    query = ", ".join(parts)

    proximity = await _geocode_tenant_proximity(ctx)
    countries = _country_to_mapbox_code(ctx.country)

    params = {
        "access_token": MAPBOX_TOKEN,
        "types":        "address",
        "limit":        3,
        "language":     "en",
        "country":      countries,
    }
    if proximity:
        params["proximity"] = proximity

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json",
                params=params,
            )
        features = resp.json().get("features", [])
        logger.info(f"[mapbox] query='{query}' results={len(features)}")

        if not features:
            return {"valid": False, "canonical": "", "suggestions": [],
                    "relevance": 0.0, "out_of_zone": False, "distance_km": None, "unit": unit}

        top       = features[0]
        relevance = top.get("relevance", 0)
        canonical = top.get("place_name", base_address)

        # Re-attach unit to canonical for delivery instructions
        if unit:
            canonical = f"Unit {unit}, {canonical}"

        center = top.get("center", [None, None])
        addr_lng, addr_lat = center if len(center) == 2 else (None, None)

        logger.info(f"[mapbox] top relevance={relevance} place={canonical[:100]}")

        if relevance < MAPBOX_THRESHOLD:
            return {
                "valid":       False,
                "canonical":   canonical,
                "suggestions": [
                    (f"Unit {unit}, " if unit else "") + f["place_name"]
                    for f in features[:3]
                ],
                "relevance":   relevance,
                "out_of_zone": False,
                "distance_km": None,
                "unit":        unit,
            }

        # Address found — now check delivery radius if configured
        distance_km = None
        out_of_zone = False
        delivery_svc = ctx.services.get("delivery")
        radius = delivery_svc.delivery_radius_km if delivery_svc else None

        if radius and radius > 0 and proximity and addr_lng is not None:
            try:
                t_lng, t_lat = [float(x) for x in proximity.split(",")]
                distance_km  = round(_haversine_km(t_lng, t_lat, addr_lng, addr_lat), 2)
                out_of_zone  = distance_km > radius
                logger.info(
                    f"[mapbox_radius] distance={distance_km}km radius={radius}km "
                    f"out_of_zone={out_of_zone}"
                )
            except Exception as e:
                logger.error(f"[mapbox_radius] error: {e}")

        return {
            "valid":       not out_of_zone,
            "canonical":   canonical,
            "suggestions": [
                (f"Unit {unit}, " if unit else "") + f["place_name"]
                for f in features[:3]
            ],
            "relevance":   relevance,
            "out_of_zone": out_of_zone,
            "distance_km": distance_km,
            "unit":        unit,
        }
    except Exception as e:
        logger.error(f"[mapbox] Error: {e}")
        return {"valid": True, "canonical": address, "suggestions": [],
                "relevance": 0.0, "out_of_zone": False, "distance_km": None, "unit": unit}


# ─────────────────────────────────────────────────────────────────────────────
# LLM Classifiers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_order_action(llm: LLMProvider, message: str, cart: OrderCart,
                          menu_text: str, tenant_id: str) -> dict:
    """
    Classifies the customer message into an order action. Uses the system
    classifier LLM if configured, else the tenant's LLM. Retries once if
    the first attempt returns an empty markdown fence (common Claude failure).
    """
    classifier = _classifier_or_tenant(llm)
    cart_display = cart.to_display() if not cart.is_empty else "Empty cart"

    def _build_prompt(strict_retry: bool = False) -> str:
        strict_preamble = ""
        if strict_retry:
            strict_preamble = (
                "CRITICAL: Your last reply was an empty markdown fence (```json with no content). "
                "DO NOT do that again. Output a single JSON object directly — no markdown, "
                "no code fences, no commentary. Just the JSON. Start with { and end with }.\n\n"
            )
        return f"""{strict_preamble}You extract structured order actions from a customer message in a restaurant chat. You ALWAYS respond with valid JSON only.

CURRENT CART:
{cart_display}

MENU (use these EXACT item_code and name values — never invent):
{menu_text[:6000]}

CUSTOMER MESSAGE: "{message}"

Decide ONE action:
- "add"     → customer wants to ADD one or more items (e.g. "I'd like 2 pupusas", "give me a horchata", "add a tamarindo", "I wanna carne asada for me and one for my wife", "lemme get a coke")
- "remove"  → customer wants to REMOVE an item from the cart (e.g. "remove the horchata", "take off the soda")
- "modify"  → customer wants to CHANGE quantity or prep notes of an item ALREADY in cart (e.g. "make it 3 instead of 2", "no onions on the pupusa", "my wife wants onions on her carne asada")
- "confirm" → customer is done and confirms the cart (e.g. "that's everything", "that's all", "ok", "yes that's it", "ready to checkout")
- "inquiry" → customer asks a QUESTION about menu/hours/ingredients without ordering (e.g. "what drinks do you have?", "is it spicy?")
- "unclear" → truly cannot tell

CRITICAL RULES:
- If the message contains a menu item name with a quantity word (one, two, 2, 3, a, an) OR phrases like "I want", "I'd like", "give me", "add", "also", "and", "plus", "I wanna", "lemme get", "I'll have" followed by an item → action is "add".
- "X for me and one for my wife" / "one for me and one for X" → quantity is 2 of the same item.
- "X for me, Y for my wife" (DIFFERENT items) → add BOTH.
- Match menu items loosely (e.g. "pupusa revuelta" → find "Pupusa Revuelta", "horchata" → find "Horchata Salvadoreña", "carne asada" → find "Carne Asada"). Use the closest match from the menu.
- For quantities: "a" / "an" / "one" = 1, "two" = 2, "three" = 3, etc.
- For "modify": only use this if the item is ALREADY in the cart and the customer wants to change it. If they want to modify an item that ISN'T in the cart, treat it as "add" with the prep_notes set.
- Never invent item_code or name values that aren't in the menu above.

EXAMPLES:

Message: "I'd like 2 pupusas revueltas"
{{"action":"add","items_to_add":[{{"item_code":"PUP-001","name":"Pupusa Revuelta","quantity":2,"prep_notes":""}}],"items_to_remove":[],"items_to_modify":[]}}

Message: "add 1 horchata please"
{{"action":"add","items_to_add":[{{"item_code":"DRK-001","name":"Horchata Salvadoreña","quantity":1,"prep_notes":""}}],"items_to_remove":[],"items_to_modify":[]}}

Message: "also 1 pupusa de queso, no onions"
{{"action":"add","items_to_add":[{{"item_code":"PUP-003","name":"Pupusa Solo Queso","quantity":1,"prep_notes":"no onions"}}],"items_to_remove":[],"items_to_modify":[]}}

Message: "I wanna carne asada for me and one for my wife"
{{"action":"add","items_to_add":[{{"item_code":"MAIN-001","name":"Carne Asada","quantity":2,"prep_notes":""}}],"items_to_remove":[],"items_to_modify":[]}}

Message: "give me one tamarindo"
{{"action":"add","items_to_add":[{{"item_code":"DRK-004","name":"Jugo de Tamarindo","quantity":1,"prep_notes":""}}],"items_to_remove":[],"items_to_modify":[]}}

Message: "i'll take a coke"
{{"action":"add","items_to_add":[{{"item_code":"DRK-005","name":"Coca-Cola","quantity":1,"prep_notes":""}}],"items_to_remove":[],"items_to_modify":[]}}

Message: "lemme get 2 tacos"
{{"action":"add","items_to_add":[{{"item_code":"TAC-001","name":"Taco","quantity":2,"prep_notes":""}}],"items_to_remove":[],"items_to_modify":[]}}

Message: "remove the horchata"
{{"action":"remove","items_to_add":[],"items_to_remove":["Horchata Salvadoreña"],"items_to_modify":[]}}

Message: "actually make it 3 pupusas revueltas instead of 2"
{{"action":"modify","items_to_add":[],"items_to_remove":[],"items_to_modify":[{{"name":"Pupusa Revuelta","prep_notes":"","quantity":3}}]}}

Message: "my wife wants onions on her carne asada"
{{"action":"modify","items_to_add":[],"items_to_remove":[],"items_to_modify":[{{"name":"Carne Asada","prep_notes":"with onions","quantity":0}}]}}

Message: "that's everything"
{{"action":"confirm","items_to_add":[],"items_to_remove":[],"items_to_modify":[]}}

Message: "what drinks do you have?"
{{"action":"inquiry","items_to_add":[],"items_to_remove":[],"items_to_modify":[]}}

Now respond with JSON ONLY for the customer message above. No markdown, no explanation.
"""

    def _parse(raw: str) -> Optional[dict]:
        """Try to parse the LLM output. Returns None if it looks like an empty fence."""
        if not raw:
            return None
        # Detect empty markdown fence (the bug pattern we saw in v4.11)
        stripped = raw.strip()
        if stripped in ("```json", "```", "```json\n```", "``` ```"):
            return None
        # Strip fences if present
        cleaned = re.sub(r"```json|```", "", stripped).strip()
        # Empty after cleaning → empty fence
        if not cleaned:
            return None
        # Extract first JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    # First attempt
    raw1 = ""
    try:
        raw1 = classifier.classify(_build_prompt(strict_retry=False), max_tokens=CLASSIFIER_TOKENS)
        logger.info(f"[cart_action_raw] message='{message[:80]}' raw={raw1[:300] if raw1 else 'EMPTY'}")
        data = _parse(raw1)
        if data is not None:
            logger.info(
                f"[cart_action] action={data.get('action')} "
                f"add={len(data.get('items_to_add',[]))} "
                f"remove={len(data.get('items_to_remove',[]))} "
                f"modify={len(data.get('items_to_modify',[]))}"
            )
            return data
    except Exception as e:
        logger.error(f"[extract_order_action] first attempt error: {e}")

    # Retry once with stricter prompt
    logger.warning(f"[cart_action_retry] first attempt unparseable ({raw1[:100]!r}); retrying with strict prompt")
    raw2 = ""
    try:
        raw2 = classifier.classify(_build_prompt(strict_retry=True), max_tokens=CLASSIFIER_TOKENS)
        logger.info(f"[cart_action_raw_retry] raw={raw2[:300] if raw2 else 'EMPTY'}")
        data = _parse(raw2)
        if data is not None:
            logger.info(
                f"[cart_action_retry] action={data.get('action')} "
                f"add={len(data.get('items_to_add',[]))}"
            )
            return data
    except Exception as e:
        logger.error(f"[extract_order_action] retry error: {e}")

    logger.error(
        f"[cart_action_FAIL] both attempts unparseable for message='{message[:80]}'. "
        f"raw1={raw1[:150]!r} raw2={raw2[:150]!r}"
    )
    return {"action": "unclear", "items_to_add": [], "items_to_remove": [], "items_to_modify": []}


def _classify_intent(llm: LLMProvider, message: str, context: str) -> str:
    classifier = _classifier_or_tenant(llm)
    prompt = (
        f"Classify this customer message in a food ordering chat.\n"
        f"Context: {context}\nMessage: \"{message}\"\n\n"
        f"Reply with EXACTLY one word:\n"
        f"provide_data — answering the question asked\n"
        f"back_to_order — wants to change/add items or ask about the menu\n"
        f"confirm — saying yes/confirming\n"
        f"cancel — wants to cancel or start over\n"
        f"other — unclear"
    )
    try:
        result = classifier.classify(prompt).split()[0]
        return result if result in ("provide_data","back_to_order","confirm","cancel","other") else "other"
    except Exception as e:
        logger.error(f"[classify_intent] error: {e}")
        return "other"


def _classify_confirmation(llm: LLMProvider, message: str) -> str:
    classifier = _classifier_or_tenant(llm)
    prompt = (
        f"Is this message a confirmation (yes), rejection (no), or something else?\n"
        f"Message: \"{message}\"\n"
        f"sí, dale, claro, listo, perfecto, ok, correcto, yes, sure = yes\n"
        f"no, nope, cancel, incorrecto, cambiar = no\n"
        f"Reply with ONE word: yes, no, or other."
    )
    try:
        result = classifier.classify(prompt).split()[0]
        return result if result in ("yes","no","other") else "other"
    except Exception as e:
        logger.error(f"[classify_confirm] error: {e}")
        return "other"


def _detect_language(message: str, supported: list[str]) -> str:
    default       = supported[0] if supported else "en"
    spanish_words = {"hola","buenos","buenas","gracias","por favor","quiero","quisiera",
                     "tengo","puedo","qué","que","sí","me","mi","tu","una","para","con"}
    french_words  = {"bonjour","bonsoir","merci","voudrais","je","vous","nous","avec","pour","une","les","des"}
    bengali_words = {"আমি","আপনি","কি","হ্যালো","ধন্যবাদ"}
    words         = set(message.lower().split())
    scores: dict[str, int] = {}
    if "es" in supported: scores["es"] = len(words & spanish_words)
    if "fr" in supported: scores["fr"] = len(words & french_words)
    if "bn" in supported: scores["bn"] = len(words & bengali_words)
    if not scores:
        return default
    best_lang  = max(scores, key=scores.get)
    best_score = scores[best_lang]
    return best_lang if best_score >= 2 and best_lang in supported else default


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_affirmative(raw: str) -> bool:
    lowered = raw.strip().lower()
    _YES = {"yes","si","sí","yep","yeah","correct","ok","okay","sure","confirm",
            "confirmed","adelante","procede","dale","claro","yup","perfecto",
            "listo","va","órale","orale","all good","sounds good"}
    if lowered in _YES:
        return True
    _PHRASES = ("yes,","yes.","yes!","si,","si.","sí,","that's correct","looks good",
                "all good","go ahead","todo bien","está bien","esta bien",
                "todo correcto","confirmo","confirmar","that's everything","that's all")
    return any(p in lowered for p in _PHRASES)


# Phrases that indicate the customer is jumping ahead to checkout/service
# selection before having an order in the cart.
_PREMATURE_CHECKOUT_KEYWORDS = {
    "delivery", "deliver", "pickup", "pick up", "pick-up", "dine in", "dine-in",
    "domicilio", "a domicilio", "recoger", "para llevar", "comer aqui", "comer aquí",
}

def _looks_like_premature_checkout(message: str) -> bool:
    """
    True if the message looks like a service-selection / address / name reply
    when the customer hasn't ordered anything yet.
    """
    msg = message.strip().lower()
    if not msg:
        return False
    # Direct service-type keywords
    for kw in _PREMATURE_CHECKOUT_KEYWORDS:
        if kw == msg or f" {kw} " in f" {msg} " or msg.startswith(kw + " ") or msg.endswith(" " + kw):
            return True
    # Looks like a street address (number + street word)
    if re.match(r"^\d{1,6}\s+\w+", msg):
        return True
    return False


def _extract_name(raw: str) -> str:
    stripped = raw.strip()
    for prefix in ("my name is ","i am ","i'm ","soy ","me llamo ","mi nombre es ","it's ","its "):
        if stripped.lower().startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    stripped = stripped.strip()
    # Defense: never accept phone-like strings as a name
    if re.match(r"^\+?\d{7,}$", stripped):
        logger.warning(f"[extract_name] rejected phone-like input as name: {stripped!r}")
        return ""
    return stripped.title()


_EMAIL_RE   = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_SKIP_EMAIL = {"no","skip","none","n/a","-","sin email","no tengo","omitir","saltar","paso"}

def _extract_email(raw: str) -> str:
    if raw.strip().lower() in _SKIP_EMAIL:
        return ""
    m = _EMAIL_RE.search(raw)
    return m.group(0) if m else ""


LANG_INSTRUCTIONS = {
    "en": "Respond ONLY in English. Never switch languages. Be warm and conversational.",
    "es": "Responde ÚNICAMENTE en español mexicano. Nunca cambies de idioma. Usa expresiones cálidas como '¡Con gusto!', '¡Claro que sí!'.",
    "fr": "Réponds UNIQUEMENT en français. Ne change jamais de langue. Sois chaleureux et naturel.",
    "bn": "শুধুমাত্র বাংলায় উত্তর দিন। কখনো ভাষা পরিবর্তন করবেন না।",
}

_LEGACY_RE = re.compile(
    r"\[ORDER_FINALIZED\]|manage_customer_data\b|query_vector_database\b|STRICT PROTOCOL[\s\S]*",
    re.IGNORECASE,
)

def _clean_base_prompt(raw: str, brand: str) -> str:
    cleaned = _LEGACY_RE.sub("", raw).strip()
    cleaned = cleaned.replace("{restaurant_name}", brand).replace("{menu_context}", "")
    cleaned = re.sub(r"Menu Context:[^\n]*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned if len(cleaned) >= 20 else f"You are a professional ordering assistant for {brand}."


def _build_final_confirm_summary(session: "Session") -> str:
    """
    Builds a complete, deterministic order summary in conversational language.
    NO LLM involved. Uses exact cart totals and Mapbox-normalized address.
    Tax is revealed here (not during the ORDER state).
    Channel-aware via _voice_safe() at the call site.
    """
    c    = session.collected
    lang = session.language
    cart = session.cart

    # Natural prose listing using each item's lang-aware plural form
    def _natural_items(items, lang):
        parts = []
        for it in items:
            qty = it.quantity
            display = it.display_name()  # already plural-aware when qty>1
            if qty == 1:
                if lang == "es":
                    prefix = "una" if display and display[0].lower() in "aeiouáéíóú" else "un"
                    part = f"{prefix} {display}"
                else:
                    prefix = "an" if display and display[0].lower() in "aeiou" else "a"
                    part = f"{prefix} {display}"
            else:
                part = f"{qty} {display}"
            if it.prep_notes:
                connector = " con " if lang == "es" else " with "
                part += f"{connector}{it.prep_notes}"
            parts.append(part)
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            joiner = " y " if lang == "es" else " and "
            return joiner.join(parts)
        joiner = ", y " if lang == "es" else ", and "
        return ", ".join(parts[:-1]) + joiner + parts[-1]

    items_prose = _natural_items(cart.items, lang)

    # Breakdown lines (subtotal, tax, delivery, total)
    if lang == "es":
        breakdown_lines = [f"Subtotal: ${cart.subtotal:.2f}"]
        if cart.tax_total > 0:
            breakdown_lines.append(f"Impuestos: ${cart.tax_total:.2f}")
        if cart.delivery_fee > 0:
            breakdown_lines.append(f"Envío: ${cart.delivery_fee:.2f}")
        breakdown_lines.append(f"Total: ${cart.grand_total:.2f}")
        breakdown = ". ".join(breakdown_lines)
    else:
        breakdown_lines = [f"Subtotal ${cart.subtotal:.2f}"]
        if cart.tax_total > 0:
            breakdown_lines.append(f"tax ${cart.tax_total:.2f}")
        if cart.delivery_fee > 0:
            breakdown_lines.append(f"delivery ${cart.delivery_fee:.2f}")
        breakdown_lines.append(f"total ${cart.grand_total:.2f}")
        breakdown = ", ".join(breakdown_lines)

    if lang == "es":
        if c.service_type == "delivery":
            where = f"para entregar en {c.address}"
        else:
            where = "para recoger en el restaurante"
        return (
            f"Perfecto, {c.full_name.split()[0] if c.full_name else ''}, déjame confirmarte: "
            f"{items_prose}, {where}. {breakdown}. "
            f"¿Está todo bien así?"
        )
    else:
        if c.service_type == "delivery":
            where = f"to be delivered to {c.address}"
        else:
            where = "for pickup at the restaurant"
        return (
            f"Alright {c.full_name.split()[0] if c.full_name else ''}, let me confirm: "
            f"{items_prose}, {where} — {breakdown}. "
            f"Does that all sound right?"
        )


# ─────────────────────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(session: "Session") -> str:
    ctx  = session.tenant
    base = _clean_base_prompt(ctx.system_prompt, ctx.brand_name)
    c    = session.collected
    lang = session.language
    li   = f"IMPORTANT: {LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS['en'])}\n\n"

    if session.status == State.ORDER:
        _, _, hours_note = _build_hours_note(ctx)

        cart_context = ""
        empty_cart_warning = ""
        if not session.cart.is_empty:
            cart_context = (
                f"\n\nCURRENT ORDER IN CART (this is the ONLY source of truth):\n"
                f"{session.cart.to_display(lang)}\n"
                f"The customer can add, remove, or modify items at any time."
            )
        else:
            empty_cart_warning = (
                "\n\n⚠️ THE CART IS CURRENTLY EMPTY. "
                "Until the customer adds menu items, you MUST NOT: "
                "ask for their name, ask for their address, ask about pickup vs delivery, "
                "or pretend an order exists. If they say 'delivery', 'pickup', 'yes', "
                "or give an address, gently redirect them: tell them you first need to "
                "know what they'd like to order from the menu."
            )

        # ─── CRITICAL anti-hallucination rule ───────────────────────────────
        hallu_guard = (
            "\n\n🚫 CRITICAL — NEVER LIE ABOUT THE CART. "
            "The CURRENT ORDER IN CART section above is the ONLY source of truth. "
            "You MUST NOT claim that an item was added, removed, or modified "
            "unless it actually appears (or no longer appears) in that section. "
            "If the system did not add an item the customer requested (because "
            "of a parsing issue), the cart will be empty or unchanged — in that case "
            "you should ASK THE CUSTOMER TO REPHRASE rather than pretend it was added. "
            "Do NOT invent quantities, prices, or items not in the cart section. "
            "Do NOT say 'you already have X from before' unless X is literally listed above."
        )

        # Channel-aware tone instruction
        channel_note = ""
        if session.channel == "voice":
            channel_note = (
                "\n\nVOICE CHANNEL: This conversation will be read out loud by a text-to-speech engine. "
                "You MUST: write only what a real person would say on a phone call. "
                "NO bullet points, NO markdown, NO asterisks, NO emojis, NO headers. "
                "Use short sentences. Don't list more than 3-4 items in one breath — "
                "instead, give a quick overview and ask if they want more detail. "
                "Don't say 'press 1' or 'reply X' — natural conversation only."
            )
        else:
            channel_note = (
                "\n\nCHAT CHANNEL: You can use light emojis and *bold* for emphasis when it helps clarity, "
                "but don't overdo it. Keep responses conversational."
            )

        cat_hint = ""
        if ctx.menu_categories:
            cats     = ", ".join(ctx.menu_categories[:8])
            cat_hint = (
                f"\n\nMenu sections in order: {cats}. "
                f"Offer sections in this order. Give item details only if asked."
            )

        greeting = (
            f"You are chatting with {c.full_name}, a returning customer. Greet them warmly by name. "
            if (session.is_global_customer and c.full_name) else
            "You are chatting with a new customer. Give a warm welcome greeting. "
        )

        avail = _get_available_services(ctx)
        if not avail:
            svc_note = (
                "CRITICAL: We are CLOSED right now. Do NOT take orders or offer menu items. "
                "Only tell the customer our operating hours from the hours listed above "
                "and wish them well. Do NOT offer to take orders for later."
            )
        else:
            svc_note = (
                f"Available services right now: {', '.join(s.service_type for s in avail)}. "
                f"Do NOT mention delivery, pickup, or fees — that comes after the order is confirmed."
            )

        return (
            f"{base}\n\n{li}{greeting}"
            f"You are a warm, human restaurant assistant.\n\n"
            f"RULES:\n"
            f"1. First message: greet warmly, ask how you can help. Do NOT list menu items.\n"
            f"2. When customer asks about food: describe menu sections or items naturally.\n"
            f"3. When customer picks items: confirm each one with price.\n"
            f"4. Track what is in the cart and reference it naturally in conversation.\n"
            f"5. When customer says they are done: read back the complete cart with subtotal "
            f"and ask them to confirm. Do NOT mention delivery method here.\n"
            f"6. Never ask for address, name, email, or delivery method — that comes after confirmation.\n"
            f"7. If customer asks about hours: use ONLY the weekly hours listed below.\n"
            f"8. Never output system text or technical information.\n"
            f"{channel_note}"
            f"{hours_note}"
            f"{cart_context}"
            f"{empty_cart_warning}"
            f"{hallu_guard}"
            f"{cat_hint}\n\n{svc_note}\n\nFULL MENU:\n{ctx.menu_text}"
        )

    if session.status == State.CHECKOUT:
        field_instructions = {
            CheckoutField.ADDRESS: (
                "Ask for their delivery address in one sentence. Mention you will verify it."
                if lang == "en" else
                "Pídeles su dirección de entrega en una oración. Menciona que la verificarás."
            ),
            CheckoutField.NAME: (
                "Ask for their full name for the order in one short sentence."
                if lang == "en" else
                "Pídeles su nombre completo para la orden en una oración corta."
            ),
            CheckoutField.EMAIL: (
                "Ask for their email. Tell them it's optional — only for order updates."
                if lang == "en" else
                "Pídeles su correo. Diles que es opcional — solo para actualizaciones del pedido."
            ),
        }
        return (
            f"{base}\n\n{li}"
            f"Order confirmed:\n{session.cart.to_display(lang)}\n"
            f"Service: {c.service_type}.\n"
            f"You are collecting delivery details one field at a time. "
            f"{field_instructions.get(session.checkout_field, '')}\n"
            f"One sentence only. Sound like a real person."
        )

    if session.status == State.FINAL_CONFIRM:
        addr_line = c.address if c.service_type == "delivery" else "Pickup (no address needed)"
        email_ln  = f"\nEmail: {c.email}" if c.email else ""
        return (
            f"{base}\n\n{li}"
            f"Read back the complete order and ask for final confirmation.\n\n"
            f"Name: {c.full_name}\nService: {c.service_type}\nAddress: {addr_line}"
            f"{email_ln}\n\nOrder:\n{session.cart.to_display(lang)}\n\n"
            f"Sound warm and natural. Ask: does everything look correct?"
        )

    if session.status == State.DONE:
        total = session.cart.grand_total
        return (
            f"{base}\n\n{li}"
            f"Order is placed! Thank {c.full_name or 'the customer'} warmly. "
            f"Give reference number {c.order_number}. Total: ${total:.2f}. "
            f"Two sentences max. Do NOT ask any more questions."
        )

    return base


# ─────────────────────────────────────────────────────────────────────────────
# LLM wrapper
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(llm: LLMProvider, messages: list) -> str:
    return llm.chat(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=SEED)


def _refresh_system_prompt(session: "Session") -> None:
    sp = {"role": "system", "content": build_system_prompt(session)}
    if session.messages and session.messages[0]["role"] == "system":
        session.messages[0] = sp
    else:
        session.messages.insert(0, sp)


_HALLUC_RE    = re.compile(r"\[ORDER_FINALIZED\]|manage_customer_data\s*[>\({].*|<function[_\s].*", re.IGNORECASE | re.DOTALL)
_ORDER_TAG_RE = re.compile(r"ORDER_CONFIRMED:[^.\n]*", re.IGNORECASE)

def strip_hallucinations(txt: str) -> str:
    return _ORDER_TAG_RE.sub("", _HALLUC_RE.sub("", txt)).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_phone = request.from_number

    ctx = db_get_tenant_context(request.to_number)
    if not ctx:
        raise HTTPException(404, "Restaurant not found.")
    if ctx.status != "Active":
        raise HTTPException(403, "Restaurant is currently inactive.")

    session = _sessions.get(user_phone)
    _GREETINGS = {"hello","hi","hola","hey","buenos dias","buenas","good morning",
                  "good afternoon","good evening","start","restart","nuevo","nueva",
                  "buenas tardes","buenas noches"}
    _DONE_PHRASES = {"thank you","thanks","gracias","ty","thx","perfecto","ok gracias",
                     "thank u","muchas gracias","de nada","awesome","great","perfect",
                     "how long","when will","where is my","track","cuanto tiempo",
                     "cuando llega","gracias por todo"}
    msg_clean      = request.message.strip().lower()
    is_greeting    = msg_clean in _GREETINGS
    is_done_phrase = msg_clean in _DONE_PHRASES
    need_new_session = (
        not session
        or (session.status == State.DONE and not is_done_phrase)
        or session.tenant_id != ctx.tenant_id
        or is_greeting
    )

    llm = get_provider(ctx.provider, ctx.api_key, ctx.model_name)

    if need_new_session:
        ctx.timezone        = _resolve_timezone_from_address(llm, ctx)
        _check_service_availability(ctx)
        detected_lang       = _detect_language(request.message, ctx.supported_languages)
        ctx.menu_text       = get_menu_for_language(ctx.tenant_id, detected_lang)
        ctx.menu_categories = get_menu_categories(ctx.tenant_id)
        logger.info(f"[menu] categories={ctx.menu_categories}")
        cust = db_get_customer(user_phone, ctx.tenant_id)
        session = Session(
            session_id         = str(uuid.uuid4()),
            tenant_id          = ctx.tenant_id,
            tenant             = ctx,
            cart               = OrderCart(tenant_default_tax=ctx.default_tax_rate),
            status             = State.ORDER,
            language           = detected_lang,
            channel            = request.channel,
            collected          = CustomerData(
                full_name   = cust["full_name"],
                email       = cust["email"],
                address     = cust["address_line_1"],
                customer_id = cust["customer_id"],
            ),
            is_global_customer = cust["is_global"],
            is_tenant_customer = cust["is_tenant"],
            had_address        = bool(cust["address_line_1"]),
        )
        _sessions[user_phone] = session
        logger.info(f"[session] new={session.session_id} provider={ctx.provider} tz={ctx.timezone} channel={session.channel}")
    else:
        # Update channel if it changed on this request (rare but possible)
        if request.channel and request.channel != session.channel:
            logger.info(f"[session] channel changed {session.channel} → {request.channel}")
            session.channel = request.channel
        _check_service_availability(session.tenant)
        detected = _detect_language(request.message, session.tenant.supported_languages)
        if detected != session.language and detected in session.tenant.supported_languages:
            session.language = detected
            session.tenant.menu_text = get_menu_for_language(ctx.tenant_id, session.language)

    _refresh_system_prompt(session)
    if len(session.messages) > 21:
        session.messages = [session.messages[0]] + session.messages[-20:]
    session.messages.append({"role": "user", "content": request.message})
    final_reply = ""

    # ═══════════════════════════════════════════════════════════════════════
    # FSM
    # ═══════════════════════════════════════════════════════════════════════

    if session.status == State.SERVICE_SELECT:
        avail          = _get_available_services(session.tenant)
        avail_pickup   = next((s for s in avail if s.service_type == "pickup"),   None)
        avail_delivery = next((s for s in avail if s.service_type == "delivery"), None)
        msg_lower      = request.message.strip().lower()
        pickup_kw      = {"pickup","pick up","pick-up","recoger","recojo","voy a recoger","lo recojo"}
        delivery_kw    = {"delivery","deliver","domicilio","a domicilio","entregar","entrega","llevar","traer"}
        chose_pickup   = any(w in msg_lower for w in pickup_kw)
        chose_delivery = any(w in msg_lower for w in delivery_kw)

        avail_order = [s for s in avail if s.service_type in ("pickup","delivery")]
        if not chose_pickup and not chose_delivery and _is_affirmative(request.message):
            if len(avail_order) == 1:
                if avail_order[0].service_type == "delivery":
                    chose_delivery = True
                else:
                    chose_pickup = True

        if chose_pickup and avail_pickup:
            session.collected.service_type = "pickup"
            session.collected.delivery_fee = 0.0
            session.cart.delivery_fee      = 0.0
            session.status                 = State.CHECKOUT
            if not session.collected.full_name:
                session.checkout_field = CheckoutField.NAME
                final_reply = (
                    "¡Perfecto, lo dejamos listo para recoger! ¿A nombre de quién va la orden?"
                    if session.language == "es" else
                    "Perfect, we'll have it ready for pickup. What name should I put on the order?"
                )
            else:
                session.checkout_field = CheckoutField.EMAIL
                first = session.collected.full_name.split()[0]
                final_reply = (
                    f"¡Perfecto, {first}, lo dejamos listo para recoger! "
                    f"Por último, ¿quieres dejarme un correo para mandarte la confirmación? "
                    f"Si prefieres no, está bien también."
                    if session.language == "es" else
                    f"Perfect, {first}, we'll have it ready for pickup. "
                    f"One last thing — want to leave me an email for the order confirmation? "
                    f"If you'd rather not, that's totally fine."
                )
        elif chose_delivery and avail_delivery:
            fee = avail_delivery.fee_amount if avail_delivery.fee_type == "fixed" else 0.0
            session.collected.service_type = "delivery"
            session.collected.delivery_fee = fee
            session.cart.delivery_fee      = fee
            session.status                 = State.CHECKOUT
            session.checkout_field         = CheckoutField.ADDRESS
            if session.language == "es":
                fee_phrase = (f"El envío te cuesta ${fee:.2f}. " if fee > 0
                              else "El envío es gratis. ")
                final_reply = f"¡Va, te lo llevamos! {fee_phrase}¿A qué dirección te lo mando?"
            else:
                fee_phrase = (f"Delivery is ${fee:.2f}. " if fee > 0 else "Delivery is free. ")
                final_reply = f"You got it, we'll bring it to you. {fee_phrase}What's the address?"
        else:
            avail_user = [s for s in avail if s.service_type in ("pickup","delivery")]
            has_p = any(s.service_type == "pickup"   for s in avail_user)
            has_d = any(s.service_type == "delivery" for s in avail_user)
            if session.language == "es":
                if has_p and has_d:
                    final_reply = "¿Vienes por él o te lo llevamos?"
                elif has_p:
                    final_reply = "Solo tenemos pickup ahora — ¿te queda bien recogerlo?"
                elif has_d:
                    final_reply = "Solo tenemos delivery ahora — ¿te lo llevamos?"
                else:
                    final_reply = "Lo siento, no hay servicios disponibles ahora."
            else:
                if has_p and has_d:
                    final_reply = "Would you like to pick it up or have us deliver?"
                elif has_p:
                    final_reply = "We've only got pickup right now — does that work?"
                elif has_d:
                    final_reply = "We've only got delivery right now — does that work?"
                else:
                    final_reply = "Sorry, no services are available right now."

        final_reply = _voice_safe(final_reply, session.channel)

    elif session.status == State.CHECKOUT:
        current_field = session.checkout_field
        intent        = _classify_intent(llm, request.message,
                                         f"Collecting {current_field.value} for food order")
        logger.info(f"[checkout] field={current_field.value} intent={intent}")

        if intent == "back_to_order":
            session.status = State.ORDER
            _refresh_system_prompt(session)
            final_reply = ("¡Claro! " if session.language == "es" else "Of course! ") + strip_hallucinations(call_llm(llm, session.messages))
            final_reply = _voice_safe(final_reply, session.channel)

        elif intent == "cancel":
            session.status = State.ORDER
            session.cart   = OrderCart(tenant_default_tax=session.tenant.default_tax_rate)
            final_reply = (
                "No te preocupes, empecemos de nuevo. ¿Qué se te antoja?"
                if session.language == "es" else
                "No worries, let's start fresh. What are you in the mood for?"
            )
            final_reply = _voice_safe(final_reply, session.channel)

        else:
            if current_field == CheckoutField.ADDRESS:
                result = await validate_address_mapbox(request.message.strip(), session.tenant)
                logger.info(
                    f"[mapbox] valid={result['valid']} "
                    f"relevance={result.get('relevance',0):.2f} "
                    f"out_of_zone={result.get('out_of_zone', False)} "
                    f"distance_km={result.get('distance_km')} "
                    f"canonical='{result.get('canonical','')[:80]}'"
                )

                # Case 1: address found and within delivery zone
                if result["valid"]:
                    session.collected.address           = result["canonical"]
                    session.collected.address_validated = True
                    session.address_attempts            = 0
                    canon = result["canonical"]
                    if not session.collected.full_name:
                        session.checkout_field = CheckoutField.NAME
                        final_reply = (
                            f"Perfecto, anotada — {canon}. ¿A nombre de quién va la orden?"
                            if session.language == "es" else
                            f"Got it — {canon}. And what name should I put on the order?"
                        )
                    else:
                        session.checkout_field = CheckoutField.EMAIL
                        first = session.collected.full_name.split()[0]
                        final_reply = (
                            f"Perfecto, anotada — {canon}. Por último, {first}, "
                            f"¿quieres dejarme un correo para mandarte la confirmación? "
                            f"Si prefieres no, está bien también."
                            if session.language == "es" else
                            f"Got it — {canon}. One last thing, {first}, "
                            f"want to leave me an email for the order confirmation? "
                            f"If you'd rather not, that's totally fine."
                        )

                # Case 2: address found but out of delivery zone
                elif result.get("out_of_zone"):
                    session.address_attempts += 1
                    dist = result.get("distance_km")
                    radius = (
                        session.tenant.services.get("delivery").delivery_radius_km
                        if session.tenant.services.get("delivery") else None
                    )
                    if session.language == "es":
                        final_reply = (
                            f"Ufff, esa dirección está a unos {dist:.0f} km de nosotros "
                            f"y solo entregamos hasta {radius:.0f} km a la redonda. "
                            f"¿Tienes otra dirección más cerca, o prefieres cambiar a pickup?"
                        )
                    else:
                        final_reply = (
                            f"Oof, that address is about {dist:.0f} km from us, and we only "
                            f"deliver within {radius:.0f} km. Do you have an address closer by, "
                            f"or would you rather switch to pickup?"
                        )

                # Case 3: address not found or low relevance
                else:
                    session.address_attempts += 1
                    sugg = result.get("suggestions", [])[:2]
                    if sugg:
                        sugg_text = " o ".join(sugg) if session.language == "es" else " or ".join(sugg)
                        final_reply = (
                            f"Mmm, no encontré esa dirección exacta. ¿Te referías a {sugg_text}? "
                            f"Si no, dímela con un poco más de detalle."
                            if session.language == "es" else
                            f"Hmm, I couldn't find that exact address. Did you mean {sugg_text}? "
                            f"If not, give it to me with a bit more detail."
                        )
                    else:
                        final_reply = (
                            "No la encontré. ¿Me la repites con calle, número y ciudad?"
                            if session.language == "es" else
                            "I couldn't find that one. Could you tell me again with street, number, and city?"
                        )

            elif current_field == CheckoutField.NAME:
                name = _extract_name(request.message)
                session.collected.full_name = name
                session.checkout_field      = CheckoutField.EMAIL
                first = name.split()[0] if name else ""
                final_reply = (
                    f"¡Gracias, {first}! Por último, "
                    f"¿quieres dejarme un correo para la confirmación? "
                    f"Si prefieres no, está bien también."
                    if session.language == "es" else
                    f"Thanks, {first}! One last thing — "
                    f"want to leave me an email for the order confirmation? "
                    f"If you'd rather not, that's totally fine."
                )

            elif current_field == CheckoutField.EMAIL:
                session.collected.email = _extract_email(request.message)
                session.status          = State.FINAL_CONFIRM
                final_reply = _build_final_confirm_summary(session)

        final_reply = _voice_safe(final_reply, session.channel)

    elif session.status == State.FINAL_CONFIRM:
        confirmation = _classify_confirmation(llm, request.message)
        logger.info(f"[final_confirm] result={confirmation}")

        if confirmation == "yes":
            session.status = State.SAVING
            c = session.collected
            try:
                logger.info(f"[save] full_name='{c.full_name}' address='{c.address}'")
                customer_id = db_save_customer(
                    user_phone, ctx.tenant_id, c.full_name or "Unknown",
                    c.address or "Pickup", c.email or None,
                    session.cart.to_summary_string(),
                )
                session.collected.customer_id = customer_id

                order_number = db_save_order(
                    session_id       = session.session_id,
                    tenant_id        = ctx.tenant_id,
                    customer_id      = customer_id,
                    customer_name    = c.full_name or "Unknown",
                    customer_phone   = user_phone,
                    customer_email   = c.email or None,
                    service_type     = c.service_type,
                    delivery_address = c.address or "",
                    cart             = session.cart,
                )
                session.collected.order_number = order_number
                session.status                 = State.DONE

                # ─── DETERMINISTIC closing reply — no LLM ──────────────────
                first_name = c.full_name.split()[0] if c.full_name else (
                    "amigo" if session.language == "es" else "friend"
                )
                total = session.cart.grand_total
                if session.language == "es":
                    if c.service_type == "delivery":
                        when = "Te lo mandamos enseguida."
                    else:
                        when = "Pasa a recogerlo cuando esté listo."
                    final_reply = (
                        f"Listo, {first_name}, ya quedó tu pedido. "
                        f"Tu número de orden es {order_number} y el total es ${total:.2f}. "
                        f"{when} ¡Gracias y provecho!"
                    )
                else:
                    if c.service_type == "delivery":
                        when = "We'll have it over to you shortly."
                    else:
                        when = "Come grab it whenever it's ready."
                    final_reply = (
                        f"All set, {first_name}, your order's in. "
                        f"Your order number is {order_number} and the total is ${total:.2f}. "
                        f"{when} Thanks, enjoy!"
                    )

                # Email notification (fire-and-forget)
                if c.email:
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as http:
                            await http.post(
                                f"{os.getenv('CHAT_API_URL','https://api.albertoescorcia.ca')}/notifications/send-email",
                                json={
                                    "from_email":     os.getenv("FROM_EMAIL","noreply@albertoescorcia.ca"),
                                    "from_name":      os.getenv("FROM_NAME","TenantOS"),
                                    "to":             [{"email": c.email, "name": c.full_name}],
                                    "subject":        f"Your order at {ctx.brand_name} is confirmed!",
                                    "title":          f"Order confirmed, {first_name}!",
                                    "body":           (
                                        f"Hi {first_name},\n\nYour order has been confirmed!\n\n"
                                        f"{session.cart.to_display()}\n\n"
                                        f"Service: {c.service_type}\n"
                                        f"{'Delivery to: ' + c.address + chr(10) if c.service_type == 'delivery' else ''}"
                                        f"Reference: {order_number}\n\n"
                                        f"Thank you for ordering from {ctx.brand_name}!"
                                    ),
                                    "sender_tagline": ctx.brand_name,
                                    "sender_address": ctx.physical_address,
                                }
                            )
                        logger.info(f"[email] sent to {c.email}")
                    except Exception as e:
                        logger.error(f"[email] failed: {e}")

            except Exception as e:
                logger.error(f"[save] error: {e}")
                session.status = State.FINAL_CONFIRM
                final_reply = (
                    "Uy, tuvimos un problemita técnico. ¿Me confirmas otra vez?"
                    if session.language == "es" else
                    "Oh, we had a quick technical hiccup. Could you confirm one more time?"
                )

        elif confirmation == "no":
            session.status = State.ORDER
            session.cart   = OrderCart(tenant_default_tax=session.tenant.default_tax_rate)
            final_reply = (
                "No te preocupes, volvamos al pedido. ¿Qué quieres cambiar?"
                if session.language == "es" else
                "No worries, let's head back to the order. What would you like to change?"
            )
        else:
            # Re-show the summary if customer didn't clearly confirm
            final_reply = _build_final_confirm_summary(session)

        final_reply = _voice_safe(final_reply, session.channel)

    elif session.status == State.ORDER:
        avail = _get_available_services(session.tenant)
        if not avail:
            _refresh_system_prompt(session)
            final_reply = strip_hallucinations(call_llm(llm, session.messages))
            final_reply = _voice_safe(final_reply, session.channel)
        else:
            # ─── GUARDRAIL: cart empty + premature checkout intent ──────────
            if session.cart.is_empty and _looks_like_premature_checkout(request.message):
                logger.info(f"[guardrail] empty cart + premature checkout: '{request.message[:60]}'")
                final_reply = (
                    "¡Por supuesto! Pero primero cuéntame qué te gustaría comer. "
                    "Cuando tengamos tu pedido listo, te pregunto cómo prefieres recibirlo."
                    if session.language == "es" else
                    "Sure thing! But first, tell me what you'd like to order. "
                    "Once we have your food sorted, I'll ask how you'd like to get it."
                )
                final_reply = _voice_safe(final_reply, session.channel)
                session.messages.append({"role": "assistant", "content": final_reply})
                return _response(session, final_reply)

            action = _extract_order_action(
                llm, request.message, session.cart,
                ctx.menu_text, ctx.tenant_id
            )

            if action["action"] == "add":
                for item_data in action.get("items_to_add", []):
                    name = item_data.get("name", "")
                    db_item = lookup_menu_item(ctx.tenant_id, name)
                    if db_item:
                        unit_price = float(db_item["price"])
                        item_code  = db_item["item_code"]
                        name       = db_item["name"]
                        # Lang-specific plural (falls back inside display_name)
                        translated_plural = lookup_translated_plural(
                            ctx.tenant_id, item_code, session.language
                        )
                        plural = translated_plural or (db_item.get("name_plural") or "")
                        is_exempt = bool(db_item.get("is_tax_exempt", False))
                        rate_override = db_item.get("tax_rate_override")
                        rate_override = float(rate_override) if rate_override is not None else None
                        logger.info(
                            f"[cart] DB match: {name} code={item_code} price=${unit_price} "
                            f"plural='{plural}' tax_exempt={is_exempt} override={rate_override}"
                        )
                    else:
                        unit_price = float(item_data.get("unit_price", 0))
                        item_code  = item_data.get("item_code", "")
                        plural     = ""
                        is_exempt  = False
                        rate_override = None
                        logger.warning(f"[cart] item not found in DB: '{name}'")

                    session.cart.add_item(
                        name              = name,
                        quantity          = int(item_data.get("quantity", 1)),
                        unit_price        = unit_price,
                        prep_notes        = item_data.get("prep_notes", ""),
                        item_code         = item_code,
                        name_plural       = plural,
                        is_tax_exempt     = is_exempt,
                        tax_rate_override = rate_override,
                    )

            elif action["action"] == "remove":
                for name in action.get("items_to_remove", []):
                    session.cart.remove_item(name)

            elif action["action"] == "modify":
                for mod in action.get("items_to_modify", []):
                    if mod.get("prep_notes"):
                        session.cart.update_prep_notes(mod["name"], mod["prep_notes"])
                    if mod.get("quantity", 0) > 0:
                        session.cart.update_quantity(mod["name"], mod["quantity"])

            elif action["action"] == "confirm":
                # ─── FIX: explicit response when confirm fires on empty cart ──
                if session.cart.is_empty:
                    logger.info(f"[order] confirm intent but cart is empty")
                    final_reply = (
                        "Aún no has agregado nada al pedido. ¿Qué te apetece?"
                        if session.language == "es" else
                        "You haven't added anything yet. What would you like?"
                    )
                    final_reply = _voice_safe(final_reply, session.channel)
                    session.messages.append({"role": "assistant", "content": final_reply})
                    return _response(session, final_reply)
                # Cart has items — proceed to service selection
                session.status = State.SERVICE_SELECT
                logger.info(f"[order] confirmed subtotal=${session.cart.subtotal} items={len(session.cart.items)}")
                avail_user = [s for s in avail if s.service_type in ("pickup","delivery")]
                # Build natural service prompt
                has_pickup   = any(s.service_type == "pickup"   for s in avail_user)
                has_delivery = any(s.service_type == "delivery" for s in avail_user)
                delivery_svc = next((s for s in avail_user if s.service_type == "delivery"), None)
                fee_amt      = delivery_svc.fee_amount if delivery_svc and delivery_svc.fee_type == "fixed" else 0.0

                if session.language == "es":
                    if has_pickup and has_delivery:
                        fee_txt = f" — el envío te cuesta {('${:.2f}'.format(fee_amt)) if fee_amt > 0 else 'gratis'}" if delivery_svc else ""
                        final_reply = (
                            f"¡Perfecto! ¿Cómo prefieres tu pedido, vienes por él o te lo llevamos?{fee_txt}"
                        )
                    elif has_pickup:
                        final_reply = "¡Perfecto! Por ahora solo tenemos disponible recogerlo. ¿Te queda bien?"
                    elif has_delivery:
                        fee_txt = (f" El envío te cuesta ${fee_amt:.2f}." if fee_amt > 0 else " El envío es gratis.")
                        final_reply = f"¡Perfecto! Por ahora solo tenemos delivery.{fee_txt} ¿Te queda bien?"
                    else:
                        final_reply = "Lo siento, no tenemos servicios disponibles en este momento."
                else:
                    if has_pickup and has_delivery:
                        fee_txt = f" — delivery is {('${:.2f}'.format(fee_amt)) if fee_amt > 0 else 'free'}" if delivery_svc else ""
                        final_reply = (
                            f"Perfect! Would you like to pick it up or have us deliver?{fee_txt}"
                        )
                    elif has_pickup:
                        final_reply = "Perfect! Right now we only have pickup available — does that work for you?"
                    elif has_delivery:
                        fee_txt = (f" Delivery is ${fee_amt:.2f}." if fee_amt > 0 else " Delivery is free.")
                        final_reply = f"Perfect! Right now we only have delivery.{fee_txt} Does that work?"
                    else:
                        final_reply = "Sorry, no services are available right now."

                final_reply = _voice_safe(final_reply, session.channel)
                session.messages.append({"role": "assistant", "content": final_reply})
                return _response(session, final_reply)

            _refresh_system_prompt(session)
            final_reply = strip_hallucinations(call_llm(llm, session.messages))
            if not final_reply:
                final_reply = (
                    "¿Qué se te antoja?" if session.language == "es"
                    else "What are you in the mood for?"
                )
            final_reply = _voice_safe(final_reply, session.channel)

    else:
        c = session.collected
        first = c.full_name.split()[0] if c.full_name else ("amigo" if session.language == "es" else "friend")
        final_reply = (
            f"Tu pedido ya está confirmado, {first}. ¡Gracias otra vez!"
            if session.language == "es" else
            f"Your order's already confirmed, {first}. Thanks again!"
        )
        final_reply = _voice_safe(final_reply, session.channel)

    session.messages.append({"role": "assistant", "content": final_reply})
    return _response(session, final_reply)


# ─────────────────────────────────────────────────────────────────────────────
# Response helper
# ─────────────────────────────────────────────────────────────────────────────

def _response(session: Session, reply: str) -> dict:
    c = session.collected
    return {
        "reply":      reply,
        "session_id": session.session_id,
        "status":     session.status.value,
        "language":   session.language,
        "channel":    session.channel,
        "provider":   session.tenant.provider,
        "cart":       session.cart.to_list(),
        "cart_total": session.cart.grand_total,
        "debug": {
            "checkout_field":  session.checkout_field.value,
            "service_type":    c.service_type,
            "address_valid":   c.address_validated,
            "order_number":    c.order_number,
            "cart_items":      len(session.cart.items),
            "cart_subtotal":   session.cart.subtotal,
            "cart_tax":        session.cart.tax_total,
            "delivery_fee":    session.cart.delivery_fee,
            "collected": {
                "full_name":   c.full_name,
                "address":     c.address,
                "email":       c.email,
                "customer_id": c.customer_id,
            },
            "tenant_tz":     session.tenant.timezone,
            "services_open": [s.service_type for s in _get_available_services(session.tenant)],
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Debug endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.delete("/session/{from_number}")
async def reset_session(from_number: str):
    phone = from_number.replace("-", "+")
    if phone in _sessions:
        del _sessions[phone]
        return {"cleared": True, "from_number": phone}
    return {"cleared": False, "from_number": phone, "reason": "no active session"}


@app.get("/session/{from_number}")
async def get_session(from_number: str):
    phone   = from_number.replace("-", "+")
    session = _sessions.get(phone)
    if not session:
        return {"session": None}
    return {
        "session_id":     session.session_id,
        "status":         session.status.value,
        "language":       session.language,
        "channel":        session.channel,
        "provider":       session.tenant.provider,
        "checkout_field": session.checkout_field.value,
        "service_type":   session.collected.service_type,
        "cart":           session.cart.to_list(),
        "cart_subtotal":  session.cart.subtotal,
        "cart_tax":       session.cart.tax_total,
        "cart_delivery":  session.cart.delivery_fee,
        "cart_total":     session.cart.grand_total,
        "tenant_tax_rate": session.tenant.default_tax_rate,
        "delivery_radius_km": (
            session.tenant.services.get("delivery").delivery_radius_km
            if session.tenant.services.get("delivery") else None
        ),
        "collected": {
            "full_name":    session.collected.full_name,
            "address":      session.collected.address,
            "email":        session.collected.email,
            "order_number": session.collected.order_number,
        },
        "tenant_tz":     session.tenant.timezone,
        "services_open": [s.service_type for s in _get_available_services(session.tenant)],
        "message_count": len(session.messages),
        "last_messages": [
            {"role": m["role"], "content": (m.get("content") or "")[:120]}
            for m in session.messages[-6:]
        ],
    }


@app.get("/debug/tenant")
async def debug_tenant(to_number: str):
    ctx = db_get_tenant_context(to_number)
    if not ctx:
        return JSONResponse({"error": "not found"}, status_code=404)
    _, _, hours_note = _build_hours_note(ctx)
    return {
        "brand_name": ctx.brand_name,
        "model_name": ctx.model_name,
        "provider":   ctx.provider,
        "status":     ctx.status,
        "timezone":   ctx.timezone,
        "languages":  {"primary": ctx.primary_language, "supported": ctx.supported_languages},
        "services":   {
            k: {
                "active":    v.is_active,
                "fee":       v.fee_amount,
                "hours_today": v.hours_by_day.get(_DAYS[datetime.now().weekday()], {}),
            }
            for k, v in ctx.services.items()
        },
        "hours_note_preview": hours_note[:300] if hours_note else "N/A",
    }


@app.delete("/debug/proximity-cache")
async def clear_proximity_cache():
    """Clear cached Mapbox proximity coords (use after updating tenant address)."""
    n = len(_TENANT_PROXIMITY)
    _TENANT_PROXIMITY.clear()
    return {"cleared": n}


@app.get("/debug/classifier")
async def debug_classifier():
    """Returns info about the active classifier LLM (system-level or per-tenant)."""
    return {
        "system_classifier_active": _CLASSIFIER_LLM is not None,
        "provider": CLASSIFIER_PROVIDER if _CLASSIFIER_LLM else "(per-tenant LLM)",
        "model":    CLASSIFIER_MODEL    if _CLASSIFIER_LLM else "(varies)",
        "note": (
            "Set CLASSIFIER_PROVIDER, CLASSIFIER_API_KEY, and CLASSIFIER_MODEL "
            "env vars to enable a system-wide classifier. Recommended: "
            "Groq Llama 3.1 8B for speed & cost."
        ) if not _CLASSIFIER_LLM else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Order management endpoints
# ─────────────────────────────────────────────────────────────────────────────

ORDER_STATUS_FLOW = {
    "pending":          {"confirmed", "cancelled"},
    "confirmed":        {"preparing", "cancelled"},
    "preparing":        {"ready", "cancelled"},
    "ready":            {"out_for_delivery", "delivered", "cancelled"},
    "out_for_delivery": {"delivered", "cancelled"},
    "delivered":        set(),   # terminal
    "cancelled":        set(),   # terminal
}

VALID_PAYMENT_METHODS = {"card", "interac", "e_transfer", "cash"}
VALID_PAYMENT_STATUS  = {"pending", "paid", "failed", "refunded"}


class OrderStatusUpdate(BaseModel):
    status: str   # one of ORDER_STATUS_FLOW keys


class OrderPaymentUpdate(BaseModel):
    payment_method: Optional[str] = None
    payment_status: Optional[str] = None
    mark_paid:      bool          = False   # if True, sets paid_at = NOW()


class OrderCancel(BaseModel):
    reason: Optional[str] = None


def _fetch_order_row(conn, order_number: str) -> Optional[dict]:
    row = conn.execute(text("""
        SELECT id, order_number, tenant_id, customer_id,
               customer_name, customer_phone, customer_email,
               service_type, delivery_address,
               order_date, status,
               subtotal, tax_total, tip_total, delivery_total, grand_total,
               notes, session_id,
               payment_method, payment_status,
               paid_at, prepared_at, out_for_delivery_at, delivered_at,
               cancelled_at, cancellation_reason, created_at
        FROM orders
        WHERE order_number = :num
    """), {"num": order_number}).mappings().first()
    return dict(row) if row else None


def _fetch_order_items(conn, order_id: int) -> list[dict]:
    rows = conn.execute(text("""
        SELECT id, line_type, item_code, item_name,
               quantity, unit_price, line_total, prep_notes,
               is_tax_exempt, tax_rate, tax_amount, created_at
        FROM order_items
        WHERE order_id = :oid
        ORDER BY
            CASE line_type WHEN 'item' THEN 1
                           WHEN 'delivery' THEN 2
                           WHEN 'tax' THEN 3
                           ELSE 4 END,
            id
    """), {"oid": order_id}).mappings().all()
    return [dict(r) for r in rows]


def _coerce_value(v):
    """Convert Decimal → float, datetime/date → ISO string, leave others alone."""
    if v is None:
        return None
    if hasattr(v, "quantize"):
        return float(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _serialize_order(order: dict, items: list[dict]) -> dict:
    return {
        "order_number":        order["order_number"],
        "tenant_id":           order["tenant_id"],
        "customer_id":         str(order["customer_id"]) if order["customer_id"] else None,
        "customer_name":       order["customer_name"],
        "customer_phone":      order["customer_phone"],
        "customer_email":      order["customer_email"],
        "service_type":        order["service_type"],
        "delivery_address":    order["delivery_address"],
        "status":              order["status"],
        "payment_method":      order["payment_method"],
        "payment_status":      order["payment_status"],
        "subtotal":            _coerce_value(order["subtotal"]),
        "tax_total":           _coerce_value(order["tax_total"]),
        "tip_total":           _coerce_value(order["tip_total"]),
        "delivery_total":      _coerce_value(order["delivery_total"]),
        "grand_total":         _coerce_value(order["grand_total"]),
        "notes":               order["notes"],
        "session_id":          order["session_id"],
        "cancellation_reason": order["cancellation_reason"],
        "timestamps": {
            "order_date":          _coerce_value(order["order_date"]),
            "created_at":          _coerce_value(order["created_at"]),
            "paid_at":             _coerce_value(order["paid_at"]),
            "prepared_at":         _coerce_value(order["prepared_at"]),
            "out_for_delivery_at": _coerce_value(order["out_for_delivery_at"]),
            "delivered_at":        _coerce_value(order["delivered_at"]),
            "cancelled_at":        _coerce_value(order["cancelled_at"]),
        },
        "items": [
            {
                "line_type":     it["line_type"],
                "item_code":     it["item_code"],
                "item_name":     it["item_name"],
                "quantity":      _coerce_value(it["quantity"]),
                "unit_price":    _coerce_value(it["unit_price"]),
                "line_total":    _coerce_value(it["line_total"]),
                "prep_notes":    it["prep_notes"],
                "is_tax_exempt": it["is_tax_exempt"],
                "tax_rate":      _coerce_value(it["tax_rate"]),
                "tax_amount":    _coerce_value(it["tax_amount"]),
            }
            for it in items
        ],
    }


@app.get("/orders/{order_number}")
async def get_order(order_number: str):
    """Full detail of a single order."""
    with engine.connect() as conn:
        order = _fetch_order_row(conn, order_number)
        if not order:
            raise HTTPException(404, f"Order {order_number} not found")
        items = _fetch_order_items(conn, order["id"])
    return _serialize_order(order, items)


@app.patch("/orders/{order_number}/status")
async def update_order_status(order_number: str, body: OrderStatusUpdate):
    """
    Move the order through the workflow. Enforces valid transitions:
      pending → confirmed → preparing → ready → out_for_delivery → delivered
      (cancelled reachable from any non-terminal state)
    Automatically stamps the matching timestamp.
    """
    new_status = body.status.strip().lower()
    if new_status not in ORDER_STATUS_FLOW:
        raise HTTPException(400, f"Invalid status '{new_status}'. "
                                 f"Allowed: {sorted(ORDER_STATUS_FLOW.keys())}")

    with engine.begin() as conn:
        order = _fetch_order_row(conn, order_number)
        if not order:
            raise HTTPException(404, f"Order {order_number} not found")

        current = order["status"]
        if new_status == current:
            return {"ok": True, "order_number": order_number, "status": current,
                    "note": "Status already set, no change."}

        allowed = ORDER_STATUS_FLOW.get(current, set())
        if new_status not in allowed:
            raise HTTPException(409,
                f"Invalid transition '{current}' → '{new_status}'. "
                f"From '{current}', allowed next states are: {sorted(allowed) or 'none (terminal)'}.")

        ts_col_by_status = {
            "preparing":        "prepared_at",
            "ready":            "prepared_at",
            "out_for_delivery": "out_for_delivery_at",
            "delivered":        "delivered_at",
            "cancelled":        "cancelled_at",
        }
        ts_col = ts_col_by_status.get(new_status)

        if ts_col:
            sql = f"UPDATE orders SET status = :s, {ts_col} = COALESCE({ts_col}, NOW()) WHERE id = :oid"
        else:
            sql = "UPDATE orders SET status = :s WHERE id = :oid"

        conn.execute(text(sql), {"s": new_status, "oid": order["id"]})

    logger.info(f"[orders] {order_number}: {current} → {new_status}")
    return {"ok": True, "order_number": order_number,
            "previous_status": current, "status": new_status}


@app.patch("/orders/{order_number}/payment")
async def update_order_payment(order_number: str, body: OrderPaymentUpdate):
    """
    Set payment_method and/or payment_status. Enforces:
      - Valid enum values for both fields.
      - Cash is only allowed when service_type='pickup' (also enforced by DB).
      - If `mark_paid` is true OR payment_status='paid', stamps paid_at.
    """
    if body.payment_method is None and body.payment_status is None and not body.mark_paid:
        raise HTTPException(400, "Provide at least one of payment_method, payment_status, or mark_paid.")

    if body.payment_method is not None:
        pm = body.payment_method.strip().lower()
        if pm not in VALID_PAYMENT_METHODS:
            raise HTTPException(400, f"Invalid payment_method '{pm}'. "
                                     f"Allowed: {sorted(VALID_PAYMENT_METHODS)}")
        body.payment_method = pm

    if body.payment_status is not None:
        ps = body.payment_status.strip().lower()
        if ps not in VALID_PAYMENT_STATUS:
            raise HTTPException(400, f"Invalid payment_status '{ps}'. "
                                     f"Allowed: {sorted(VALID_PAYMENT_STATUS)}")
        body.payment_status = ps

    with engine.begin() as conn:
        order = _fetch_order_row(conn, order_number)
        if not order:
            raise HTTPException(404, f"Order {order_number} not found")

        # Pre-check the cash/pickup business rule at the app layer too,
        # so the error message is friendlier than the raw constraint violation.
        new_method = body.payment_method if body.payment_method is not None else order["payment_method"]
        if new_method == "cash" and order["service_type"] != "pickup":
            raise HTTPException(400,
                f"Payment method 'cash' is only allowed for pickup orders. "
                f"This order's service_type is '{order['service_type']}'.")

        sets   = []
        params = {"oid": order["id"]}
        if body.payment_method is not None:
            sets.append("payment_method = :pm")
            params["pm"] = body.payment_method
        if body.payment_status is not None:
            sets.append("payment_status = :ps")
            params["ps"] = body.payment_status
        should_stamp_paid = body.mark_paid or body.payment_status == "paid"
        if should_stamp_paid:
            sets.append("paid_at = COALESCE(paid_at, NOW())")
            if body.payment_status is None:
                sets.append("payment_status = 'paid'")

        sql = f"UPDATE orders SET {', '.join(sets)} WHERE id = :oid"
        conn.execute(text(sql), params)

    logger.info(f"[orders] {order_number} payment updated: "
                f"method={body.payment_method} status={body.payment_status} mark_paid={body.mark_paid}")
    return {"ok": True, "order_number": order_number}


@app.post("/orders/{order_number}/cancel")
async def cancel_order(order_number: str, body: OrderCancel):
    """Convenience endpoint: status='cancelled', stamps cancelled_at, stores reason."""
    with engine.begin() as conn:
        order = _fetch_order_row(conn, order_number)
        if not order:
            raise HTTPException(404, f"Order {order_number} not found")
        if order["status"] in ("delivered", "cancelled"):
            raise HTTPException(409, f"Cannot cancel order in '{order['status']}' state.")

        conn.execute(text("""
            UPDATE orders
            SET status = 'cancelled',
                cancelled_at = COALESCE(cancelled_at, NOW()),
                cancellation_reason = :reason
            WHERE id = :oid
        """), {"reason": body.reason, "oid": order["id"]})

    logger.info(f"[orders] {order_number} cancelled. reason={body.reason!r}")
    return {"ok": True, "order_number": order_number, "status": "cancelled"}


@app.get("/tenants/{tenant_id}/orders")
async def list_tenant_orders(
    tenant_id: str,
    status: Optional[str] = None,
    payment_status: Optional[str] = None,
    service_type: Optional[str] = None,
    date_from: Optional[str] = None,   # ISO date 'YYYY-MM-DD'
    date_to:   Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Filtered list of orders for a tenant. Useful for dashboards.
    Returns header info per order (no line items — use /orders/{number} for detail).
    """
    where  = ["tenant_id = :tid"]
    params = {"tid": tenant_id, "lim": min(max(limit, 1), 500), "off": max(offset, 0)}
    if status:
        where.append("status = :st"); params["st"] = status.strip().lower()
    if payment_status:
        where.append("payment_status = :ps"); params["ps"] = payment_status.strip().lower()
    if service_type:
        where.append("service_type = :svc"); params["svc"] = service_type.strip().lower()
    if date_from:
        where.append("order_date >= :df"); params["df"] = date_from
    if date_to:
        where.append("order_date <= :dt"); params["dt"] = date_to

    sql = f"""
        SELECT order_number, customer_name, service_type, status,
               payment_method, payment_status,
               subtotal, tax_total, delivery_total, grand_total,
               order_date, created_at
        FROM orders
        WHERE {' AND '.join(where)}
        ORDER BY order_date DESC, id DESC
        LIMIT :lim OFFSET :off
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()

    return {
        "tenant_id": tenant_id,
        "count":     len(rows),
        "limit":     params["lim"],
        "offset":    params["off"],
        "orders":    [{k: _coerce_value(v) for k, v in r.items()} for r in rows],
    }


@app.get("/tenants/{tenant_id}/orders/today")
async def today_orders_snapshot(tenant_id: str):
    """Today's operational snapshot: counts by status + revenue totals."""
    with engine.connect() as conn:
        agg = conn.execute(text("""
            SELECT
                COUNT(*)                                                   AS orders_count,
                COUNT(*) FILTER (WHERE status = 'cancelled')               AS cancelled_count,
                COUNT(*) FILTER (WHERE status = 'delivered')               AS delivered_count,
                COUNT(*) FILTER (WHERE status IN ('confirmed','preparing','ready','out_for_delivery')) AS in_progress_count,
                COUNT(*) FILTER (WHERE payment_status = 'paid')            AS paid_count,
                COALESCE(SUM(subtotal),       0)                           AS subtotal_sum,
                COALESCE(SUM(tax_total),      0)                           AS tax_sum,
                COALESCE(SUM(delivery_total), 0)                           AS delivery_sum,
                COALESCE(SUM(grand_total) FILTER (WHERE status <> 'cancelled'), 0) AS revenue_excl_cancelled
            FROM orders
            WHERE tenant_id = :tid AND DATE(order_date) = CURRENT_DATE
        """), {"tid": tenant_id}).mappings().first()

        by_status = conn.execute(text("""
            SELECT status, COUNT(*) AS n
            FROM orders
            WHERE tenant_id = :tid AND DATE(order_date) = CURRENT_DATE
            GROUP BY status
            ORDER BY n DESC
        """), {"tid": tenant_id}).mappings().all()

    return {
        "tenant_id":        tenant_id,
        "date":             datetime.now().strftime("%Y-%m-%d"),
        "summary":          {k: _coerce_value(v) for k, v in agg.items()},
        "orders_by_status": [{"status": r["status"], "count": r["n"]} for r in by_status],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)