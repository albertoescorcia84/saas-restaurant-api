"""
SaaS Restaurant Multi-Tenant Chat API — v4.6
=============================================
Changes from v4.5:
  - FIX: Cart-empty guardrail in STATE.ORDER — intercepts "delivery"/"pickup"/
    "yes"/address-like messages BEFORE calling the LLM, redirecting customer
    to order food first. Prevents LLM from improvising checkout questions.
  - FIX: "confirm" action with empty cart now responds explicitly instead of
    silently falling through to the LLM.
  - FIX: Mapbox now uses tenant's city/state/country dynamically, with
    geocoded proximity cached per-tenant. Threshold lowered to 0.5.
  - System prompt for ORDER state has stronger empty-cart rules.
"""

import os
import re
import uuid
import json
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

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
app    = FastAPI(title="SaaS Restaurant Multi-Tenant API", version="4.6.0")
app.include_router(notifications_router)

TEMPERATURE       = 0.1
MAX_TOKENS        = 400
SEED              = 42
CLASSIFIER_TOKENS = 200
CLASSIFIER_TEMP   = 0.0
MAPBOX_THRESHOLD  = 0.5

_DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
_DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# Cache of geocoded tenant proximity coords: tenant_id -> "lng,lat"
_TENANT_PROXIMITY: dict[str, str] = {}


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
    name:       str
    quantity:   int
    unit_price: float
    prep_notes: str = ""
    item_code:  str = ""

    @property
    def line_total(self) -> float:
        return round(self.quantity * self.unit_price, 2)

    def to_display(self) -> str:
        note = f" ({self.prep_notes})" if self.prep_notes else ""
        qty  = f"{self.quantity}x " if self.quantity > 1 else ""
        return f"{qty}{self.name}{note} — ${self.line_total:.2f}"

    def to_dict(self) -> dict:
        return {
            "item_code":  self.item_code,
            "name":       self.name,
            "quantity":   self.quantity,
            "unit_price": self.unit_price,
            "prep_notes": self.prep_notes,
            "line_total": self.line_total,
        }


@dataclass
class OrderCart:
    items:        list[OrderItem] = field(default_factory=list)
    tax_rate:     float           = 0.0
    delivery_fee: float           = 0.0

    @property
    def subtotal(self) -> float:
        return round(sum(i.line_total for i in self.items), 2)

    @property
    def tax_total(self) -> float:
        return round(self.subtotal * self.tax_rate, 2)

    @property
    def grand_total(self) -> float:
        return round(self.subtotal + self.tax_total + self.delivery_fee, 2)

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    def add_item(self, name: str, quantity: int, unit_price: float,
                 prep_notes: str = "", item_code: str = "") -> None:
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
            name=name, quantity=quantity, unit_price=unit_price,
            prep_notes=prep_notes, item_code=item_code
        ))
        logger.info(f"[cart] added {name} x{quantity} @ ${unit_price} code={item_code}")

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
        lines = [i.to_display() for i in self.items]
        lines.append(f"\nSubtotal: ${self.subtotal:.2f}")
        if self.tax_total > 0:
            lines.append(f"Tax ({self.tax_rate*100:.0f}%): ${self.tax_total:.2f}")
        if self.delivery_fee > 0:
            lines.append(f"Delivery: ${self.delivery_fee:.2f}")
        lines.append(f"Total: ${self.grand_total:.2f}")
        return "\n".join(lines)

    def to_list(self) -> list[dict]:
        return [i.to_dict() for i in self.items]


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ServiceInfo:
    service_type: str
    is_active:    bool
    fee_type:     str
    fee_amount:   float
    is_open_now:  bool = False
    hours_by_day: dict = field(default_factory=dict)


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
    provider:            str = "groq"
    primary_language:    str = "en"
    supported_languages: list[str] = field(default_factory=lambda: ["en"])
    services:            dict[str, ServiceInfo] = field(default_factory=dict)
    menu_text:           str = ""
    menu_categories:     list[str] = field(default_factory=list)


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
    is_global_customer: bool          = False
    is_tenant_customer: bool          = False
    had_address:        bool          = False
    address_attempts:   int           = 0


_sessions: dict[str, Session] = {}


class ChatRequest(BaseModel):
    to_number:   str
    from_number: str
    message:     str


# ─────────────────────────────────────────────────────────────────────────────
# Database — Tenant
# ─────────────────────────────────────────────────────────────────────────────

def db_get_tenant_context(to_number: str) -> Optional[TenantContext]:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                t.id, t.brand_name, t.status,
                t.physical_address, t.city, t.state, t.country, t.timezone,
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
    )
    for s in svc_rows:
        hours = {
            day: {"open": s[f"{day}_open"], "close": s[f"{day}_close"]}
            for day in _DAYS
        }
        ctx.services[s["service_type"]] = ServiceInfo(
            service_type = s["service_type"],
            is_active    = s["is_active"],
            fee_type     = s["fee_type"],
            fee_amount   = float(s["fee_amount"] or 0),
            hours_by_day = hours,
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
    return {
        "is_global":      True,
        "is_tenant":      row["tc_id"] is not None,
        "full_name":      row["full_name"] or "",
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
    order_number = _generate_order_number(tenant_id)
    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO orders (
                order_number, tenant_id, customer_id,
                customer_name, customer_phone, customer_email,
                service_type, delivery_address,
                subtotal, tax_total, tip_total, delivery_total, grand_total,
                notes, session_id
            ) VALUES (
                :num, :tid, :cid, :name, :phone, :email,
                :stype, :addr, :sub, :tax, 0, :del, :grand, :notes, :sid
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

        for item in cart.items:
            conn.execute(text("""
                INSERT INTO order_items
                    (order_id, line_type, item_code, item_name, quantity, unit_price, line_total, prep_notes)
                VALUES (:oid, 'item', :code, :name, :qty, :price, :total, :notes)
            """), {
                "oid": order_id, "code": item.item_code or None,
                "name": item.name, "qty": item.quantity,
                "price": item.unit_price, "total": item.line_total,
                "notes": item.prep_notes or None,
            })

        if cart.delivery_fee > 0:
            conn.execute(text("""
                INSERT INTO order_items (order_id, line_type, item_name, quantity, unit_price, line_total)
                VALUES (:oid, 'delivery', 'Delivery fee', 1, :fee, :fee)
            """), {"oid": order_id, "fee": cart.delivery_fee})

        if cart.tax_total > 0:
            conn.execute(text("""
                INSERT INTO order_items (order_id, line_type, item_name, quantity, unit_price, line_total)
                VALUES (:oid, 'tax', 'Tax', 1, :tax, :tax)
            """), {"oid": order_id, "tax": cart.tax_total})

    logger.info(f"[db_save_order] order_number={order_number} grand_total={cart.grand_total}")
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
                SELECT item_code, name, price
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
    location = f"{ctx.city}, {ctx.state}, {ctx.country}"
    try:
        tz = llm.classify(
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
    # Default: tenant country + neighbours for cross-border deliveries
    if code == "ca": return "ca,us"
    if code == "us": return "us,ca"
    if code: return code
    return "ca,us"  # safe default


async def validate_address_mapbox(address: str, ctx: TenantContext) -> dict:
    """
    Validates an address using Mapbox geocoding, biased to the tenant's location.
    Threshold is MAPBOX_THRESHOLD (currently 0.5).
    """
    if not MAPBOX_TOKEN:
        return {"valid": True, "canonical": address, "suggestions": [], "relevance": 1.0}

    # Build query with tenant city/state for better matching
    parts = [address]
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
            return {"valid": False, "canonical": "", "suggestions": [], "relevance": 0.0}

        top = features[0]
        relevance = top.get("relevance", 0)
        logger.info(f"[mapbox] top relevance={relevance} place={top.get('place_name','')[:80]}")

        return {
            "valid":       relevance >= MAPBOX_THRESHOLD,
            "canonical":   top.get("place_name", address),
            "suggestions": [f["place_name"] for f in features[:3]],
            "relevance":   relevance,
        }
    except Exception as e:
        logger.error(f"[mapbox] Error: {e}")
        return {"valid": True, "canonical": address, "suggestions": [], "relevance": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# LLM Classifiers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_order_action(llm: LLMProvider, message: str, cart: OrderCart,
                          menu_text: str, tenant_id: str) -> dict:
    cart_display = cart.to_display() if not cart.is_empty else "Empty cart"
    prompt = f"""You are an order action extractor for a restaurant.

Current cart:
{cart_display}

Menu (use EXACT item codes and names):
{menu_text[:3000]}

Customer message: "{message}"

Reply with JSON ONLY. No explanation, no markdown.

{{
  "action": "add|remove|modify|confirm|unclear|inquiry",
  "items_to_add": [
    {{"item_code": "e.g. PUP-001", "name": "exact item name", "quantity": 1, "prep_notes": "e.g. no sauce"}}
  ],
  "items_to_remove": ["exact item name"],
  "items_to_modify": [
    {{"name": "exact item name", "prep_notes": "new notes", "quantity": 0}}
  ]
}}

Rules:
- action "confirm" = customer confirms order (yes/ok/correct/that's everything)
- action "inquiry" = question about menu, hours, ingredients
- action "unclear" = cannot determine
- quantity 0 in modify = keep existing
"""
    result = ""
    try:
        result = llm.classify(prompt, max_tokens=CLASSIFIER_TOKENS)
        result = re.sub(r"```json|```", "", result).strip()
        json_match = re.search(r"\{.*\}", result, re.DOTALL)
        if json_match:
            result = json_match.group(0)
        data = json.loads(result)
        logger.info(f"[cart_action] action={data.get('action')} add={len(data.get('items_to_add',[]))}")
        return data
    except Exception as e:
        logger.error(f"[extract_order_action] error: {e} raw={result[:200] if result else 'N/A'}")
        return {"action": "unclear", "items_to_add": [], "items_to_remove": [], "items_to_modify": []}


def _classify_intent(llm: LLMProvider, message: str, context: str) -> str:
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
        result = llm.classify(prompt).split()[0]
        return result if result in ("provide_data","back_to_order","confirm","cancel","other") else "other"
    except Exception as e:
        logger.error(f"[classify_intent] error: {e}")
        return "other"


def _classify_confirmation(llm: LLMProvider, message: str) -> str:
    prompt = (
        f"Is this message a confirmation (yes), rejection (no), or something else?\n"
        f"Message: \"{message}\"\n"
        f"sí, dale, claro, listo, perfecto, ok, correcto, yes, sure = yes\n"
        f"no, nope, cancel, incorrecto, cambiar = no\n"
        f"Reply with ONE word: yes, no, or other."
    )
    try:
        result = llm.classify(prompt).split()[0]
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
    return stripped.strip().title()


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
                f"\n\nCURRENT ORDER IN CART:\n{session.cart.to_display(lang)}\n"
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
            f"{hours_note}"
            f"{cart_context}"
            f"{empty_cart_warning}"
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
            cart               = OrderCart(),
            status             = State.ORDER,
            language           = detected_lang,
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
        logger.info(f"[session] new={session.session_id} provider={ctx.provider} tz={ctx.timezone}")
    else:
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
            session.checkout_field         = CheckoutField.NAME if not session.collected.full_name else CheckoutField.EMAIL
            final_reply = (
                "¡Perfecto, pickup! 🏃 Sin costo de envío. ¿Me puedes dar tu nombre completo?"
                if session.language == "es" else
                "Perfect, pickup it is! 🏃 No delivery fee. What's your full name for the order?"
            )
        elif chose_delivery and avail_delivery:
            fee = avail_delivery.fee_amount if avail_delivery.fee_type == "fixed" else 0.0
            session.collected.service_type = "delivery"
            session.collected.delivery_fee = fee
            session.cart.delivery_fee      = fee
            session.status                 = State.CHECKOUT
            session.checkout_field         = CheckoutField.ADDRESS
            fee_msg = f"${fee:.2f}" if fee > 0 else ("gratis" if session.language == "es" else "free")
            final_reply = (
                f"¡Delivery! 🛵 Costo de envío: {fee_msg}. ¿Cuál es tu dirección de entrega?"
                if session.language == "es" else
                f"Delivery it is! 🛵 Delivery fee: {fee_msg}. What's your delivery address?"
            )
        else:
            opts = _format_service_options(
                [s for s in avail if s.service_type in ("pickup","delivery")], session.language
            )
            final_reply = (
                (f"¿Cómo prefieres recibir tu pedido?\n\n{opts}" if opts
                 else "Lo sentimos, no hay servicios disponibles ahora.")
                if session.language == "es" else
                (f"How would you like to receive your order?\n\n{opts}" if opts
                 else "Sorry, no services are available right now.")
            )

    elif session.status == State.CHECKOUT:
        current_field = session.checkout_field
        intent        = _classify_intent(llm, request.message,
                                         f"Collecting {current_field.value} for food order")
        logger.info(f"[checkout] field={current_field.value} intent={intent}")

        if intent == "back_to_order":
            session.status = State.ORDER
            _refresh_system_prompt(session)
            final_reply = ("¡Claro! " if session.language == "es" else "Of course! ") + strip_hallucinations(call_llm(llm, session.messages))

        elif intent == "cancel":
            session.status = State.ORDER
            session.cart   = OrderCart()
            final_reply = (
                "¡Sin problema! Empecemos de nuevo. ¿Qué te gustaría pedir?"
                if session.language == "es" else
                "No problem! Let's start over. What would you like to order?"
            )

        else:
            if current_field == CheckoutField.ADDRESS:
                result = await validate_address_mapbox(request.message.strip(), session.tenant)
                logger.info(
                    f"[mapbox] valid={result['valid']} "
                    f"relevance={result.get('relevance',0):.2f} "
                    f"canonical='{result.get('canonical','')[:80]}'"
                )
                if result["valid"]:
                    session.collected.address           = result["canonical"]
                    session.collected.address_validated = True
                    session.address_attempts            = 0
                    session.checkout_field              = CheckoutField.NAME if not session.collected.full_name else CheckoutField.EMAIL
                    _refresh_system_prompt(session)
                    final_reply = strip_hallucinations(call_llm(llm, session.messages))
                else:
                    session.address_attempts += 1
                    sugg = "\n".join(f"• {s}" for s in result.get("suggestions",[])[:2])
                    if sugg:
                        final_reply = (
                            f"Hmm, no pude encontrar esa dirección. ¿Quisiste decir?\n{sugg}\n\nO escríbela completa."
                            if session.language == "es" else
                            f"Hmm, I couldn't find that address. Did you mean?\n{sugg}\n\nOr please re-enter it fully."
                        )
                    else:
                        final_reply = (
                            "No encontré esa dirección. ¿Podrías escribirla completa?"
                            if session.language == "es" else
                            "I couldn't find that address. Could you write it out fully?"
                        )

            elif current_field == CheckoutField.NAME:
                session.collected.full_name = _extract_name(request.message)
                session.checkout_field      = CheckoutField.EMAIL
                _refresh_system_prompt(session)
                final_reply = strip_hallucinations(call_llm(llm, session.messages))

            elif current_field == CheckoutField.EMAIL:
                session.collected.email = _extract_email(request.message)
                session.status          = State.FINAL_CONFIRM
                _refresh_system_prompt(session)
                final_reply = strip_hallucinations(call_llm(llm, session.messages))

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
                _refresh_system_prompt(session)
                final_reply = strip_hallucinations(call_llm(llm, session.messages))

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
                                    "title":          f"Order confirmed, {c.full_name.split()[0]}!",
                                    "body":           (
                                        f"Hi {c.full_name.split()[0]},\n\nYour order has been confirmed!\n\n"
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

                total = session.cart.grand_total
                closing_bad = ("correct","correcto","look good","everything","todo","?")
                if not final_reply or any(t in final_reply.lower() for t in closing_bad):
                    final_reply = (
                        f"¡Listo, {c.full_name}! Tu pedido está confirmado. "
                        f"Orden: {order_number}. Total: ${total:.2f}. ¡Gracias y buen provecho!"
                        if session.language == "es" else
                        f"You're all set, {c.full_name}! Order confirmed. "
                        f"Order: {order_number}. Total: ${total:.2f}. Thank you and enjoy your meal!"
                    )

            except Exception as e:
                logger.error(f"[save] error: {e}")
                session.status = State.FINAL_CONFIRM
                final_reply = (
                    "Hubo un problema técnico. ¿Puedes confirmar nuevamente?"
                    if session.language == "es" else
                    "There was a technical issue. Could you confirm again?"
                )

        elif confirmation == "no":
            session.status = State.ORDER
            session.cart   = OrderCart()
            final_reply = (
                "¡Sin problema! Regresemos al pedido. ¿Qué cambios quieres hacer?"
                if session.language == "es" else
                "No problem! Let's go back. What would you like to change?"
            )
        else:
            _refresh_system_prompt(session)
            final_reply = strip_hallucinations(call_llm(llm, session.messages))

    elif session.status == State.ORDER:
        avail = _get_available_services(session.tenant)
        if not avail:
            _refresh_system_prompt(session)
            final_reply = strip_hallucinations(call_llm(llm, session.messages))
        else:
            # ─── GUARDRAIL: cart empty + premature checkout intent ──────────
            # Customer said "delivery"/"pickup"/an address before ordering.
            # Intercept BEFORE calling the LLM so it can't improvise.
            if session.cart.is_empty and _looks_like_premature_checkout(request.message):
                logger.info(f"[guardrail] empty cart + premature checkout: '{request.message[:60]}'")
                final_reply = (
                    "¡Claro! Pero primero dime qué te gustaría ordenar de nuestro menú. 😊 "
                    "Una vez que tengamos tu pedido, te pregunto si lo prefieres delivery o pickup."
                    if session.language == "es" else
                    "Of course! But first, let me know what you'd like to order from our menu. 😊 "
                    "Once we have your items, I'll ask about delivery or pickup."
                )
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
                        logger.info(f"[cart] DB match: {name} code={item_code} price=${unit_price}")
                    else:
                        unit_price = float(item_data.get("unit_price", 0))
                        item_code  = item_data.get("item_code", "")
                        logger.warning(f"[cart] item not found in DB: '{name}'")

                    session.cart.add_item(
                        name       = name,
                        quantity   = int(item_data.get("quantity", 1)),
                        unit_price = unit_price,
                        prep_notes = item_data.get("prep_notes", ""),
                        item_code  = item_code,
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
                        "Tu carrito está vacío todavía. ¿Qué te gustaría ordenar?"
                        if session.language == "es" else
                        "Your cart is still empty. What would you like to order?"
                    )
                    session.messages.append({"role": "assistant", "content": final_reply})
                    return _response(session, final_reply)
                # Cart has items — proceed to service selection
                session.status = State.SERVICE_SELECT
                logger.info(f"[order] confirmed subtotal=${session.cart.subtotal} items={len(session.cart.items)}")
                avail_user = [s for s in avail if s.service_type in ("pickup","delivery")]
                opts       = _format_service_options(avail_user, session.language)
                final_reply = (
                    (f"¡Perfecto! 🎉 ¿Cómo prefieres recibir tu pedido?\n\n{opts}" if opts
                     else "Lo sentimos, no hay pickup ni delivery disponibles ahora.")
                    if session.language == "es" else
                    (f"Perfect! 🎉 How would you like to receive your order?\n\n{opts}" if opts
                     else "Sorry, pickup and delivery are not available right now.")
                )
                session.messages.append({"role": "assistant", "content": final_reply})
                return _response(session, final_reply)

            _refresh_system_prompt(session)
            final_reply = strip_hallucinations(call_llm(llm, session.messages))
            if not final_reply:
                final_reply = (
                    "¿Qué te gustaría ordenar?" if session.language == "es"
                    else "What would you like to order?"
                )

    else:
        c = session.collected
        final_reply = (
            f"Tu pedido ya fue confirmado, {c.full_name or 'amigo'}. ¡Gracias!"
            if session.language == "es" else
            f"Your order is already confirmed, {c.full_name or 'friend'}. Thank you!"
        )

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
        "provider":       session.tenant.provider,
        "checkout_field": session.checkout_field.value,
        "service_type":   session.collected.service_type,
        "cart":           session.cart.to_list(),
        "cart_subtotal":  session.cart.subtotal,
        "cart_total":     session.cart.grand_total,
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)