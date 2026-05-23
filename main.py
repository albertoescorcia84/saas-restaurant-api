"""
SaaS Restaurant Multi-Tenant Chat API — v4.2
=============================================
Changes in this version:
  - Provider pattern: supports Groq (LLaMA) and Anthropic (Claude) per tenant
  - Provider loaded from llm_models.provider column in DB
  - Groq and Anthropic clients abstracted behind LLMProvider interface
  - call_llm() simplified to use provider directly
  - _classify_intent() and _classify_confirmation() use provider
  - _resolve_timezone_from_address() uses provider
"""

import os
import re
import uuid
import logging
import httpx
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
app    = FastAPI(title="SaaS Restaurant Multi-Tenant API", version="4.2.0")
app.include_router(notifications_router)

TEMPERATURE       = 0.1
MAX_TOKENS        = 350
SEED              = 42
MAX_TOOL_ITERS    = 5
CLASSIFIER_TOKENS = 80
CLASSIFIER_TEMP   = 0.0


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
# Data models
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ServiceInfo:
    service_type: str
    is_active:    bool
    fee_type:     str
    fee_amount:   float
    open_time:    time
    close_time:   time
    is_open_now:  bool = False


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
    order_items:       list[dict] = field(default_factory=list)
    order_summary:     str = ""
    order_total:       float = 0.0
    service_type:      str = ""
    delivery_fee:      float = 0.0
    order_number:      str = ""
    customer_id:       Optional[str] = None


@dataclass
class Session:
    session_id:         str
    tenant_id:          str
    tenant:             TenantContext
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
# Database
# ─────────────────────────────────────────────────────────────────────────────

def db_get_tenant_context(to_number: str) -> Optional[TenantContext]:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                t.id              AS tenant_id,
                t.brand_name,
                t.status,
                t.physical_address,
                t.city,
                t.state,
                t.country,
                s.system_prompt,
                s.primary_language,
                s.supported_languages,
                m.model_name,
                m.api_key,
                m.provider
            FROM tenants t
            JOIN tenant_ai_settings s ON t.id = s.tenant_id
            JOIN llm_models m          ON s.model_id = m.id
            WHERE t.phone_number = :ph
        """), {"ph": to_number}).mappings().first()

        if not row:
            return None

        svc_rows = conn.execute(text("""
            SELECT service_type, is_active, fee_type, fee_amount, open_time, close_time
            FROM tenant_services
            WHERE tenant_id = :tid
        """), {"tid": row["tenant_id"]}).mappings().all()

    ctx = TenantContext(
        tenant_id           = row["tenant_id"],
        brand_name          = row["brand_name"],
        status              = row["status"],
        system_prompt       = row["system_prompt"],
        model_name          = row["model_name"],
        api_key             = row["api_key"],
        provider            = row["provider"] or "groq",
        timezone            = "America/Toronto",
        physical_address    = row["physical_address"] or "",
        city                = row["city"] or "",
        state               = row["state"] or "",
        country             = row["country"] or "",
        primary_language    = row["primary_language"] or "en",
        supported_languages = [l.strip() for l in (row["supported_languages"] or "en").split(",")],
    )
    for s in svc_rows:
        ctx.services[s["service_type"]] = ServiceInfo(
            service_type = s["service_type"],
            is_active    = s["is_active"],
            fee_type     = s["fee_type"],
            fee_amount   = float(s["fee_amount"] or 0),
            open_time    = s["open_time"],
            close_time   = s["close_time"],
        )
    return ctx


def db_get_customer(from_number: str, tenant_id: str) -> dict:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT c.id AS customer_id, c.full_name, c.email,
                   tc.id AS tc_id, a.address_line_1
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
                SELECT id FROM tenant_customers
                WHERE tenant_id = :tid AND customer_id = :cid
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


def db_save_order(
    session_id:       str,
    tenant_id:        str,
    customer_id:      Optional[str],
    customer_name:    str,
    customer_phone:   str,
    customer_email:   Optional[str],
    service_type:     str,
    delivery_address: str,
    order_summary:    str,
    order_total:      float,
    delivery_fee:     float,
    notes:            Optional[str] = None,
) -> str:
    order_number = _generate_order_number(tenant_id)
    grand_total  = order_total + delivery_fee

    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO orders (
                order_number, tenant_id, customer_id,
                customer_name, customer_phone, customer_email,
                service_type, delivery_address,
                subtotal, tax_total, tip_total,
                delivery_total, grand_total,
                notes, session_id
            ) VALUES (
                :num, :tid, :cid,
                :name, :phone, :email,
                :stype, :addr,
                :sub, 0, 0,
                :del, :grand,
                :notes, :sid
            ) RETURNING id
        """), {
            "num": order_number, "tid": tenant_id, "cid": customer_id,
            "name": customer_name, "phone": customer_phone,
            "email": customer_email or None, "stype": service_type,
            "addr": delivery_address or "",
            "sub": round(order_total, 2), "del": round(delivery_fee, 2),
            "grand": round(grand_total, 2), "notes": notes, "sid": session_id,
        }).fetchone()
        order_id = row[0]

        _item_re = re.compile(r"([^|$\n]+?)\s*[\(\:]?\s*\$(\d+(?:\.\d{1,2})?)", re.IGNORECASE)
        segments = [s.strip() for s in order_summary.split("|") if s.strip()]
        if not segments:
            segments = [order_summary.strip()]

        for seg in segments:
            if "?" in seg:
                continue
            m = _item_re.search(seg)
            if m:
                name  = m.group(1).strip().strip("(,.-").strip()
                price = float(m.group(2))
            else:
                name  = seg.strip()
                price = 0.0
            if not name:
                continue
            conn.execute(text("""
                INSERT INTO order_items (order_id, line_type, item_name, quantity, unit_price, line_total)
                VALUES (:oid, 'item', :name, 1, :price, :price)
            """), {"oid": order_id, "name": name, "price": round(price, 2)})

        if delivery_fee > 0:
            conn.execute(text("""
                INSERT INTO order_items (order_id, line_type, item_name, quantity, unit_price, line_total)
                VALUES (:oid, 'delivery', 'Delivery fee', 1, :fee, :fee)
            """), {"oid": order_id, "fee": round(delivery_fee, 2)})

    logger.info(f"[db_save_order] order_number={order_number} grand_total={grand_total}")
    return order_number


# ─────────────────────────────────────────────────────────────────────────────
# Menu
# ─────────────────────────────────────────────────────────────────────────────

def get_full_menu(tenant_id: str) -> str:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT content FROM menu_vectors WHERE tenant_id = :tid ORDER BY id"
            ), {"tid": tenant_id}).fetchall()
        return "\n".join(r[0] for r in rows) if rows else "Menu information is not available."
    except Exception as e:
        logger.error(f"[menu] DB error: {e}")
        return "Menu information could not be retrieved."


def _extract_menu_categories(menu_text: str) -> list[str]:
    categories = []
    patterns = [
        r"^===\s*(.+?)\s*===",
        r"^#+\s+\*?\*?(.+?)\*?\*?",
        r"^([A-Z][A-Z\s/]+[A-Z])\s*$",
        r"^\*\*([A-Z][A-Z\s/]+)\*\*",
    ]
    for line in menu_text.split("\n"):
        line = line.strip()
        for pat in patterns:
            m = re.match(pat, line)
            if m:
                cat = m.group(1).strip().rstrip("*").strip()
                if len(cat) > 2 and cat not in categories:
                    categories.append(cat)
                break
    return categories


# ─────────────────────────────────────────────────────────────────────────────
# Timezone & Service availability
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_timezone_from_address(llm: LLMProvider, ctx: TenantContext) -> str:
    location = f"{ctx.city}, {ctx.state}, {ctx.country}"
    try:
        tz = llm.classify(
            f"What is the IANA timezone identifier for: {location}? "
            f"Reply with ONLY the timezone string, e.g. America/Toronto. No explanation.",
            max_tokens=30,
        ).strip().strip('"').strip("'")
        if "/" in tz and len(tz) < 50:
            logger.info(f"[tz] {location} → {tz}")
            return tz
    except Exception as e:
        logger.error(f"[tz] error: {e}")
    return "America/Toronto"


def _check_service_availability(ctx: TenantContext) -> None:
    try:
        import zoneinfo
        now = datetime.now(zoneinfo.ZoneInfo(ctx.timezone)).time()
    except Exception:
        now = datetime.utcnow().time()

    for svc in ctx.services.values():
        if not svc.is_active:
            svc.is_open_now = False
        elif svc.close_time > svc.open_time:
            svc.is_open_now = svc.open_time <= now <= svc.close_time
        else:
            svc.is_open_now = now >= svc.open_time or now <= svc.close_time
        logger.info(f"[svc] {svc.service_type}: active={svc.is_active} open={svc.is_open_now} now={now}")


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


# ─────────────────────────────────────────────────────────────────────────────
# Mapbox
# ─────────────────────────────────────────────────────────────────────────────

async def validate_address_mapbox(address: str, city: str = "") -> dict:
    if not MAPBOX_TOKEN:
        return {"valid": True, "canonical": address, "suggestions": []}
    query = f"{address}, {city}".strip(", ")
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json",
                params={
                    "access_token": MAPBOX_TOKEN,
                    "types":        "address",
                    "limit":        3,
                    "language":     "en",
                    "country":      "ca,us",
                    "proximity":    "-79.3832,43.6532",
                }
            )
        features = resp.json().get("features", [])
        if not features:
            return {"valid": False, "canonical": "", "suggestions": []}
        top = features[0]
        return {
            "valid":       top.get("relevance", 0) >= 0.6,
            "canonical":   top.get("place_name", address),
            "suggestions": [f["place_name"] for f in features[:3]],
            "relevance":   top.get("relevance", 0),
        }
    except Exception as e:
        logger.error(f"[mapbox] Error: {e}")
        return {"valid": True, "canonical": address, "suggestions": []}


# ─────────────────────────────────────────────────────────────────────────────
# LLM Classifiers — now use LLMProvider interface
# ─────────────────────────────────────────────────────────────────────────────

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
        return result if result in ("provide_data", "back_to_order", "confirm", "cancel", "other") else "other"
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
        return result if result in ("yes", "no", "other") else "other"
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
    _YES    = {"yes","si","sí","yep","yeah","correct","ok","okay","sure","confirm",
               "confirmed","adelante","procede","dale","claro","yup","perfecto",
               "listo","va","órale","orale","all good","sounds good"}
    if lowered in _YES:
        return True
    _PHRASES = ("yes,","yes.","yes!","si,","si.","sí,","that's correct","looks good",
                "all good","go ahead","todo bien","está bien","esta bien",
                "todo correcto","confirmo","confirmar")
    return any(p in lowered for p in _PHRASES)


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


_PRICE_RE    = re.compile(r"\$[\d]+\.[\d]{2}")
_TOTAL_WORDS = ("total", "subtotal", "comes to", "that's", "grand total", "amount")


def _detect_order_confirmation(llm_reply: str, customer_msg: str, conversation: list) -> tuple[bool, str, float]:
    if not _is_affirmative(customer_msg):
        return False, "", 0.0

    def extract(text: str) -> tuple[list[str], float]:
        lines, total = [], 0.0
        for line in text.split("\n"):
            stripped = line.strip().lstrip("*•-– ").strip()
            if not stripped or len(stripped) < 4:
                continue
            if "?" in stripped:
                continue
            if any(w in stripped.lower() for w in _TOTAL_WORDS):
                continue
            if any(w in stripped.lower() for w in ("would you","anything else","shall i","want to add","how about","add anything")):
                continue
            prices = _PRICE_RE.findall(stripped)
            if prices:
                lines.append(stripped)
                for p in prices:
                    try:
                        total += float(p.replace("$", ""))
                    except ValueError:
                        pass
        return lines, total

    lines, total = extract(llm_reply)
    if lines:
        return True, " | ".join(lines), total

    for msg in reversed(conversation):
        if msg.get("role") == "assistant" and msg.get("content") and "$" in msg["content"]:
            lines, total = extract(msg["content"])
            if lines:
                return True, " | ".join(lines), total
            break

    return False, "", 0.0


# ─────────────────────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────────────────────

_LEGACY_RE = re.compile(
    r"\[ORDER_FINALIZED\]|manage_customer_data\b|query_vector_database\b|STRICT PROTOCOL[\s\S]*",
    re.IGNORECASE,
)

LANG_INSTRUCTIONS = {
    "en": "Respond ONLY in English. Never switch languages. Be warm and conversational.",
    "es": "Responde ÚNICAMENTE en español mexicano. Nunca cambies de idioma. Usa expresiones cálidas como '¡Con gusto!', '¡Claro que sí!'.",
    "fr": "Réponds UNIQUEMENT en français. Ne change jamais de langue. Sois chaleureux et naturel.",
    "bn": "শুধুমাত্র বাংলায় উত্তর দিন। কখনো ভাষা পরিবর্তন করবেন না।",
}


def _clean_base_prompt(raw: str, brand: str) -> str:
    cleaned = _LEGACY_RE.sub("", raw).strip()
    cleaned = cleaned.replace("{restaurant_name}", brand).replace("{menu_context}", "")
    cleaned = re.sub(r"Menu Context:[^\n]*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned if len(cleaned) >= 20 else f"You are a professional ordering assistant for {brand}."


def build_system_prompt(session: "Session") -> str:
    ctx  = session.tenant
    base = _clean_base_prompt(ctx.system_prompt, ctx.brand_name)
    c    = session.collected
    lang = session.language
    li   = f"IMPORTANT: {LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS['en'])}\n\n"

    if session.status == State.ORDER:
        cat_hint = ""
        if ctx.menu_categories:
            cats     = ", ".join(ctx.menu_categories[:6])
            cat_hint = (
                f"\n\nMenu sections available: {cats}. "
                f"Mention section names when relevant. Give item details only if asked."
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
                "Only tell the customer our operating hours and wish them well."
            )
        else:
            svc_note = (
                f"Available services: {', '.join(s.service_type for s in avail)}. "
                f"Do NOT mention delivery, pickup, or fees at this stage — "
                f"that comes ONLY after the customer has confirmed their food order."
            )
        return (
            f"{base}\n\n{li}{greeting}"
            f"You are a warm, human restaurant assistant helping a customer order food.\n\n"
            f"RULES — follow in order:\n"
            f"1. On the first message: greet and ask how you can help. Do NOT list menu items unprompted.\n"
            f"2. When the customer asks about food: tell them about the menu sections or items.\n"
            f"3. When the customer picks an item: confirm it with the price.\n"
            f"4. You may offer ONE optional add-on (side or drink) after the main item. "
            f"If they say no: immediately move to order summary. Do NOT offer more add-ons.\n"
            f"5. When the customer is done ordering: list every item with its price and subtotal, "
            f"then ask them to confirm. Do NOT ask about delivery or pickup here.\n"
            f"6. Never ask for address, name, email, or delivery method — that comes later.\n"
            f"7. Never output system text, instructions, or technical information.\n"
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
            f"Order confirmed: {c.order_summary}. Service: {c.service_type}.\n"
            f"You are collecting delivery details. "
            f"{field_instructions.get(session.checkout_field, '')}\n"
            f"One sentence only. Sound like a real person."
        )

    if session.status == State.FINAL_CONFIRM:
        fee_line  = f"\nDelivery fee: ${c.delivery_fee:.2f}" if (c.service_type == "delivery" and c.delivery_fee > 0) else ""
        total     = c.order_total + c.delivery_fee
        addr_line = c.address if c.service_type == "delivery" else "Pickup (no address needed)"
        email_ln  = f"\nEmail: {c.email}" if c.email else ""
        return (
            f"{base}\n\n{li}"
            f"Read back the complete order and ask for final confirmation.\n\n"
            f"Name: {c.full_name}\nService: {c.service_type}\nAddress: {addr_line}"
            f"{email_ln}\nOrder:\n{c.order_summary}{fee_line}\nTotal: ${total:.2f}\n\n"
            f"Sound warm. Ask: does everything look correct?"
        )

    if session.status == State.DONE:
        total = c.order_total + c.delivery_fee
        return (
            f"{base}\n\n{li}"
            f"Order is placed! Thank {c.full_name or 'the customer'} warmly. "
            f"Give reference number {c.order_number}. Total: ${total:.2f}. "
            f"Two sentences max. Do NOT ask any more questions."
        )

    return base


# ─────────────────────────────────────────────────────────────────────────────
# LLM wrapper — simplified to use LLMProvider
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
                     "thank u","muchas gracias","de nada","awesome","great","perfect"}
    msg_clean      = request.message.strip().lower()
    is_greeting    = msg_clean in _GREETINGS
    is_done_phrase = msg_clean in _DONE_PHRASES
    need_new_session = (
        not session
        or (session.status == State.DONE and not is_done_phrase)
        or session.tenant_id != ctx.tenant_id
        or is_greeting
    )

    # ── Initialize provider from tenant config ─────────────────────────────
    llm = get_provider(ctx.provider, ctx.api_key, ctx.model_name)

    if need_new_session:
        ctx.timezone        = _resolve_timezone_from_address(llm, ctx)
        _check_service_availability(ctx)
        ctx.menu_text       = get_full_menu(ctx.tenant_id)
        ctx.menu_categories = _extract_menu_categories(ctx.menu_text)
        logger.info(f"[menu] categories={ctx.menu_categories}")
        cust = db_get_customer(user_phone, ctx.tenant_id)
        session = Session(
            session_id          = str(uuid.uuid4()),
            tenant_id           = ctx.tenant_id,
            tenant              = ctx,
            status              = State.ORDER,
            language            = _detect_language(request.message, ctx.supported_languages),
            collected           = CustomerData(
                full_name   = cust["full_name"],
                email       = cust["email"],
                address     = cust["address_line_1"],
                customer_id = cust["customer_id"],
            ),
            is_global_customer  = cust["is_global"],
            is_tenant_customer  = cust["is_tenant"],
            had_address         = bool(cust["address_line_1"]),
        )
        _sessions[user_phone] = session
        logger.info(f"[session] new={session.session_id} provider={ctx.provider} tz={ctx.timezone} lang={session.language}")
    else:
        _check_service_availability(session.tenant)
        detected = _detect_language(request.message, session.tenant.supported_languages)
        if detected != session.language and detected in session.tenant.supported_languages:
            session.language = detected

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

        if chose_pickup and avail_pickup:
            session.collected.service_type = "pickup"
            session.collected.delivery_fee = 0.0
            session.status                 = State.CHECKOUT
            session.checkout_field         = CheckoutField.NAME if not session.collected.full_name else CheckoutField.EMAIL
            final_reply = (
                "¡Perfecto, pickup! 🏃 Sin costo de envío. ¿Me puedes dar tu nombre completo para la orden?"
                if session.language == "es" else
                "Perfect, pickup it is! 🏃 No delivery fee. What's your full name for the order?"
            )
        elif chose_delivery and avail_delivery:
            session.collected.service_type = "delivery"
            session.collected.delivery_fee = avail_delivery.fee_amount if avail_delivery.fee_type == "fixed" else 0.0
            session.status                 = State.CHECKOUT
            session.checkout_field         = CheckoutField.ADDRESS
            fee_msg = (
                f"${avail_delivery.fee_amount:.2f}" if avail_delivery.fee_amount > 0
                else ("gratis" if session.language == "es" else "free")
            )
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
                (f"¿Cómo prefieres recibir tu pedido?\n\n{opts}" if opts else
                 "Lo sentimos, no hay servicios disponibles ahora.")
                if session.language == "es" else
                (f"How would you like to receive your order?\n\n{opts}" if opts else
                 "Sorry, no services are available right now.")
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
            session.status                  = State.ORDER
            session.collected.order_summary = ""
            session.collected.order_total   = 0.0
            final_reply = (
                "¡Sin problema! Empecemos de nuevo. ¿Qué te gustaría pedir?"
                if session.language == "es" else
                "No problem! Let's start over. What would you like to order?"
            )

        else:
            if current_field == CheckoutField.ADDRESS:
                result = await validate_address_mapbox(request.message.strip(), session.tenant.city)
                logger.info(f"[mapbox] valid={result['valid']} canonical={result.get('canonical','')}")
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
                            f"Hmm, no pude encontrar esa dirección. ¿Quisiste decir?\n{sugg}\n\nO escríbela completa (número, calle, ciudad)."
                            if session.language == "es" else
                            f"Hmm, I couldn't find that address. Did you mean?\n{sugg}\n\nOr please re-enter it fully (number, street, city)."
                        )
                    else:
                        final_reply = (
                            "No encontré esa dirección. ¿Podrías escribirla completa? (número, calle, ciudad)"
                            if session.language == "es" else
                            "I couldn't find that address. Could you write it out fully? (number, street, city)"
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
                customer_id = db_save_customer(
                    user_phone, ctx.tenant_id, c.full_name,
                    c.address or "Pickup", c.email or None, c.order_summary,
                )
                session.collected.customer_id = customer_id

                order_number = db_save_order(
                    session_id       = session.session_id,
                    tenant_id        = ctx.tenant_id,
                    customer_id      = customer_id,
                    customer_name    = c.full_name,
                    customer_phone   = user_phone,
                    customer_email   = c.email or None,
                    service_type     = c.service_type,
                    delivery_address = c.address or "",
                    order_summary    = c.order_summary,
                    order_total      = c.order_total,
                    delivery_fee     = c.delivery_fee,
                )
                session.collected.order_number = order_number
                session.status                 = State.DONE
                _refresh_system_prompt(session)
                final_reply = strip_hallucinations(call_llm(llm, session.messages))

                # Send confirmation email
                if c.email:
                    try:
                        chat_api_url = os.getenv("CHAT_API_URL", "https://api.albertoescorcia.ca")
                        total        = c.order_total + c.delivery_fee
                        async with httpx.AsyncClient(timeout=10.0) as http:
                            await http.post(
                                f"{chat_api_url}/notifications/send-email",
                                json={
                                    "from_email":     os.getenv("FROM_EMAIL", "noreply@albertoescorcia.ca"),
                                    "from_name":      os.getenv("FROM_NAME", "TenantOS"),
                                    "to":             [{"email": c.email, "name": c.full_name}],
                                    "subject":        f"Your order at {ctx.brand_name} is confirmed!",
                                    "title":          f"Order confirmed, {c.full_name.split()[0]}!",
                                    "body":           (
                                        f"Hi {c.full_name.split()[0]},\n\n"
                                        f"Your order has been confirmed!\n\n"
                                        f"Order: {c.order_summary}\n"
                                        f"Service: {c.service_type}\n"
                                        f"{'Delivery to: ' + c.address + chr(10) if c.service_type == 'delivery' else ''}"
                                        f"{'Delivery fee: $' + f'{c.delivery_fee:.2f}' + chr(10) if c.delivery_fee > 0 else ''}"
                                        f"Total: ${total:.2f}\n\n"
                                        f"Reference: {order_number}\n\n"
                                        f"Thank you for ordering from {ctx.brand_name}!"
                                    ),
                                    "sender_tagline": ctx.brand_name,
                                    "sender_address": ctx.physical_address,
                                }
                            )
                        logger.info(f"[email] confirmation sent to {c.email}")
                    except Exception as e:
                        logger.error(f"[email] failed: {e}")

                # Safety net closing message
                total = c.order_total + c.delivery_fee
                closing_bad = ("correct", "correcto", "look good", "everything", "todo", "?")
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
            session.status                  = State.ORDER
            session.collected.order_summary = ""
            session.collected.order_total   = 0.0
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
            for _ in range(MAX_TOOL_ITERS):
                raw_reply = call_llm(llm, session.messages)
                if not raw_reply:
                    break
                order_confirmed, order_summary, order_total = _detect_order_confirmation(
                    raw_reply, request.message, session.messages
                )
                final_reply = strip_hallucinations(raw_reply)

                if order_confirmed and order_summary:
                    session.collected.order_summary = order_summary
                    session.collected.order_total   = order_total
                    session.status                  = State.SERVICE_SELECT
                    logger.info(f"[order] confirmed total=${order_total}")
                    avail_user = [s for s in avail if s.service_type in ("pickup","delivery")]
                    opts       = _format_service_options(avail_user, session.language)
                    final_reply = (
                        (f"¡Perfecto! 🎉 ¿Cómo prefieres recibir tu pedido?\n\n{opts}" if opts else
                         "Lo sentimos, no hay pickup ni delivery disponibles ahora.")
                        if session.language == "es" else
                        (f"Perfect! 🎉 How would you like to receive your order?\n\n{opts}" if opts else
                         "Sorry, pickup and delivery are not available right now.")
                    )
                    session.messages.append({"role": "assistant", "content": final_reply})
                    return _response(session, final_reply)
                break

            if not final_reply:
                final_reply = (
                    "¿Qué te gustaría ordenar?" if session.language == "es"
                    else "What would you like to order?"
                )
    else:
        c = session.collected
        final_reply = (
            f"Tu pedido ya fue confirmado, {c.full_name or ''}. ¡Gracias!"
            if session.language == "es" else
            f"Your order is already confirmed, {c.full_name or ''}. Thank you!"
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
        "debug": {
            "checkout_field": session.checkout_field.value,
            "service_type":   c.service_type,
            "address_valid":  c.address_validated,
            "order_total":    c.order_total,
            "delivery_fee":   c.delivery_fee,
            "order_number":   c.order_number,
            "collected": {
                "full_name":     c.full_name,
                "address":       c.address,
                "email":         c.email,
                "order_summary": c.order_summary,
                "customer_id":   c.customer_id,
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
        "collected": {
            "full_name":     session.collected.full_name,
            "address":       session.collected.address,
            "email":         session.collected.email,
            "order_summary": session.collected.order_summary,
            "order_total":   session.collected.order_total,
            "delivery_fee":  session.collected.delivery_fee,
            "order_number":  session.collected.order_number,
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
    return {
        "brand_name": ctx.brand_name,
        "model_name": ctx.model_name,
        "provider":   ctx.provider,
        "status":     ctx.status,
        "timezone":   ctx.timezone,
        "languages":  {"primary": ctx.primary_language, "supported": ctx.supported_languages},
        "services":   {k: {"active": v.is_active, "fee": v.fee_amount,
                           "open": str(v.open_time), "close": str(v.close_time)}
                       for k, v in ctx.services.items()},
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)