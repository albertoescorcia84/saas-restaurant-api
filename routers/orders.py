import re
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger("restaurant_api.orders")

# Matches item patterns like "Half Chicken ($12)", "Half Chicken for $12", or "Salad: $3"
_ITEM_RE = re.compile(r"([^|$\n]+?)\s*[\(\:\-]?\s*\$\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE)
_FILLER_WORDS = {"total", "subtotal", "comes to", "that's", "grand total", "amount", "summary", "here is", "anything else"}


def parse_order_summary(order_summary: str) -> list[dict]:
    """
    Parses an order summary string into structured line items.
    Filters out conversational filler and calculates prices.
    """
    items = []
    # Split by pipe or newline
    segments = [s.strip() for s in re.split(r'[|\n]', order_summary) if s.strip()]

    for seg in segments:
        # Skip questions and AI filler lines
        if "?" in seg:
            continue
        if any(w in seg.lower() for w in _FILLER_WORDS):
            continue
        
        # Extract item name and price
        m = _ITEM_RE.search(seg)
        if m:
            name = m.group(1).strip().strip("(,.-*").strip()
            # Remove trailing words like "for" (e.g., "Half Chicken for")
            name = re.sub(r'(?i)\s+for$', '', name).strip()
            price = float(m.group(2))
            
            if len(name) < 2:
                continue
                
            items.append({"name": name, "price": price})
            
    return items


def _generate_order_number(engine: Engine, tenant_id: str) -> str:
    """Generate readable order number: TEN-YYYYMMDD-XXXX"""
    prefix = tenant_id.replace("-", "")[:3].upper()
    date_str = datetime.now().strftime("%Y%m%d")
    
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) FROM orders
            WHERE tenant_id = :tid
            AND DATE(order_date) = CURRENT_DATE
        """), {"tid": tenant_id}).fetchone()
        
    seq = str((row[0] or 0) + 1).zfill(4)
    return f"{prefix}-{date_str}-{seq}"


def db_save_order(
    engine: Engine,
    session_id: str,
    tenant_id: str,
    customer_id: Optional[str],
    customer_name: str,
    customer_phone: str,
    customer_email: Optional[str],
    service_type: str,
    delivery_address: str,
    order_summary: str,
    delivery_fee: float,
    notes: Optional[str] = None,
) -> tuple[str, str, float]:
    """
    Parses the order summary, saves order and items to the database.
    Returns a tuple of (order_number, cleaned_summary, calculated_subtotal).
    """
    parsed_items = parse_order_summary(order_summary)
    
    calculated_subtotal = sum(item["price"] for item in parsed_items)
    grand_total = calculated_subtotal + delivery_fee
    
    cleaned_summary = " | ".join(f"{item['name']} (${item['price']:.2f})" for item in parsed_items)
    if not cleaned_summary:
        cleaned_summary = order_summary
    
    order_number = _generate_order_number(engine, tenant_id)
    
    with engine.begin() as conn:
        # Insert order header
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
            "name": customer_name, "phone": customer_phone, "email": customer_email or None,
            "stype": service_type, "addr": delivery_address or "",
            "sub": round(calculated_subtotal, 2), "del": round(delivery_fee, 2),
            "grand": round(grand_total, 2), "notes": notes, "sid": session_id,
        }).fetchone()
        order_id = row[0]
        
        # Insert line items
        for item in parsed_items:
            conn.execute(text("""
                INSERT INTO order_items (order_id, line_type, item_name, quantity, unit_price, line_total)
                VALUES (:oid, 'item', :name, 1, :price, :price)
            """), {"oid": order_id, "name": item["name"], "price": round(item["price"], 2)})
            
        if delivery_fee > 0:
            conn.execute(text("""
                INSERT INTO order_items (order_id, line_type, item_name, quantity, unit_price, line_total)
                VALUES (:oid, 'delivery', 'Delivery fee', 1, :fee, :fee)
            """), {"oid": order_id, "fee": round(delivery_fee, 2)})
            
    logger.info(f"[db_save_order] order_number={order_number} grand_total={grand_total}")
    return order_number, cleaned_summary, calculated_subtotal


def db_get_order(engine: Engine, order_number: str) -> Optional[dict]:
    with engine.connect() as conn:
        order = conn.execute(text("SELECT * FROM orders WHERE order_number = :num"), {"num": order_number}).mappings().first()
        if not order:
            return None
        items = conn.execute(text("SELECT * FROM order_items WHERE order_id = :oid ORDER BY id"), {"oid": order["id"]}).mappings().all()
        result = dict(order)
        result["items"] = [dict(i) for i in items]
        return result


def db_get_orders_by_tenant(engine: Engine, tenant_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    with engine.connect() as conn:
        orders = conn.execute(text("""
            SELECT * FROM orders WHERE tenant_id = :tid 
            ORDER BY id DESC LIMIT :limit OFFSET :offset
        """), {"tid": tenant_id, "limit": limit, "offset": offset}).mappings().all()
        return [dict(o) for o in orders]