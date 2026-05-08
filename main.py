import os
import uuid
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="SaaS Restaurant Multi-Tenant API")

# ── LLM Tuning ───────────────────────────────────────────────────────────────
TEMPERATURE    = 0.1   # Near-deterministic
MAX_TOKENS     = 256   # Short chat replies only
SEED           = 42
MAX_TOOL_ITERS = 4

# ── FSM States ───────────────────────────────────────────────────────────────
#
#  STATE_ORDER → (model calls order_ready) → STATE_ASK_ADDRESS
#             → STATE_ASK_NAME → STATE_ASK_EMAIL → STATE_CONFIRM
#             → (customer says YES, model calls save_order) → STATE_DONE
#
STATE_ORDER   = "taking_order"
STATE_ADDR    = "asking_address"
STATE_NAME    = "asking_name"
STATE_EMAIL   = "asking_email"
STATE_CONFIRM = "awaiting_confirmation"
STATE_DONE    = "completed"

# ── Global session store ─────────────────────────────────────────────────────
sessions: dict = {}

# ── Request model ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    to_number:   str   # restaurant phone → tenant lookup key
    from_number: str   # customer phone  → session key
    message:     str

# ── Tool sets — LLaMA only sees the tools relevant to its current state ───────
TOOLS_ORDER = [
    {
        "type": "function",
        "function": {
            "name": "query_menu",
            "description": (
                "Search the restaurant menu for dish names, prices, or ingredients. "
                "Call when the customer asks what is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "order_ready",
            "description": (
                "Call this once the customer has confirmed exactly what they want to order. "
                "Do NOT call speculatively. Do NOT call before the customer confirms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_summary": {
                        "type": "string",
                        "description": "Short summary of confirmed items, e.g. '1x Roast Chicken, 2x Yuca'"
                    }
                },
                "required": ["order_summary"]
            }
        }
    }
]

TOOLS_SAVE = [
    {
        "type": "function",
        "function": {
            "name": "save_order",
            "description": (
                "Save the order and customer data to the database. "
                "Call ONLY after the customer has explicitly said YES to the confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name":     {"type": "string"},
                    "address":       {"type": "string"},
                    "email":         {"type": "string"},
                    "order_summary": {"type": "string"}
                },
                "required": ["full_name", "address", "order_summary"]
            }
        }
    }
]

# ── Prompt builder — one tight prompt per FSM state ──────────────────────────
def build_prompt(state: str, brand: str, base_prompt: str, session: dict) -> str:
    base = base_prompt.format(restaurant_name=brand, menu_context="[use query_menu tool]")
    c    = session["collected"]

    prompts = {
        STATE_ORDER: f"""{base}

{"Returning customer: " + c["full_name"] + ". Greet by name." if session["returning"] else "New customer. Do NOT ask for name or address yet."}

YOUR ONLY JOB NOW:
- Help the customer choose what to order.
- Use query_menu if they ask about dishes, prices, or ingredients.
- Once the customer confirms their items, call order_ready with a short order_summary.
- Do NOT ask for name, address, or email at this stage.
- Ask only ONE thing at a time. Be brief and friendly.""",

        STATE_ADDR: f"""{base}

The customer ordered: {c["order_summary"]}

YOUR ONLY JOB: Ask for their delivery address. One sentence only. Nothing else.""",

        STATE_NAME: f"""{base}

Order: {c["order_summary"]}
Address: {c["address"]}

YOUR ONLY JOB: Ask for the customer's full name. One sentence only.""",

        STATE_EMAIL: f"""{base}

Order: {c["order_summary"]}
Name: {c["full_name"]}
Address: {c["address"]}

YOUR ONLY JOB: Ask for their email address (optional). Tell them they can skip it. One sentence.""",

        STATE_CONFIRM: f"""{base}

All information collected. Write a confirmation message with this data and ask the customer to reply YES or NO:

  Name: {c["full_name"]}
  Delivery address: {c["address"]}
  {"Email: " + c["email"] if c["email"] else ""}
  Order: {c["order_summary"]}

End with: "Is everything correct? Reply YES to confirm or NO to make changes."
Do NOT call any tool. Do NOT save anything yet.""",

        STATE_DONE: f"""{base}

Order saved successfully.
Thank {c["full_name"]} warmly. Confirm their order ({c["order_summary"]}) will be delivered to {c["address"]}. One or two sentences max."""
    }

    return prompts.get(state, base)


# ── Tool executor ─────────────────────────────────────────────────────────────
def execute_tool(name: str, args: dict, session: dict, tenant_id: str, user_phone: str):
    """Returns (result_str, next_state_or_None)."""

    if name == "query_menu":
        query = args.get("query", "")
        logger.info(f"[query_menu] query='{query}'")
        # ── Replace with your real vector DB call ──────────────────────────
        return (
            "MENU:\n- Roast Chicken $20 (gluten-free)\n- Yuca $4\n- Garden Salad $3",
            None
        )

    if name == "order_ready":
        summary = args.get("order_summary", "").strip()
        if not summary:
            return "ERROR: order_summary cannot be empty.", None
        session["collected"]["order_summary"] = summary
        logger.info(f"[order_ready] summary='{summary}'")
        return "Order confirmed internally. Moving to address collection.", STATE_ADDR

    if name == "save_order":
        full_name = args.get("full_name", "").strip()
        address   = args.get("address", "").strip()
        email     = args.get("email", "").strip() or None
        order_sum = args.get("order_summary", "").strip()

        if not full_name or not address:
            return "ERROR: full_name and address are required.", None

        try:
            with engine.begin() as conn:
                row = conn.execute(text("""
                    INSERT INTO customers (phone_number, full_name, email)
                    VALUES (:ph, :name, :em)
                    ON CONFLICT (phone_number)
                    DO UPDATE SET full_name = EXCLUDED.full_name,
                                  email = COALESCE(EXCLUDED.email, customers.email)
                    RETURNING id
                """), {"ph": user_phone, "name": full_name, "em": email}).fetchone()
                cid = row[0]

                tc = conn.execute(text("""
                    INSERT INTO tenant_customers (tenant_id, customer_id)
                    VALUES (:tid, :cid) ON CONFLICT DO NOTHING RETURNING id
                """), {"tid": tenant_id, "cid": cid}).fetchone()

                tc_id = tc[0] if tc else conn.execute(text("""
                    SELECT id FROM tenant_customers
                    WHERE tenant_id = :tid AND customer_id = :cid
                """), {"tid": tenant_id, "cid": cid}).fetchone()[0]

                conn.execute(text("""
                    INSERT INTO tenant_customer_addresses
                        (tenant_customer_id, address_line_1, is_default, city, state, country)
                    VALUES (:tcid, :addr, true, 'Toronto', 'ON', 'Canada')
                    ON CONFLICT (tenant_customer_id, is_default)
                    DO UPDATE SET address_line_1 = EXCLUDED.address_line_1
                """), {"tcid": tc_id, "addr": address})

                # Uncomment if you have an orders table:
                # conn.execute(text("INSERT INTO orders (tenant_customer_id, summary) VALUES (:tcid, :s)"),
                #              {"tcid": tc_id, "s": order_sum})

            logger.info(f"[save_order] customer_id={cid} saved.")
            return "SUCCESS: Order and customer data saved.", STATE_DONE

        except Exception as e:
            logger.error(f"[save_order] DB error: {e}")
            return f"ERROR saving to DB: {e}", None

    return f"ERROR: Unknown tool '{name}'.", None


# ── LLM call helper ───────────────────────────────────────────────────────────
def call_llm(client: Groq, model: str, messages: list, tools=None, force_tool: str = None) -> object:
    kwargs = dict(model=model, messages=messages, temperature=TEMPERATURE,
                  max_tokens=MAX_TOKENS, seed=SEED)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = (
            {"type": "function", "function": {"name": force_tool}}
            if force_tool else "auto"
        )
    return client.chat.completions.create(**kwargs).choices[0].message


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_phone = request.from_number

    try:
        with engine.connect() as conn:

            # 1 — Resolve tenant
            tenant = conn.execute(text("""
                SELECT t.id AS tenant_id, t.brand_name, t.status,
                       s.system_prompt, m.model_name, m.api_key
                FROM tenants t
                JOIN tenant_ai_settings s ON t.id = s.tenant_id
                JOIN llm_models m          ON s.model_id = m.id
                WHERE t.phone_number = :ph
            """), {"ph": request.to_number}).mappings().first()

            if not tenant:
                raise HTTPException(404, "Restaurant not found.")
            if tenant["status"] != "Active":
                raise HTTPException(403, "Restaurant is currently inactive.")

            tenant_id = tenant["tenant_id"]
            client    = Groq(api_key=tenant["api_key"])

            # 2 — Resolve customer
            customer = conn.execute(text("""
                SELECT c.full_name, a.address_line_1
                FROM customers c
                LEFT JOIN tenant_customers tc
                    ON c.id = tc.customer_id AND tc.tenant_id = :tid
                LEFT JOIN tenant_customer_addresses a
                    ON tc.id = a.tenant_customer_id AND a.is_default = true
                WHERE c.phone_number = :ph
            """), {"tid": tenant_id, "ph": user_phone}).mappings().first()

            known_name = customer["full_name"]      if customer else ""
            known_addr = customer["address_line_1"] if customer else ""
            is_returning = bool(known_name)

        # 3 — Init or resume session
        session = sessions.get(user_phone)
        if not session or session["status"] == STATE_DONE:
            sessions[user_phone] = {
                "session_id": str(uuid.uuid4()),
                "status":     STATE_ORDER,
                "returning":  is_returning,
                "collected":  {
                    "full_name":     known_name,
                    "address":       known_addr,
                    "email":         "",
                    "order_summary": "",
                },
                "messages": [],
            }
        session = sessions[user_phone]
        state   = session["status"]

        # 4 — Refresh system prompt and append user message
        def refresh_prompt():
            sp = {"role": "system", "content": build_prompt(
                session["status"], tenant["brand_name"], tenant["system_prompt"], session)}
            if session["messages"] and session["messages"][0]["role"] == "system":
                session["messages"][0] = sp
            else:
                session["messages"].insert(0, sp)

        refresh_prompt()
        session["messages"].append({"role": "user", "content": request.message})
        final_reply = ""

        # ── 5 — FSM dispatcher ────────────────────────────────────────────

        # ── A) Field-collection states: capture input, move to next state ─
        if state in (STATE_ADDR, STATE_NAME, STATE_EMAIL):
            user_input = request.message.strip()

            if state == STATE_ADDR:
                session["collected"]["address"] = user_input
                # Skip name if returning customer already has it
                session["status"] = STATE_NAME if not session["collected"]["full_name"] else STATE_EMAIL

            elif state == STATE_NAME:
                session["collected"]["full_name"] = user_input
                session["status"] = STATE_EMAIL

            elif state == STATE_EMAIL:
                skip_words = {"no", "skip", "none", "n/a", "-", ""}
                if user_input.lower() not in skip_words:
                    session["collected"]["email"] = user_input
                session["status"] = STATE_CONFIRM

            refresh_prompt()
            msg = call_llm(client, tenant["model_name"], session["messages"])
            final_reply = msg.content or ""

        # ── B) Confirmation: detect YES / NO explicitly ───────────────────
        elif state == STATE_CONFIRM:
            YES_WORDS = {"yes","si","sí","yep","yeah","correct","ok","okay",
                         "sure","confirm","confirmed","adelante","procede","dale","claro"}
            confirmed = request.message.strip().lower() in YES_WORDS

            if confirmed:
                c = session["collected"]
                # Force the model to call save_order with all collected data
                save_instruction = (
                    f"Customer confirmed YES. Call save_order immediately with: "
                    f"full_name='{c['full_name']}', address='{c['address']}', "
                    f"email='{c.get('email','')}', order_summary='{c['order_summary']}'."
                )
                session["messages"][-1] = {"role": "user", "content": save_instruction}
                refresh_prompt()

                saved = False
                for _ in range(MAX_TOOL_ITERS):
                    msg = call_llm(client, tenant["model_name"], session["messages"],
                                   tools=TOOLS_SAVE, force_tool="save_order")

                    if not msg.tool_calls:
                        # LLM refused to call tool — execute directly as safety net
                        logger.warning("[confirm] LLM skipped save_order — executing directly.")
                        result, next_st = execute_tool("save_order", c, session, tenant_id, user_phone)
                        if next_st:
                            session["status"] = next_st
                        saved = True
                        break

                    session["messages"].append(msg)
                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments)
                        result, next_st = execute_tool(tc.function.name, args,
                                                       session, tenant_id, user_phone)
                        session["messages"].append({
                            "role": "tool", "tool_call_id": tc.id,
                            "name": tc.function.name, "content": result
                        })
                        if next_st:
                            session["status"] = next_st
                        if "SUCCESS" in result:
                            saved = True

                    if saved:
                        break

                # Generate thank-you message
                refresh_prompt()
                msg2 = call_llm(client, tenant["model_name"], session["messages"])
                final_reply = msg2.content or ""

            else:
                # Customer said NO — restart order
                session["status"] = STATE_ORDER
                session["collected"]["order_summary"] = ""
                refresh_prompt()
                msg = call_llm(client, tenant["model_name"], session["messages"])
                final_reply = msg.content or ""

        # ── C) Order-taking state ─────────────────────────────────────────
        else:  # STATE_ORDER
            for _ in range(MAX_TOOL_ITERS):
                msg = call_llm(client, tenant["model_name"], session["messages"],
                               tools=TOOLS_ORDER)

                if not msg.tool_calls:
                    final_reply = msg.content or ""
                    break

                session["messages"].append(msg)
                transitioned = False

                for tc in msg.tool_calls:
                    args   = json.loads(tc.function.arguments)
                    result, next_st = execute_tool(tc.function.name, args,
                                                   session, tenant_id, user_phone)
                    session["messages"].append({
                        "role": "tool", "tool_call_id": tc.id,
                        "name": tc.function.name, "content": result
                    })
                    if next_st:
                        session["status"] = next_st
                        transitioned = True

                if transitioned:
                    # State is now STATE_ADDR — generate the address question
                    refresh_prompt()
                    msg2 = call_llm(client, tenant["model_name"], session["messages"])
                    final_reply = msg2.content or ""
                    break

        # 6 — Store reply and respond
        session["messages"].append({"role": "assistant", "content": final_reply})

        return {
            "reply":      final_reply,
            "session_id": session["session_id"],
            "status":     session["status"],
            # Remove "collected" before going to production:
            "debug":      session["collected"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error phone={user_phone}: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)