import os
import uuid
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from typing import Optional

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="SaaS Restaurant Multi-Tenant API")

# ==========================================
# CONSTANTS — Tune these for LLaMA behavior
# ==========================================
TEMPERATURE = 0.2        # Low = deterministic, fewer hallucinations on tool calls
MAX_TOKENS = 512         # Keep responses concise
SEED = 42                # Reproducibility across identical inputs
MAX_TOOL_ITERATIONS = 5  # Safety limit to avoid infinite tool loops

# ==========================================
# IN-MEMORY SESSION STORE
# ==========================================
sessions: dict = {}

# ==========================================
# DATA MODELS
# ==========================================
class ChatRequest(BaseModel):
    to_number: str    # Restaurant's phone number (tenant identifier)
    from_number: str  # Customer's phone number
    message: str      # Incoming customer message

# ==========================================
# TOOL DEFINITIONS
# Descriptions are written for LLaMA 3:
# short, imperative, no ambiguity.
# ==========================================
GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_menu",
            "description": (
                "Search the restaurant menu. "
                "Call this ONLY for questions about dishes, prices, ingredients, or availability. "
                "Do NOT call for order placement or customer data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about the menu, e.g. 'gluten-free options' or 'price of roast chicken'."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_customer_data",
            "description": (
                "Persist customer name, address, and optionally email to the database. "
                "Call this ONLY once, after you have confirmed ALL required fields with the customer. "
                "Do NOT call speculatively or with missing fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name": {
                        "type": "string",
                        "description": "Customer's full name as provided."
                    },
                    "address": {
                        "type": "string",
                        "description": "Complete delivery address including street, number, and any unit."
                    },
                    "email": {
                        "type": "string",
                        "description": "Customer email (optional, include only if provided)."
                    }
                },
                "required": ["full_name", "address"]
            }
        }
    }
]

# ==========================================
# TOOL EXECUTOR
# Isolated so each handler is independently
# testable and errors don't collapse the loop.
# ==========================================
def execute_tool(tool_name: str, args: dict, tenant_id: str, user_phone: str) -> str:
    """
    Dispatches a tool call and returns a plain-text result string.
    Raises ValueError on bad tool name or missing args.
    """
    if tool_name == "query_menu":
        query = args.get("query", "")
        if not query:
            return "ERROR: No query provided to menu search."
        # --- Replace this stub with your real vector DB call ---
        # Example: results = vector_db.similarity_search(query, tenant_id=tenant_id)
        logger.info(f"[query_menu] tenant={tenant_id} query='{query}'")
        return (
            "MENU RESULTS:\n"
            "- Roast Chicken: $20 (contains gluten-free option available)\n"
            "- Yuca: $4\n"
            "- Garden Salad: $3\n"
            "Source: menu vector index."
        )

    elif tool_name == "save_customer_data":
        full_name = args.get("full_name")
        address = args.get("address")
        email = args.get("email")

        if not full_name or not address:
            return "ERROR: full_name and address are required. Ask the customer for missing data."

        logger.info(f"[save_customer_data] phone={user_phone} name='{full_name}'")

        try:
            with engine.begin() as conn:
                # Upsert customer
                row = conn.execute(text("""
                    INSERT INTO customers (phone_number, full_name, email)
                    VALUES (:ph, :name, :em)
                    ON CONFLICT (phone_number)
                    DO UPDATE SET full_name = EXCLUDED.full_name,
                                  email     = COALESCE(EXCLUDED.email, customers.email)
                    RETURNING id
                """), {"ph": user_phone, "name": full_name, "em": email}).fetchone()
                customer_id = row[0]

                # Ensure tenant ↔ customer link
                tc_row = conn.execute(text("""
                    INSERT INTO tenant_customers (tenant_id, customer_id)
                    VALUES (:tid, :cid)
                    ON CONFLICT DO NOTHING
                    RETURNING id
                """), {"tid": tenant_id, "cid": customer_id}).fetchone()

                if tc_row:
                    tc_id = tc_row[0]
                else:
                    tc_id = conn.execute(text("""
                        SELECT id FROM tenant_customers
                        WHERE tenant_id = :tid AND customer_id = :cid
                    """), {"tid": tenant_id, "cid": customer_id}).fetchone()[0]

                # Upsert default address (avoid duplicates on re-orders)
                conn.execute(text("""
                    INSERT INTO tenant_customer_addresses
                        (tenant_customer_id, address_line_1, is_default, city, state, country)
                    VALUES (:tcid, :addr, true, 'Toronto', 'ON', 'Canada')
                    ON CONFLICT (tenant_customer_id, is_default)
                    DO UPDATE SET address_line_1 = EXCLUDED.address_line_1
                """), {"tcid": tc_id, "addr": address})

            return "SUCCESS: Customer data saved correctly."

        except Exception as db_err:
            logger.error(f"[save_customer_data] DB error: {db_err}")
            return f"ERROR: Could not save customer data ({db_err}). Please retry."

    else:
        return f"ERROR: Unknown tool '{tool_name}'."


# ==========================================
# SYSTEM PROMPT BUILDER
# Kept short and directive — LLaMA 3 drifts
# with long, paragraph-heavy prompts.
# ==========================================
def build_system_prompt(base_prompt: str, brand_name: str, customer_context: str) -> str:
    return f"""{base_prompt.format(restaurant_name=brand_name, menu_context="[use query_menu tool]")}

--- CUSTOMER CONTEXT ---
{customer_context}

--- STRICT RULES ---
- For ANY menu question → call query_menu first, then answer.
- To save the order → call save_customer_data ONCE with full_name + address confirmed by the customer.
- Never invent menu items or prices. Always use query_menu.
- Never call save_customer_data until BOTH full_name AND address are explicitly confirmed.
- When save_customer_data returns SUCCESS → append exactly [ORDER_FINALIZED] at the end of your reply.
- Be concise. One question at a time."""


# ==========================================
# AGENTIC TOOL LOOP
# Runs up to MAX_TOOL_ITERATIONS rounds so
# the model can chain tools without hanging.
# ==========================================
def run_tool_loop(client: Groq, model_name: str, messages: list, tenant_id: str, user_phone: str) -> str:
    """
    Executes the model + tool loop until the model returns a plain
    text response (no more tool calls) or we hit the iteration cap.
    Returns the final assistant text.
    """
    for iteration in range(MAX_TOOL_ITERATIONS):
        logger.info(f"[tool_loop] iteration={iteration + 1}")

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=GROQ_TOOLS,
            tool_choice="auto",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )

        response_message = completion.choices[0].message

        # No tool calls → model is done, return text
        if not response_message.tool_calls:
            return response_message.content or ""

        # Append the assistant turn ONCE (outside inner loop)
        messages.append(response_message)

        # Process every tool call in this turn
        for tool_call in response_message.tool_calls:
            f_name = tool_call.function.name
            try:
                f_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                f_args = {}
                logger.warning(f"[tool_loop] Bad JSON args for {f_name}")

            tool_result = execute_tool(f_name, f_args, tenant_id, user_phone)
            logger.info(f"[tool_loop] {f_name} → {tool_result[:80]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": f_name,
                "content": tool_result,
            })

    # Fallback: ask model for a plain response after exhausting iterations
    logger.warning("[tool_loop] Hit max iterations, forcing final response.")
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEED,
    )
    return completion.choices[0].message.content or ""


# ==========================================
# CHAT ENDPOINT
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_phone = request.from_number

    try:
        with engine.connect() as connection:

            # ── 1. Resolve Tenant ──────────────────────────────────────────
            tenant_data = connection.execute(text("""
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
                WHERE t.phone_number = :to_num
            """), {"to_num": request.to_number}).mappings().first()

            if not tenant_data:
                raise HTTPException(status_code=404, detail="Restaurant not found.")
            if tenant_data["status"] != "Active":
                raise HTTPException(status_code=403, detail="Restaurant is currently inactive.")

            tenant_id = tenant_data["tenant_id"]
            client = Groq(api_key=tenant_data["api_key"])

            # ── 2. Resolve Customer ────────────────────────────────────────
            customer_data = connection.execute(text("""
                SELECT c.full_name, a.address_line_1
                FROM customers c
                LEFT JOIN tenant_customers tc
                    ON c.id = tc.customer_id AND tc.tenant_id = :tenant_id
                LEFT JOIN tenant_customer_addresses a
                    ON tc.id = a.tenant_customer_id AND a.is_default = true
                WHERE c.phone_number = :from_num
            """), {"tenant_id": tenant_id, "from_num": user_phone}).mappings().first()

            if customer_data and customer_data["full_name"]:
                c_name = customer_data["full_name"]
                c_addr = customer_data["address_line_1"] or "not on file"
                customer_context = (
                    f"Returning customer: {c_name}. Default address: {c_addr}. "
                    "Greet by name. Confirm address before finalizing."
                )
            else:
                customer_context = (
                    "New customer. You MUST collect full_name and address "
                    "before calling save_customer_data."
                )

            # ── 3. Build System Prompt ─────────────────────────────────────
            system_prompt = build_system_prompt(
                tenant_data["system_prompt"],
                tenant_data["brand_name"],
                customer_context,
            )

            # ── 4. Session Management ──────────────────────────────────────
            session = sessions.get(user_phone)
            if not session or session.get("status") == "completed":
                sessions[user_phone] = {
                    "session_id": str(uuid.uuid4()),
                    "status": "active",
                    "messages": [{"role": "system", "content": system_prompt}],
                }
            else:
                # Refresh system prompt on every turn (customer data may change)
                sessions[user_phone]["messages"][0] = {
                    "role": "system",
                    "content": system_prompt,
                }

            sessions[user_phone]["messages"].append({
                "role": "user",
                "content": request.message,
            })

            # ── 5. Agentic Tool Loop ───────────────────────────────────────
            final_text = run_tool_loop(
                client=client,
                model_name=tenant_data["model_name"],
                messages=sessions[user_phone]["messages"],
                tenant_id=tenant_id,
                user_phone=user_phone,
            )

            # ── 6. Closure Detection ───────────────────────────────────────
            if "[ORDER_FINALIZED]" in final_text:
                final_text = final_text.replace("[ORDER_FINALIZED]", "").strip()
                sessions[user_phone]["status"] = "completed"

            sessions[user_phone]["messages"].append({
                "role": "assistant",
                "content": final_text,
            })

            return {
                "reply": final_text,
                "session_id": sessions[user_phone]["session_id"],
                "status": sessions[user_phone]["status"],
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error for phone={user_phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)