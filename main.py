import os
import uuid
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

app = FastAPI(title="SaaS Restaurant Multi-Tenant API")

# ==========================================
# 1. IN-MEMORY STATE MANAGER (Sessions)
# ==========================================
sessions = {}

# ==========================================
# DATA MODELS
# ==========================================
class ChatRequest(BaseModel):
    to_number: str      # The restaurant's phone number
    from_number: str    # The customer's phone number
    message: str        # The message sent by the customer

# ==========================================
# TOOL DEFINITIONS (Function Calling)
# ==========================================
groq_tools = [
    {
        "type": "function",
        "function": {
            "name": "query_vector_database",
            "description": "ALWAYS use this when the user asks about the menu, ingredients, prices, or restaurant info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query for the menu"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_customer_data",
            "description": "MANDATORY: Call this at the end of an order to save the customer's Full Name, Email, and Address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string", "description": "Full name of the customer"},
                    "address": {"type": "string", "description": "The complete delivery address"},
                    "email": {"type": "string", "description": "Customer's email address"},
                    "is_default": {"type": "boolean", "description": "Set as default address?"}
                },
                "required": ["full_name", "address"]
            }
        }
    }
]

# ==========================================
# CHAT ENDPOINT
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        with engine.connect() as connection:
            # --- STEP 1: Resolver Tenant & Model (Llamada a DB) ---
            tenant_query = text("""
                SELECT 
                    t.id as tenant_id, t.brand_name, t.status, 
                    s.system_prompt, 
                    m.model_name, m.api_key
                FROM tenants t
                JOIN tenant_ai_settings s ON t.id = s.tenant_id
                JOIN llm_models m ON s.model_id = m.id
                WHERE t.phone_number = :to_num
            """)
            tenant_data = connection.execute(tenant_query, {"to_num": request.to_number}).mappings().first()

            if not tenant_data:
                raise HTTPException(status_code=404, detail="Restaurant not found.")
            if tenant_data["status"] != "Active":
                raise HTTPException(status_code=403, detail="Restaurant is currently inactive.")

            client = Groq(api_key=tenant_data["api_key"])

            # --- STEP 2: Resolver Customer ---
            customer_query = text("""
                SELECT c.id as customer_id, c.full_name, a.address_line_1
                FROM customers c
                LEFT JOIN tenant_customers tc ON c.id = tc.customer_id AND tc.tenant_id = :tenant_id
                LEFT JOIN tenant_customer_addresses a ON tc.id = a.tenant_customer_id AND a.is_default = true
                WHERE c.phone_number = :from_num
            """)
            customer_data = connection.execute(customer_query, {
                "tenant_id": tenant_data["tenant_id"], 
                "from_num": request.from_number
            }).mappings().first()

            # --- STEP 3: Context Injection ---
            if customer_data:
                c_name = customer_data["full_name"]
                c_addr = customer_data["address_line_1"]
                customer_context = f"CUSTOMER FOUND: {c_name}. Default Address: {c_addr if c_addr else 'None'}. Greet them by name."
            else:
                customer_context = "NEW CUSTOMER: No name or address known. You MUST collect Full Name and Address before finishing."

            # Aquí es donde se lee y formatea el prompt de la base de datos
            base_prompt = tenant_data['system_prompt'].format(
                restaurant_name=tenant_data['brand_name'], 
                menu_context="(Use query_vector_database tool)"
            )

            dynamic_system_prompt = f"""
            {base_prompt}
            
            {customer_context}
            
            STRICT PROTOCOL:
            1. Take the order first.
            2. Before confirming the order, ensure you have the Name and Address.
            3. If data is missing, ask the customer politely.
            4. Once you have the info, call 'manage_customer_data'.
            5. ONLY after the tool succeeds, append [ORDER_FINALIZED].
            """

            # --- STEP 4: Session Logic ---
            user_phone = request.from_number
            if user_phone not in sessions or sessions[user_phone].get("status") == "completed":
                sessions[user_phone] = {
                    "session_id": str(uuid.uuid4()),
                    "status": "active",
                    "messages": [{"role": "system", "content": dynamic_system_prompt}]
                }

            sessions[user_phone]["messages"].append({"role": "user", "content": request.message})

            # --- STEP 5: AI Core & Tool Handling ---
            completion = client.chat.completions.create(
                model=tenant_data["model_name"],
                messages=sessions[user_phone]["messages"],
                tools=groq_tools,
                tool_choice="auto"
            )

            response_message = completion.choices[0].message

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    f_name = tool_call.function.name
                    f_args = json.loads(tool_call.function.arguments)

                    if f_name == "query_vector_database":
                        tool_result = "Menu Info: Roast Chicken is $20. Yuca is $4. Salad is $3."
                    
                    elif f_name == "manage_customer_data":
                        # LÓGICA DE PERSISTENCIA REAL
                        with engine.begin() as conn:
                            # Insert/Update Customer
                            c_id = conn.execute(text("""
                                INSERT INTO customers (phone_number, full_name, email)
                                VALUES (:ph, :name, :em) ON CONFLICT (phone_number) 
                                DO UPDATE SET full_name = EXCLUDED.full_name RETURNING id
                            """), {"ph": user_phone, "name": f_args.get("full_name"), "em": f_args.get("email")}).fetchone()[0]

                            # Ensure Tenant Relationship
                            tc_res = conn.execute(text("""
                                INSERT INTO tenant_customers (tenant_id, customer_id)
                                VALUES (:tid, :cid) ON CONFLICT DO NOTHING RETURNING id
                            """), {"tid": tenant_data["tenant_id"], "cid": c_id}).fetchone()
                            
                            tc_id = tc_res[0] if tc_res else conn.execute(text("SELECT id FROM tenant_customers WHERE tenant_id=:tid AND customer_id=:cid"), {"tid": tenant_data["tenant_id"], "cid": c_id}).fetchone()[0]

                            # Insert Address
                            conn.execute(text("""
                                INSERT INTO tenant_customer_addresses (tenant_customer_id, address_line_1, is_default, city, state, country)
                                VALUES (:tcid, :addr, true, 'Toronto', 'ON', 'Canada')
                            """), {"tcid": tc_id, "addr": f_args.get("address")})

                        tool_result = "SUCCESS: Customer data saved."

                    sessions[user_phone]["messages"].append(response_message)
                    sessions[user_phone]["messages"].append({"role": "tool", "tool_call_id": tool_call.id, "name": f_name, "content": tool_result})

                # Final response after tool execution
                second_call = client.chat.completions.create(model=tenant_data["model_name"], messages=sessions[user_phone]["messages"])
                final_ai_response = second_call.choices[0].message.content
            else:
                final_ai_response = response_message.content

            # --- STEP 6: Closure ---
            if "[ORDER_FINALIZED]" in final_ai_response:
                final_ai_response = final_ai_response.replace("[ORDER_FINALIZED]", "").strip()
                sessions[user_phone]["status"] = "completed"

            sessions[user_phone]["messages"].append({"role": "assistant", "content": final_ai_response})

            return {"reply": final_ai_response, "session_id": sessions[user_phone]["session_id"], "status": sessions[user_phone]["status"]}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)