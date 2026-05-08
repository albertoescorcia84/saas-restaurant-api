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
# TOOL DEFINITIONS (Function Calling Router)
# ==========================================
# All tool descriptions MUST be in English for the LLM to understand them perfectly
groq_tools = [
    {
        "type": "function",
        "function": {
            "name": "query_vector_database",
            "description": "Use this tool ALWAYS when the user asks about the menu, ingredients, prices, hours of operation, or restaurant rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The exact user query to search in the vector database"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_customer_data",
            "description": "Use this tool ONLY at the end of the order to register or update the customer's delivery address, name, or email in the relational database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string", "description": "Customer's full name if provided"},
                    "address": {"type": "string", "description": "Confirmed delivery address"},
                    "email": {"type": "string", "description": "Customer's email address"},
                    "is_default": {"type": "boolean", "description": "Does the customer want to save this as their default address?"}
                },
                "required": ["address"]
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
            # --- STEP 1: Tenant & Model Resolver ---
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

            # --- STEP 2: Customer Resolver ---
            customer_query = text("""
                SELECT c.id as customer_id, c.full_name, a.address_line_1, a.is_default
                FROM customers c
                LEFT JOIN tenant_customers tc ON c.id = tc.customer_id AND tc.tenant_id = :tenant_id AND tc.tenant_specific_status = 'Active'
                LEFT JOIN tenant_customer_addresses a ON tc.id = a.tenant_customer_id AND a.is_default = true
                WHERE c.phone_number = :from_num
            """)
            customer_data = connection.execute(customer_query, {
                "tenant_id": tenant_data["tenant_id"], 
                "from_num": request.from_number
            }).mappings().first()

            # --- STEP 3: Context & Prompt Engineering (ALL IN ENGLISH) ---
            if customer_data:
                customer_name = customer_data["full_name"]
                address_info = f"Their default delivery address is: {customer_data['address_line_1']}." if customer_data["address_line_1"] else "They have no registered address for this location."
                customer_context = f"CUSTOMER INFO: The customer's name is {customer_name}. {address_info} Greet them by their first name (e.g., 'Hi {customer_name.split()[0]}')."
            else:
                customer_context = "CUSTOMER INFO: This is a new customer. We do not know their name or address. Greet them generically (e.g., 'Hi, how can I help you today?')."

            # Combine the prompt from the database with the injected English rules
            dynamic_system_prompt = f"""
            {tenant_data['system_prompt'].format(restaurant_name=tenant_data['brand_name'], menu_context='(Available via tools)')}
            
            {customer_context}
            
            STRICT WORKFLOW RULES:
            1. DO NOT ask for their name or address at the beginning. Focus strictly on taking their order.
            2. If they ask about the menu, use the 'query_vector_database' tool.
            3. At the end of the order process:
               - If you already have their address in CUSTOMER INFO, ask them to confirm it or provide a new one.
               - If they are a new customer, ask for their name and delivery address.
               - Once you have the confirmed delivery address, you MUST use the 'manage_customer_data' tool.
            4. When the transaction is completely finalized and ready to close, you MUST append this exact code at the end of your message: [ORDER_FINALIZED]
            """

            # --- STEP 4: Session Management ---
            user_phone = request.from_number
            if user_phone not in sessions or sessions[user_phone].get("status") == "completed":
                sessions[user_phone] = {
                    "session_id": str(uuid.uuid4()),
                    "status": "active",
                    "messages": [{"role": "system", "content": dynamic_system_prompt}]
                }

            sessions[user_phone]["messages"].append({"role": "user", "content": request.message})

            # --- STEP 5: Call LLM with Tools ---
            completion = client.chat.completions.create(
                model=tenant_data["model_name"],
                messages=sessions[user_phone]["messages"],
                tools=groq_tools,
                tool_choice="auto",
                temperature=0.3
            )

            response_message = completion.choices[0].message

            # --- STEP 6: Handle Tool Calls ---
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name == "query_vector_database":
                        # TODO: Connect to pgvector here
                        tool_result = "Vector DB Mock: Whole Roast Chicken is $20. Yuca is $4."
                        
                    elif function_name == "manage_customer_data":
                        # TODO: Execute SQL INSERT/UPDATE for customer and address
                        tool_result = "Customer data successfully saved in PostgreSQL."

                    sessions[user_phone]["messages"].append(response_message)
                    sessions[user_phone]["messages"].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result
                    })

                # Second call to get the final conversational response
                second_completion = client.chat.completions.create(
                    model=tenant_data["model_name"],
                    messages=sessions[user_phone]["messages"]
                )
                final_ai_response = second_completion.choices[0].message.content
            else:
                final_ai_response = response_message.content

            # --- STEP 7: Intercept Order Completion ---
            if "[ORDER_FINALIZED]" in final_ai_response:
                final_ai_response = final_ai_response.replace("[ORDER_FINALIZED]", "").strip()
                sessions[user_phone]["status"] = "completed"

            sessions[user_phone]["messages"].append({"role": "assistant", "content": final_ai_response})

            return {
                "reply": final_ai_response,
                "session_id": sessions[user_phone]["session_id"],
                "status": sessions[user_phone]["status"]
            }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)