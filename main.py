import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables from .env file
load_dotenv()

# Database and Groq configuration
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize database engine and Groq client
engine = create_engine(DATABASE_URL)
client = Groq(api_key=GROQ_API_KEY)

# Initialize FastAPI application
app = FastAPI(title="SaaS Restaurant POC")

# ==========================================
# 1. IN-MEMORY STATE MANAGER (Sessions)
# ==========================================
# Dictionary to store session state, ID, and chat history per phone number
sessions = {}

# ==========================================
# DATA MODELS
# ==========================================
class ChatRequest(BaseModel):
    to_number: str      # The restaurant's phone number
    from_number: str    # The customer's phone number
    message: str        # The message sent by the customer

# ==========================================
# CHAT ENDPOINT
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # --- STEP 1: Tenant Resolver & Menu Retrieval (Railway DB) ---
        with engine.connect() as connection:
            # Query the tenant and their menu context using a JOIN
            query = text("""
                SELECT t.name, t.is_active, m.content 
                FROM tenants t
                JOIN menu_vectors m ON t.tenant_id = m.tenant_id
                WHERE t.phone_number = :to_num
            """)
            
            result = connection.execute(query, {"to_num": request.to_number}).fetchone()

            # Handle case where restaurant is not found
            if result is None:
                raise HTTPException(status_code=404, detail=f"Restaurant with number {request.to_number} not found.")

            restaurant_name, is_active, menu_context = result

            # Handle case where subscription is expired
            if not is_active:
                raise HTTPException(status_code=403, detail="Subscription is inactive.")

# --- STEP 2: Configure System Prompt with Conversational Rules ---
        system_prompt = f"""
        You are the friendly and professional virtual assistant for {restaurant_name}.
        Menu Context: {menu_context}
        
        Strict Rules: 
        1. CONVERSATION: Mirror the user's intent. If they just say hello, greet them warmly and ask how you can help today. DO NOT list the menu immediately unless they ask to order or see options.
        2. LENGTH: Keep responses natural but concise (max 2-3 sentences).
        3. ORDER FLOW: If they order a main dish, gently suggest a side. If they decline the side, summarize and confirm the final order.
        4. CRITICAL: When the user completely confirms the final order and the transaction is ready to close, you MUST append this exact code at the end of your message: [ORDER_FINALIZED]
        """

        # --- STEP 3: Handle Chat Memory and Session ID ---
        user_phone = request.from_number
        
        # Check if it's a new user OR if their previous session was already completed
        if user_phone not in sessions or sessions[user_phone].get("status") == "completed":
            sessions[user_phone] = {
                "session_id": str(uuid.uuid4()), # Generate a unique identifier for this order
                "status": "active",
                "messages": [{"role": "system", "content": system_prompt}]
            }
            print(f"New session started: {sessions[user_phone]['session_id']}")

        # Append the new user message to the active session
        sessions[user_phone]["messages"].append({"role": "user", "content": request.message})

        # --- STEP 4: Call Groq AI with Full History ---
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=sessions[user_phone]["messages"],
            temperature=0.5
        )

        ai_response = completion.choices[0].message.content

        # --- STEP 5: Intercept Order Completion Trigger ---
        if "[ORDER_FINALIZED]" in ai_response:
            # Clean the response so the customer doesn't see the secret tag
            ai_response = ai_response.replace("[ORDER_FINALIZED]", "").strip()
            
            # Mark the session as completed to reset it on the next message
            sessions[user_phone]["status"] = "completed"
            print(f"Session closed: {sessions[user_phone]['session_id']}")

        # Save the clean AI response to the history
        sessions[user_phone]["messages"].append({"role": "assistant", "content": ai_response})

        # Return the response and session details to the client
        return {
            "reply": ai_response,
            "session_id": sessions[user_phone]["session_id"],
            "status": sessions[user_phone]["status"],
            "source": "Railway DB + Stateful Memory"
        }

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Run the server locally using uvicorn
if __name__ == "__main__":
    import uvicorn
    import os
    # Railway inyecta la variable PORT automáticamente
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)