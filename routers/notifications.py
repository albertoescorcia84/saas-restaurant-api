"""
routers/notifications.py
========================
Email notification endpoints using Brevo (formerly Sendinblue).

Endpoint:
  POST /notifications/send-email

The endpoint accepts a flexible payload and renders the transactional
HTML template before sending via Brevo's REST API.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, field_validator

logger = logging.getLogger("restaurant_api")

router = APIRouter(prefix="/notifications", tags=["Notifications"])

# ── Config ────────────────────────────────────────────────────────────────────
BREVO_API_KEY   = os.getenv("BREVO_API_KEY")
BREVO_SEND_URL  = "https://api.brevo.com/v3/smtp/email"
TEMPLATE_PATH   = Path(__file__).parent.parent / "templates" / "email_transactional.html"

# ── Request model ─────────────────────────────────────────────────────────────

class EmailRecipient(BaseModel):
    email: EmailStr
    name:  Optional[str] = None


class SendEmailRequest(BaseModel):
    # Sender — both required so the email looks professional
    from_email:      EmailStr
    from_name:       str

    # Recipients
    to:              list[EmailRecipient]        # at least one required
    cc:              Optional[list[EmailRecipient]] = None
    reply_to_email:  Optional[EmailStr]          = None   # defaults to from_email

    # Content
    subject:         str
    title:           str                         # H1 shown inside the email body
    body:            str                         # Main message (plain text or basic HTML)

    # Template extras (all optional)
    sender_tagline:  Optional[str]  = "Powered by TenantOS"
    sender_address:  Optional[str]  = ""
    logo_url:        Optional[str]  = None

    @field_validator("to")
    @classmethod
    def to_must_not_be_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("'to' must contain at least one recipient.")
        return v


# ── Template renderer ─────────────────────────────────────────────────────────

def render_template(req: SendEmailRequest) -> str:
    """
    Load the HTML template and replace {{placeholders}} with request values.
    Kept simple (no Jinja2 dependency) — just string replacement.
    """
    html = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Convert plain-text body newlines to <br> for HTML rendering
    body_html = req.body.replace("\n", "<br/>")

    replacements = {
        "{{subject}}":        req.subject,
        "{{sender_name}}":    req.from_name,
        "{{sender_tagline}}": req.sender_tagline or "",
        "{{title}}":          req.title,
        "{{body}}":           body_html,
        "{{reply_to}}":       req.reply_to_email or req.from_email,
        "{{year}}":           str(datetime.now().year),
        "{{sender_address}}": req.sender_address or "",
        "{{logo_url}}":       req.logo_url or "",
    }

    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    return html


# ── Brevo payload builder ─────────────────────────────────────────────────────

def build_brevo_payload(req: SendEmailRequest, html_content: str) -> dict:
    payload: dict = {
        "sender": {
            "email": req.from_email,
            "name":  req.from_name,
        },
        "to": [
            {"email": r.email, "name": r.name or r.email}
            for r in req.to
        ],
        "subject":     req.subject,
        "htmlContent": html_content,
        "replyTo": {
            "email": req.reply_to_email or req.from_email,
            "name":  req.from_name,
        },
    }

    if req.cc:
        payload["cc"] = [
            {"email": r.email, "name": r.name or r.email}
            for r in req.cc
        ]

    return payload


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/send-email",
    summary="Send a transactional email via Brevo",
    response_description="Brevo message ID on success",
)
async def send_notification_by_email(request: SendEmailRequest):
    """
    Send a branded transactional email using the HTML template.

    - **from_email** / **from_name**: sender identity (must be from a verified Brevo domain)
    - **to**: list of recipients `[{"email": "...", "name": "..."}]`
    - **cc**: optional CC recipients
    - **subject**: email subject line
    - **title**: H1 heading shown inside the email
    - **body**: main message body (plain text; newlines become `<br>`)
    - **sender_tagline**: small subtitle shown under the brand name in the header
    - **sender_address**: physical address shown in the footer
    """
    if not BREVO_API_KEY:
        logger.error("[email] BREVO_API_KEY is not set")
        raise HTTPException(500, "Email service is not configured. Set BREVO_API_KEY.")

    # Render template
    try:
        html_content = render_template(request)
    except FileNotFoundError:
        logger.error(f"[email] Template not found at {TEMPLATE_PATH}")
        raise HTTPException(500, "Email template not found.")
    except Exception as e:
        logger.error(f"[email] Template rendering error: {e}")
        raise HTTPException(500, f"Template rendering failed: {e}")

    # Build payload
    payload = build_brevo_payload(request, html_content)

    logger.info(
        f"[email] Sending '{request.subject}' "
        f"from {request.from_email} "
        f"to {[r.email for r in request.to]}"
    )

    # Send via Brevo REST API
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                BREVO_SEND_URL,
                headers={
                    "accept":       "application/json",
                    "content-type": "application/json",
                    "api-key":      BREVO_API_KEY,
                },
                content=json.dumps(payload),
            )

        if response.status_code not in (200, 201):
            error_body = response.text
            logger.error(f"[email] Brevo error {response.status_code}: {error_body}")
            raise HTTPException(
                status_code=502,
                detail=f"Brevo rejected the request ({response.status_code}): {error_body}",
            )

        result = response.json()
        message_id = result.get("messageId", "unknown")
        logger.info(f"[email] Sent successfully. messageId={message_id}")

        return {
            "success":    True,
            "message_id": message_id,
            "to":         [r.email for r in request.to],
            "cc":         [r.email for r in request.cc] if request.cc else [],
            "subject":    request.subject,
        }

    except httpx.TimeoutException:
        logger.error("[email] Brevo request timed out")
        raise HTTPException(504, "Email service timed out. Please retry.")
    except httpx.RequestError as e:
        logger.error(f"[email] Network error: {e}")
        raise HTTPException(502, f"Could not reach email service: {e}")
