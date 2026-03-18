import json
from urllib import request, error

from app.core.config import settings
from langsmith import traceable


RESEND_API_URL = "https://api.resend.com/emails"


@traceable(name="send_email")
def send_email(apartment, lead_data):
    agent_email = apartment.get("agent_email")
    apartment_id = apartment.get("apartment_id")

    if not agent_email:
        return {
            "success": False,
            "message": f"No agent email found for apartment {apartment_id}.",
        }

    resend_api_key = getattr(settings, "resend_api_key", None)
    resend_from_email = getattr(settings, "resend_from_email", None)

    if not resend_api_key or not resend_from_email:
        return {
            "success": False,
            "message": "Resend is not configured. Missing RESEND_API_KEY or RESEND_FROM_EMAIL.",
        }

    bedrooms = apartment.get("bedrooms")
    bathrooms = apartment.get("bathrooms")

    subject = f"New Lead for Apartment {apartment_id}"
    html = f"""
    <h2>New Lead for Apartment {apartment_id}</h2>

    <h3>Apartment Details</h3>
    <ul>
      <li><strong>Apartment ID:</strong> {apartment.get("apartment_id")}</li>
      <li><strong>Title:</strong> {apartment.get("title")}</li>
      <li><strong>City:</strong> {apartment.get("city")}</li>
      <li><strong>Area:</strong> {apartment.get("area")}</li>
      <li><strong>Price:</strong> {apartment.get("price")} EGP</li>
      <li><strong>Bedrooms:</strong> {bedrooms if bedrooms is not None else "N/A"}</li>
      <li><strong>Bathrooms:</strong> {bathrooms if bathrooms is not None else "N/A"}</li>
      <li><strong>Area Size:</strong> {apartment.get("area_sqm")} sqm</li>
      <li><strong>View:</strong> {apartment.get("view")}</li>
    </ul>

    <h3>Lead Details</h3>
    <ul>
      <li><strong>Name:</strong> {lead_data.get("name")}</li>
      <li><strong>Phone:</strong> {lead_data.get("phone")}</li>
      <li><strong>Email:</strong> {lead_data.get("email")}</li>
      <li><strong>Preferred Contact Time:</strong> {lead_data.get("preferred_contact_time")}</li>
    </ul>

    <p>Regards,<br>Dorra AI Assistant</p>
    """.strip()

    payload = {
        "from": resend_from_email,
        "to": [agent_email],
        "subject": subject,
        "html": html,
    }

    req = request.Request(
        RESEND_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {resend_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=15) as response:
            response_body = response.read().decode("utf-8")
            response_data = json.loads(response_body) if response_body else {}

        return {
            "success": True,
            "message": f"Lead email sent successfully to {agent_email}.",
            "provider_response": response_data,
        }

    except error.HTTPError as http_error:
        error_body = http_error.read().decode("utf-8", errors="ignore")
        return {
            "success": False,
            "message": f"Failed to send lead email: {http_error.code} {error_body}",
        }

    except Exception as err:
        return {
            "success": False,
            "message": f"Failed to send lead email: {err}",
        }