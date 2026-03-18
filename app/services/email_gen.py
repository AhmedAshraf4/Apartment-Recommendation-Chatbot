from brevo import Brevo
from app.core.config import settings
from langsmith import traceable


@traceable(name="send_email")
def send_email(apartment, lead_data):
    agent_email = apartment.get("agent_email")
    apartment_id = apartment.get("apartment_id")

    if not agent_email:
        return {
            "success": False,
            "message": f"No agent email found for apartment {apartment_id}.",
        }

    if not settings.brevo_api_key or not settings.brevo_from_email:
        return {
            "success": False,
            "message": "Brevo is not configured. Missing BREVO_API_KEY or BREVO_FROM_EMAIL.",
        }

    bedrooms = apartment.get("bedrooms")
    bathrooms = apartment.get("bathrooms")

    subject = f"New Lead for Apartment {apartment_id}"
    html_content = f"""
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

    try:
        client = Brevo(api_key=settings.brevo_api_key, timeout=15.0)

        response = client.transactional_emails.with_raw_response.send_transac_email(
            sender={
                "email": settings.brevo_from_email,
                "name": settings.brevo_from_name or "Dorra AI Assistant",
            },
            to=[{"email": agent_email}],
            subject=subject,
            html_content=html_content,
            request_options={"timeout_in_seconds": 15},
        )

        if 200 <= response.status_code < 300:
            return {
                "success": True,
                "message": f"Lead email sent successfully to {agent_email}.",
            }

        return {
            "success": False,
            "message": f"Failed to send lead email: status {response.status_code}, body {response.data}",
        }

    except Exception as error:
        return {
            "success": False,
            "message": f"Failed to send lead email: {error}",
        }