from brevo import Brevo
from app.core.config import settings
from app.services.lead_prepare import format_contact_time_for_display
from langsmith import traceable


def _build_agent_email_html(apartment, lead_data, formatted_contact_time: str) -> str:
    bedrooms = apartment.get("bedrooms")
    bathrooms = apartment.get("bathrooms")

    return f"""
    <h2>New Lead for Apartment {apartment.get("apartment_id")}</h2>

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
      <li><strong>Amenities:</strong> {apartment.get("amenities")}</li>
      <li><strong>Description:</strong> {apartment.get("description")}</li>
    </ul>

    <h3>Lead Details</h3>
    <ul>
      <li><strong>Name:</strong> {lead_data.get("name")}</li>
      <li><strong>Phone:</strong> {lead_data.get("phone")}</li>
      <li><strong>Email:</strong> {lead_data.get("email")}</li>
      <li><strong>Preferred Contact Time:</strong> {formatted_contact_time}</li>
    </ul>

    <p>Regards,<br>Dorra AI Assistant</p>
    """.strip()


def _build_user_email_html(apartment, lead_data, formatted_contact_time: str) -> str:
    bedrooms = apartment.get("bedrooms")
    bathrooms = apartment.get("bathrooms")

    return f"""
    <h2>Your request has been received</h2>

    <p>Hello {lead_data.get("name")},</p>

    <p>Thank you. Your request has been sent successfully.</p>

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
      <li><strong>Amenities:</strong> {apartment.get("amenities")}</li>
      <li><strong>Description:</strong> {apartment.get("description")}</li>
    </ul>

    <h3>Your Submitted Details</h3>
    <ul>
      <li><strong>Name:</strong> {lead_data.get("name")}</li>
      <li><strong>Phone:</strong> {lead_data.get("phone")}</li>
      <li><strong>Email:</strong> {lead_data.get("email")}</li>
      <li><strong>Preferred Contact Time:</strong> {formatted_contact_time}</li>
    </ul>

    <p>Our team will contact you around <strong>{formatted_contact_time}</strong>.</p>

    <p>Regards,<br>Dorra AI Assistant</p>
    """.strip()


def _send_brevo_email(client: Brevo, to_email: str, subject: str, html_content: str) -> tuple[bool, str]:
    response = client.transactional_emails.with_raw_response.send_transac_email(
        sender={
            "email": settings.brevo_from_email,
            "name": settings.brevo_from_name or "Dorra AI Assistant",
        },
        to=[{"email": to_email}],
        subject=subject,
        html_content=html_content,
        request_options={"timeout_in_seconds": 15},
    )

    if 200 <= response.status_code < 300:
        return True, "sent"

    return False, f"status {response.status_code}, body {response.data}"


@traceable(name="email.send_mail")
def send_email(apartment, lead_data):
    agent_email = str(apartment.get("agent_email") or "").strip()
    user_email = str(lead_data.get("email") or "").strip()
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

    formatted_contact_time = format_contact_time_for_display(lead_data)

    agent_subject = f"New Lead for Apartment {apartment_id}"
    agent_html = _build_agent_email_html(apartment, lead_data, formatted_contact_time)

    user_subject = f"Your request for apartment {apartment_id} has been received"
    user_html = _build_user_email_html(apartment, lead_data, formatted_contact_time)

    try:
        client = Brevo(api_key=settings.brevo_api_key, timeout=15.0)

        agent_ok, agent_result = _send_brevo_email(client, agent_email, agent_subject, agent_html)
        if not agent_ok:
            return {
                "success": False,
                "message": f"Failed to send lead email to agent: {agent_result}",
            }

        user_email_sent = False
        user_email_message = ""

        if user_email:
            user_ok, user_result = _send_brevo_email(client, user_email, user_subject, user_html)
            user_email_sent = user_ok
            user_email_message = user_result

        if user_email and not user_email_sent:
            return {
                "success": True,
                "message": f"Lead email sent successfully to {agent_email}, but failed to send confirmation email to user: {user_email_message}",
            }

        if user_email and user_email_sent:
            return {
                "success": True,
                "message": f"Lead email sent successfully to {agent_email} and confirmation email sent to {user_email}.",
            }

        return {
            "success": True,
            "message": f"Lead email sent successfully to {agent_email}.",
        }

    except Exception as error:
        return {
            "success": False,
            "message": f"Failed to send lead email: {error}",
        }