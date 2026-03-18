import smtplib
from email.message import EmailMessage

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

    bedrooms = apartment.get("bedrooms")
    bathrooms = apartment.get("bathrooms")

    subject = f"New Lead for Apartment {apartment_id}"
    body = f"""A new lead is interested in one of your properties.

Apartment Details
-----------------
Apartment ID: {apartment.get("apartment_id")}
Title: {apartment.get("title")}
City: {apartment.get("city")}
Area: {apartment.get("area")}
Price: {apartment.get("price")} EGP
Bedrooms: {bedrooms if bedrooms is not None else "N/A"}
Bathrooms: {bathrooms if bathrooms is not None else "N/A"}
Area Size: {apartment.get("area_sqm")} sqm
View: {apartment.get("view")}

Lead Details
------------
Name: {lead_data.get("name")}
Phone: {lead_data.get("phone")}
Email: {lead_data.get("email")}
Preferred Contact Time: {lead_data.get("preferred_contact_time")}

Regards,
Dorra AI Assistant"""

    try:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = settings.smtp_from
        message["To"] = agent_email
        message.set_content(body)

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)

        return {
            "success": True,
            "message": f"Lead email sent successfully to {agent_email}.",
        }

    except Exception as error:
        return {
            "success": False,
            "message": f"Failed to send lead email: {error}",
        }