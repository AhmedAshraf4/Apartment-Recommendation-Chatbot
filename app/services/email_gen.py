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

    subject = f"New Lead for Apartment {apartment_id}"
    body = f"""
            A new lead is interested in one of your properties.
            
            Apartment Details
            -----------------
            Apartment ID: {apartment.get("apartment_id")}
            Title: {apartment.get("title")}
            City: {apartment.get("city")}
            Area: {apartment.get("area")}
            Price: {apartment.get("price")} EGP
            Bedrooms: {apartment.get("bedrooms")}
            Bathrooms: {apartment.get("bathrooms")}
            Area Size: {apartment.get("area_sqm")} sqm
            View: {apartment.get("view")}
            
            Lead Details
            ------------
            Name: {lead_data.get("name")}
            Phone: {lead_data.get("phone")}
            Email: {lead_data.get("email")}
            Preferred Contact Time: {lead_data.get("preferred_contact_time")}
            
            GoodLuck!!
            
            DORRA AI
            """.strip()

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = settings.smtp_from
        msg["To"] = agent_email
        msg.set_content(body)

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)
        return {
            "success": True,
            "message": f"Lead email sent successfully to {agent_email}.",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to send lead email: {str(e)}",
        }