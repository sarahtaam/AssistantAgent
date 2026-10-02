"""
recouvrement/actions.py — Side-effect actions (notifications).

envoyer_email_simulation() logs every notification so the agent's behavior
is visible/demoable without any mail server. Real SMTP sending is optional
and only activates if SMTP_* env vars are set — this mirrors the original's
"simulation-first, real-send-if-configured" design.

Recipients are masked in logs (see privacy.py). Called from a background
thread, never on the request path, so a slow SMTP server can't delay a
client's response; the timeout still bounds how long a send can hang.
"""
import os
import logging

from privacy import mask_contact

logger = logging.getLogger(__name__)

SMTP_TIMEOUT_SECONDS = 10


def envoyer_email_simulation(destinataire: str, sujet: str, corps: str) -> None:
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")

    if not smtp_user or not smtp_pass:
        logger.info("Email simulated (SMTP not configured) | to=%s | subject=%s",
                    mask_contact(destinataire), sujet)
        return

    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_from = os.getenv("SMTP_FROM", smtp_user)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = sujet
        msg["From"] = smtp_from
        msg["To"] = destinataire
        msg.attach(MIMEText(corps, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_from, destinataire, msg.as_string())
        logger.info("Email sent | to=%s", mask_contact(destinataire))
    except Exception as exc:
        logger.error("Email send failed | to=%s | %s", mask_contact(destinataire), type(exc).__name__)
