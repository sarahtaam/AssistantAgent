"""
recouvrement/actions.py — Side-effect actions (notifications).

envoyer_email_simulation() always logs to console so the agent's
notification behavior is visible/demoable without any mail server.
Real SMTP sending is optional and only activates if SMTP_* env vars
are set — this mirrors the original's "simulation-first, real-send-
if-configured" design, which is a sensible default for a demo/staging
environment.
"""
import os
import logging

logger = logging.getLogger(__name__)


def envoyer_email_simulation(destinataire: str, sujet: str, corps: str) -> None:
    print("\n" + "=" * 50)
    print("EMAIL")
    print(f"To     : {destinataire}")
    print(f"Subject: {sujet}")
    print("=" * 50 + "\n")

    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")

    if not smtp_user or not smtp_pass:
        logger.info("SMTP not configured - email simulated only, not sent.")
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

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_from, destinataire, msg.as_string())
        logger.info("Email sent to %s", destinataire)
    except Exception as exc:
        logger.error("Email send failed: %s", exc)
