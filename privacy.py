"""
privacy.py — Keeps personal data out of logs.

Logs are shipped to log aggregators and error trackers that many people can
read, so they must never contain what clients typed, their contact details
or their names. Log ids (client id, session id, plan id) instead, and run
anything else that identifies a person through mask_contact().
"""


def mask_contact(value: str) -> str:
    """amina.trabelsi@example.com -> a***@example.com ; +216 21234567 -> ***4567"""
    if not value:
        return ""
    value = str(value).strip()
    if "@" in value:
        local, _, domain = value.partition("@")
        return f"{local[:1]}***@{domain}"
    digits = "".join(ch for ch in value if ch.isdigit())
    return f"***{digits[-4:]}" if len(digits) > 4 else "***"
