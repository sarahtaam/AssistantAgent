"""
seed.py — Generates a synthetic demo database.

Run this once before starting the server:
    python seed.py

Applies all migrations (same as `alembic upgrade head`), then creates ~25
fictional clients spanning the full range of risk profiles (good payers,
late payers, suspended lines) so the agent's scoring, negotiation, and
guardrail behavior can all be exercised end-to-end without any real client
data. Does nothing if the database already has clients.

Demo data only — never run this against a production database.
"""
import random
from datetime import date, timedelta

import sqlalchemy as sa

from db import get_engine, migrate
from models import action_recouvrement, contrats, factures, profils_esim, users

FIRST_NAMES = ["Amina", "Youssef", "Lina", "Karim", "Salma", "Nadia", "Omar",
               "Hela", "Walid", "Rania", "Sami", "Farah", "Bilel", "Nour",
               "Adam", "Ines", "Marwan", "Sirine", "Ahmed", "Yasmine"]
LAST_NAMES = ["Trabelsi", "Bouazizi", "Cherif", "Gharbi", "Mansour", "Jlassi",
              "Khemiri", "Saidi", "Baccar", "Zribi", "Ferjani", "Amri"]


def _profile(i: int) -> str:
    """Distributes clients across risk profiles for a realistic demo mix."""
    if i % 5 == 0:
        return "critical"      # heavily suspended, multiple late payments
    if i % 5 in (1, 2):
        return "risk"          # some unpaid invoices, a few late payments
    if i % 5 == 3:
        return "average"       # one unpaid invoice, mostly on time
    return "excellent"         # fully paid up, always on time


def seed() -> None:
    random.seed(42)
    migrate()
    engine = get_engine()

    with engine.begin() as conn:
        if conn.execute(sa.select(sa.func.count()).select_from(users)).scalar():
            print("Database already has clients — skipping seed.")
            return

        today = date.today()
        for i in range(1, 26):
            prenom = random.choice(FIRST_NAMES)
            nom = random.choice(LAST_NAMES)
            profile = _profile(i)

            user_id = conn.execute(users.insert().values(
                nom=nom, prenom=prenom,
                email_or_phone=f"{prenom.lower()}.{nom.lower()}@example.com",
                telephone=f"+216 2{random.randint(1000000, 9999999)}",
            )).inserted_primary_key[0]

            esim_status = "SUSPENDU" if profile in ("critical", "risk") and random.random() < 0.5 else "ACTIF"
            contrat_id = conn.execute(contrats.insert().values(
                user_id=user_id, statut="ACTIVE", type_contrat="POSTPAYE", numero_contrat=f"ESIM-{1000 + i}",
            )).inserted_primary_key[0]
            conn.execute(profils_esim.insert().values(contrat_id=contrat_id, statut=esim_status))

            # Collections actions (drives the suspension/deactivation penalty
            # in the rules-based score) — scaled by risk profile
            n_suspensions = {"critical": 2, "risk": 1, "average": 0, "excellent": 0}[profile]
            for _ in range(n_suspensions):
                conn.execute(action_recouvrement.insert().values(contrat_id=contrat_id, type_action="SUSPENSION"))
            if profile == "critical" and random.random() < 0.3:
                conn.execute(action_recouvrement.insert().values(contrat_id=contrat_id, type_action="DESACTIVATION"))

            # Unpaid invoices, scaled by profile
            n_unpaid = {"critical": 4, "risk": 2, "average": 1, "excellent": 0}[profile]
            for _ in range(n_unpaid):
                conn.execute(factures.insert().values(
                    contrat_id=contrat_id, montant=round(random.uniform(20, 150), 2),
                    date_echeance=today - timedelta(days=random.randint(5, 90)), statut="EN_RETARD",
                ))

            # Payment history (paid invoices, some on time some late)
            n_paid = random.randint(3, 8)
            late_rate = {"critical": 0.8, "risk": 0.5, "average": 0.2, "excellent": 0.02}[profile]
            for j in range(n_paid):
                due = today - timedelta(days=30 * (j + 1))
                late_days = random.randint(1, 20) if random.random() < late_rate else 0
                conn.execute(factures.insert().values(
                    contrat_id=contrat_id, montant=round(random.uniform(20, 150), 2),
                    date_echeance=due, date_paiement=due + timedelta(days=late_days), statut="PAYEE",
                ))

    print(f"Seed complete: 25 synthetic clients created ({engine.url.render_as_string(hide_password=True)})")


if __name__ == "__main__":
    seed()
