"""
models.py — Table definitions (SQLAlchemy Core).

The schema itself is created and evolved by Alembic migrations in
migrations/versions/ — these definitions are the reference those migrations
are checked against (`alembic check`) and what seed.py inserts through, so
inserts stay portable across SQLite and Postgres.

Mirrors the conceptual structure of the original production schema (two
line-types: eSIM and physical SIM, each with their own status profile table)
but simplified and stripped of anything company-specific.
"""
import sqlalchemy as sa

metadata = sa.MetaData()

_now = sa.text("CURRENT_TIMESTAMP")

users = sa.Table(
    "users", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("nom", sa.String(255), nullable=False),
    sa.Column("prenom", sa.String(255), nullable=False),
    sa.Column("email_or_phone", sa.String(255), nullable=False),
    sa.Column("telephone", sa.String(50)),
)

# eSIM lines
contrats = sa.Table(
    "contrats", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False, index=True),
    sa.Column("statut", sa.String(50), nullable=False, server_default="ACTIVE"),
    # POSTPAYE | PREPAYE — scoring only applies to POSTPAYE
    sa.Column("type_contrat", sa.String(50), nullable=False, server_default="POSTPAYE"),
    sa.Column("numero_contrat", sa.String(100), nullable=False),
    sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
)

profils_esim = sa.Table(
    "profils_esim", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("contrat_id", sa.Integer, sa.ForeignKey("contrats.id"), nullable=False, index=True),
    # ACTIF | SUSPENDU | DESACTIVE | EN_ATTENTE
    sa.Column("statut", sa.String(50), nullable=False, server_default="ACTIF"),
)

# physical SIM lines
contrats_sim = sa.Table(
    "contrats_sim", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False, index=True),
    sa.Column("statut", sa.String(50), nullable=False, server_default="ACTIVE"),
    sa.Column("type_contrat", sa.String(50), nullable=False, server_default="POSTPAYE"),
    sa.Column("numero_contrat", sa.String(100), nullable=False),
    sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
)

souscriptions_sim = sa.Table(
    "souscriptions_sim", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("contrat_sim_id", sa.Integer, sa.ForeignKey("contrats_sim.id"), nullable=False, index=True),
    sa.Column("statut", sa.String(50), nullable=False, server_default="ACTIF"),
)

factures = sa.Table(
    "factures", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("contrat_id", sa.Integer, sa.ForeignKey("contrats.id"), index=True),
    sa.Column("contrat_sim_id", sa.Integer, sa.ForeignKey("contrats_sim.id"), index=True),
    sa.Column("montant", sa.Numeric(12, 2), nullable=False),
    sa.Column("date_echeance", sa.Date, nullable=False),
    sa.Column("date_paiement", sa.Date),
    # EN_ATTENTE | EN_RETARD | PAYEE
    sa.Column("statut", sa.String(50), nullable=False, server_default="EN_ATTENTE"),
)

# Collections actions taken against a client (suspension/deactivation),
# consumed by the rules-based scoring engine as a penalty signal.
action_recouvrement = sa.Table(
    "action_recouvrement", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    # FK into contrats OR contrats_sim (not enforced, matches original's dual-table pattern)
    sa.Column("contrat_id", sa.Integer, nullable=False, index=True),
    sa.Column("type_action", sa.String(50), nullable=False),  # SUSPENSION | DESACTIVATION
    sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
)

# Long-term memory: interaction summaries injected into the system
# prompt so the agent "remembers" previous conversations with this client.
client_interactions = sa.Table(
    "client_interactions", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("client_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False, index=True),
    sa.Column("session_id", sa.String(100)),
    sa.Column("summary", sa.Text, nullable=False),
    sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
)

admin_notifications = sa.Table(
    "admin_notifications", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("type", sa.String(50), nullable=False, server_default="PAYMENT_PLAN"),
    sa.Column("client_id", sa.Integer),
    sa.Column("reference_id", sa.Integer),
    sa.Column("message", sa.Text, nullable=False),
    sa.Column("read", sa.Integer, nullable=False, server_default="0"),
    sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
)

# Simplified support escalation (the real system's boutique/appointment
# booking subsystem is out of scope for this public repo — see README).
tickets = sa.Table(
    "tickets", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("client_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False, index=True),
    sa.Column("numero_ticket", sa.String(100)),
    sa.Column("complaint_type", sa.String(50), nullable=False),
    sa.Column("status", sa.String(50), nullable=False, server_default="OPEN"),
    sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
)

payment_plans = sa.Table(
    "payment_plans", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("client_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False, index=True),
    sa.Column("session_id", sa.String(100), nullable=False),
    sa.Column("plan_data", sa.Text, nullable=False),  # JSON
    sa.Column("summary", sa.Text),
    sa.Column("status", sa.String(50), nullable=False, server_default="CONFIRMED"),
    sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
    sa.Column("updated_at", sa.DateTime),
)
