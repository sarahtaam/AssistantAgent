-- schema.sql — Demo schema for the NovaTel collections agent.
-- Mirrors the conceptual structure of the original production schema
-- (two line-types: eSIM and physical SIM, each with their own status
-- profile table) but simplified and stripped of anything company-specific.

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nom             TEXT NOT NULL,
    prenom          TEXT NOT NULL,
    email_or_phone  TEXT NOT NULL,
    telephone       TEXT
);

CREATE TABLE IF NOT EXISTS contrats (              -- eSIM lines
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    statut          TEXT NOT NULL DEFAULT 'ACTIVE',
    type_contrat    TEXT NOT NULL DEFAULT 'POSTPAYE',   -- POSTPAYE | PREPAYE — scoring only applies to POSTPAYE
    numero_contrat  TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS profils_esim (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    contrat_id   INTEGER NOT NULL REFERENCES contrats(id),
    statut       TEXT NOT NULL DEFAULT 'ACTIF'     -- ACTIF | SUSPENDU | DESACTIVE | EN_ATTENTE
);

CREATE TABLE IF NOT EXISTS contrats_sim (           -- physical SIM lines
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    statut          TEXT NOT NULL DEFAULT 'ACTIVE',
    type_contrat    TEXT NOT NULL DEFAULT 'POSTPAYE',
    numero_contrat  TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS souscriptions_sim (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    contrat_sim_id  INTEGER NOT NULL REFERENCES contrats_sim(id),
    statut          TEXT NOT NULL DEFAULT 'ACTIF'
);

CREATE TABLE IF NOT EXISTS factures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    contrat_id      INTEGER REFERENCES contrats(id),
    contrat_sim_id  INTEGER REFERENCES contrats_sim(id),
    montant         REAL NOT NULL,
    date_echeance   TEXT NOT NULL,
    date_paiement   TEXT,
    statut          TEXT NOT NULL DEFAULT 'EN_ATTENTE'  -- EN_ATTENTE | EN_RETARD | PAYEE
);

-- Collections actions taken against a client (suspension/deactivation),
-- consumed by the rules-based scoring engine as a penalty signal.
CREATE TABLE IF NOT EXISTS action_recouvrement (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    contrat_id      INTEGER NOT NULL,   -- FK into contrats OR contrats_sim (not enforced, matches original's dual-table pattern)
    type_action     TEXT NOT NULL,      -- SUSPENSION | DESACTIVATION
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Long-term memory: interaction summaries injected into the system
-- prompt so the agent "remembers" previous conversations with this client.
CREATE TABLE IF NOT EXISTS client_interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id   INTEGER NOT NULL REFERENCES users(id),
    session_id  TEXT,
    summary     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS admin_notifications (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    type         TEXT NOT NULL DEFAULT 'PAYMENT_PLAN',
    client_id    INTEGER,
    reference_id INTEGER,
    message      TEXT NOT NULL,
    read         INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Simplified support escalation (the real system's boutique/appointment
-- booking subsystem is out of scope for this public repo — see README).
CREATE TABLE IF NOT EXISTS tickets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id       INTEGER NOT NULL REFERENCES users(id),
    numero_ticket   TEXT,
    complaint_type  TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'OPEN',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS payment_plans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id   INTEGER NOT NULL REFERENCES users(id),
    session_id  TEXT NOT NULL,
    plan_data   TEXT NOT NULL,   -- JSON
    summary     TEXT,
    status      TEXT NOT NULL DEFAULT 'CONFIRMED',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT
);
