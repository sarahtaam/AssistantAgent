"""
recouvrement/predict.py — ML inference (the "30%" side of the hybrid score).

Feature extraction queries were originally Postgres-specific
(EXTRACT(DAY FROM ...)). They now use only SQL both SQLite and Postgres
accept, with day arithmetic done in Python (db.days_between).
Model artifacts are produced by `recouvrement/train.py`.
"""
import os
from datetime import date

import joblib
import pandas as pd
from sqlalchemy import text

from db import days_between, get_db_connection

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "model_lgbm.pkl")
FEATURES_PATH = os.path.join(_HERE, "model_features.pkl")

_model = None


def _load_model():
    """Loads the model once per process instead of on every request."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

FEATURES = [
    'retard_moyen_jours', 'nb_retards', 'dette_impayee',
    'anciennete_jours', 'nb_credits', 'taux_interet',
    'nb_comptes', 'ratio_utilisation',
    'score_retard', 'ratio_dette_anciennete',
    'risque_global', 'charge_credit', 'complexite_financiere',
]


def extraire_features_client(client_id: int) -> dict:
    conn = get_db_connection()
    try:
        late_rows = conn.execute(text("""
            SELECT f.date_echeance, f.date_paiement
            FROM factures f JOIN contrats c ON f.contrat_id = c.id
            WHERE c.user_id = :cid AND f.statut = 'PAYEE' AND f.date_paiement > f.date_echeance
        """), {"cid": client_id}).fetchall()
        nb_retards = len(late_rows)
        retard = (
            sum(days_between(r.date_echeance, r.date_paiement) for r in late_rows) / nb_retards
            if late_rows else 0
        )

        dette = conn.execute(text("""
            SELECT COALESCE(SUM(f.montant), 0) FROM factures f JOIN contrats c ON f.contrat_id = c.id
            WHERE c.user_id = :cid AND f.statut IN ('EN_ATTENTE', 'EN_RETARD')
        """), {"cid": client_id}).scalar() or 0

        first_contract = conn.execute(
            text("SELECT MIN(created_at) FROM contrats WHERE user_id = :cid"), {"cid": client_id}
        ).scalar()
        # Seniority in months (30-day units), matching the training features.
        anciennete = days_between(first_contract, date.today()) / 30 if first_contract else 0

        nb_credits = conn.execute(
            text("SELECT COUNT(*) FROM contrats WHERE user_id = :cid"), {"cid": client_id}
        ).scalar() or 0

        nb_impayes = conn.execute(text("""
            SELECT COUNT(*) FROM factures f JOIN contrats c ON f.contrat_id = c.id
            WHERE c.user_id = :cid AND f.statut IN ('EN_ATTENTE', 'EN_RETARD')
        """), {"cid": client_id}).scalar() or 0
        taux_interet = min(5 + nb_impayes * 2, 34)

        nb_comptes = conn.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT id FROM contrats WHERE user_id = :cid
                UNION ALL
                SELECT id FROM contrats_sim WHERE user_id = :cid
            ) AS lignes
        """), {"cid": client_id}).scalar() or 0

        ratio_util = min(20 + float(dette) / 100, 50)

        retard_val = max(float(retard), 0)
        dette_val = max(float(dette), 0)
        anciennete_v = max(float(anciennete), 0)

        return {
            'retard_moyen_jours': round(retard_val, 2),
            'nb_retards': int(nb_retards),
            'dette_impayee': round(dette_val, 2),
            'anciennete_jours': round(anciennete_v, 2),
            'nb_credits': int(nb_credits),
            'taux_interet': round(taux_interet, 2),
            'nb_comptes': int(nb_comptes),
            'ratio_utilisation': round(ratio_util, 2),
            'score_retard': round(nb_retards * retard_val, 2),
            'ratio_dette_anciennete': round(dette_val / (anciennete_v + 1), 2),
            'risque_global': round(nb_retards * taux_interet / (anciennete_v + 1), 2),
            'charge_credit': round(dette_val / (nb_credits + 1), 2),
            'complexite_financiere': round(nb_comptes * nb_credits, 2),
        }
    finally:
        conn.close()


def predire_paiement_ml(client_id: int) -> dict:
    if not os.path.exists(MODEL_PATH):
        return {"disponible": False, "message": "Model not found — run `python recouvrement/train.py` first."}

    model = _load_model()
    features = extraire_features_client(client_id)
    X = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)

    proba = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]
    proba_payer = round(float(proba[1]) * 100, 1)
    proba_pas_payer = round(float(proba[0]) * 100, 1)

    if proba_payer >= 80:
        risque_ml, couleur = "LOW", "\U0001F7E2"
    elif proba_payer >= 60:
        risque_ml, couleur = "MODERATE", "\U0001F7E1"
    elif proba_payer >= 40:
        risque_ml, couleur = "HIGH", "\U0001F7E0"
    else:
        risque_ml, couleur = "CRITICAL", "\U0001F534"

    return {
        "disponible": True,
        "client_id": client_id,
        "prediction": "WILL_PAY" if prediction == 1 else "WONT_PAY",
        "probabilite_paiement": f"{proba_payer}%",
        "probabilite_non_paiement": f"{proba_pas_payer}%",
        "risque_ml": risque_ml,
        "couleur": couleur,
        "confiance": "HIGH" if max(proba) > 0.8 else "MEDIUM" if max(proba) > 0.6 else "LOW",
        "features_utilisees": features,
    }


def score_combine(client_id: int, score_regles: int) -> dict:
    """Final score = business rules (70%) + ML (30%)."""
    ml = predire_paiement_ml(client_id)

    if not ml.get("disponible", False):
        return {"score_final": score_regles, "score_regles": score_regles, "score_ml": None, "ml_disponible": False}

    proba_val = float(ml["probabilite_paiement"].replace("%", ""))
    score_ml = round(proba_val)
    score_final = max(0, min(100, round(score_regles * 0.70 + score_ml * 0.30)))

    return {
        "client_id": client_id,
        "score_regles": score_regles,
        "score_ml": score_ml,
        "score_final": score_final,
        "prediction_ml": ml["prediction"],
        "probabilite": ml["probabilite_paiement"],
        "risque_ml": ml["risque_ml"],
        "ml_disponible": True,
        "poids": {"regles_metier": "70%", "machine_learning": "30%"},
    }
