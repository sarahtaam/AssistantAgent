"""
recouvrement/scoring.py — Rules-based client risk scoring (the "70%" side
of the hybrid score_combine() in ml_scoring.py).

This is a faithful port of the original scoring formula: start at 50,
add up to 50 points for on-time payment ratio, subtract penalties for
late payments, unpaid balance, past suspensions, and past deactivations.
New clients with no billing history yet get a neutral default score (70)
rather than being penalized for lack of history.

Kept intentionally simple and fully auditable — a pure ML score is hard
to justify to a client or a regulator in consumer credit; blending in
transparent rules is standard practice in real collections systems.
"""
from sqlalchemy import text

from db import get_db_connection


def calculer_score_client(client_id: int) -> dict:
    conn = get_db_connection()
    try:
        def _scalar(query: str) -> int:
            return conn.execute(text(query), {"cid": client_id}).scalar() or 0

        total_esim = _scalar("""
            SELECT COUNT(*) FROM factures f
            JOIN contrats c ON f.contrat_id = c.id
            WHERE c.user_id = :cid AND c.type_contrat = 'POSTPAYE'
        """)
        total_sim = _scalar("""
            SELECT COUNT(*) FROM factures f
            JOIN contrats_sim c ON f.contrat_sim_id = c.id
            WHERE c.user_id = :cid AND c.type_contrat = 'POSTPAYE'
        """)
        total = total_esim + total_sim

        payees_esim = _scalar("""
            SELECT COUNT(*) FROM factures f
            JOIN contrats c ON f.contrat_id = c.id
            WHERE c.user_id = :cid AND f.statut = 'PAYEE' AND f.date_paiement <= f.date_echeance
        """)
        payees_sim = _scalar("""
            SELECT COUNT(*) FROM factures f
            JOIN contrats_sim c ON f.contrat_sim_id = c.id
            WHERE c.user_id = :cid AND f.statut = 'PAYEE' AND f.date_paiement <= f.date_echeance
        """)
        payees_a_temps = payees_esim + payees_sim

        retard_esim = _scalar("""
            SELECT COUNT(*) FROM factures f
            JOIN contrats c ON f.contrat_id = c.id
            WHERE c.user_id = :cid AND f.statut = 'PAYEE' AND f.date_paiement > f.date_echeance
        """)
        retard_sim = _scalar("""
            SELECT COUNT(*) FROM factures f
            JOIN contrats_sim c ON f.contrat_sim_id = c.id
            WHERE c.user_id = :cid AND f.statut = 'PAYEE' AND f.date_paiement > f.date_echeance
        """)
        en_retard = retard_esim + retard_sim

        impaye_esim = _scalar("""
            SELECT COUNT(*) FROM factures f
            JOIN contrats c ON f.contrat_id = c.id
            WHERE c.user_id = :cid AND f.statut IN ('EN_ATTENTE', 'EN_RETARD')
        """)
        impaye_sim = _scalar("""
            SELECT COUNT(*) FROM factures f
            JOIN contrats_sim c ON f.contrat_sim_id = c.id
            WHERE c.user_id = :cid AND f.statut IN ('EN_ATTENTE', 'EN_RETARD')
        """)
        impayees = impaye_esim + impaye_sim

        susp_esim = _scalar("""
            SELECT COUNT(*) FROM action_recouvrement ar
            JOIN contrats c ON ar.contrat_id = c.id
            WHERE c.user_id = :cid AND ar.type_action = 'SUSPENSION'
        """)
        susp_sim = _scalar("""
            SELECT COUNT(*) FROM action_recouvrement ar
            JOIN contrats_sim c ON ar.contrat_id = c.id
            WHERE c.user_id = :cid AND ar.type_action = 'SUSPENSION'
        """)
        suspensions = susp_esim + susp_sim

        desact_esim = _scalar("""
            SELECT COUNT(*) FROM action_recouvrement ar
            JOIN contrats c ON ar.contrat_id = c.id
            WHERE c.user_id = :cid AND ar.type_action = 'DESACTIVATION'
        """)
        desact_sim = _scalar("""
            SELECT COUNT(*) FROM action_recouvrement ar
            JOIN contrats_sim c ON ar.contrat_id = c.id
            WHERE c.user_id = :cid AND ar.type_action = 'DESACTIVATION'
        """)
        desactivations = desact_esim + desact_sim

        # ── Score calculation ──
        if total == 0:
            score = 70  # new client -> neutral default, not penalized for lack of history
        else:
            ratio_a_temps = (payees_a_temps / total) * 50
            penalite_retard = (en_retard / total) * 20
            penalite_impaye = (impayees / total) * 20
            penalite_susp = min(suspensions * 5, 10)
            penalite_desact = min(desactivations * 10, 20)

            score = 50 + ratio_a_temps - penalite_retard - penalite_impaye - penalite_susp - penalite_desact
            score = max(0, min(100, round(score)))

        # ── 5-tier classification ──
        if score >= 85:
            categorie, tolerance_jours, couleur, blocage = "EXCELLENT", 15, "\U0001F7E2", False
            strategie = "Maximum tolerance — highly reliable client"
        elif score >= 65:
            categorie, tolerance_jours, couleur, blocage = "FIABLE", 10, "\U0001F535", False
            strategie = "Normal tolerance — light monitoring"
        elif score >= 45:
            categorie, tolerance_jours, couleur, blocage = "MOYEN", 5, "\U0001F7E1", False
            strategie = "Standard monitoring — regular reminders"
        elif score >= 25:
            categorie, tolerance_jours, couleur, blocage = "RISQUE", 2, "\U0001F7E0", False
            strategie = "Fast-track suspension — priority follow-up"
        else:
            categorie, tolerance_jours, couleur, blocage = "CRITIQUE", 0, "\U0001F534", True
            strategie = "Purchase blocked + immediate suspension"

        return {
            "client_id": client_id,
            "score": score,
            "categorie": categorie,
            "couleur": couleur,
            "tolerance_jours": tolerance_jours,
            "blocage_achat": blocage,
            "strategie": strategie,
            "stats": {
                "total_factures": total,
                "payees_a_temps": payees_a_temps,
                "en_retard": en_retard,
                "impayees": impayees,
                "suspensions": suspensions,
                "desactivations": desactivations,
            },
        }
    finally:
        conn.close()
