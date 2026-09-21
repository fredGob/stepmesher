"""Estimation *a priori* de la maillabilité SC8R d'une pièce.

But : donner tôt (après l'analyse, avant les essais coûteux) un avis sur la
faisabilité d'un maillage SC8R « 1 élément dans l'épaisseur », à partir des
caractéristiques déjà calculées (peaux, chants, dispersion d'épaisseur). Deux
usages :

- **triage** (toujours) : un score et un verdict journalisés et repris dans
  `summary.csv` / le rapport JSON — utile pour classer une campagne de centaines
  de pièces et, à terme, alimenter un modèle appris (jalon 2).
- **déviation** (optionnelle) : si `[mesh] skip_sc8r_below > 0` et que le score est
  sous ce seuil, le pipeline part directement en tétra sans épuiser les essais SC8R.
  **Désactivé par défaut** (seuil 0) : les seuils sont indicatifs et doivent être
  calibrés sur une vraie campagne (dont les cas tout-tétra, ex. jonctions en T)
  avant d'automatiser la déviation.

Le score n'a rien d'un oracle : c'est une combinaison lisible de signaux, pas un
critère de qualité. Il ne remplace jamais le verdict réel du mailleur.
"""
from __future__ import annotations


def assess_feasibility(pa) -> dict:
    """Avis de faisabilité SC8R pour une pièce analysée (`PartAnalysis`).

    Retourne un dict : `score` (0..1, plus haut = plus favorable au SC8R),
    `verdict` ("sc8r" / "hard" / "tet"), `reason` (texte court) et `signals`
    (les valeurs brutes, pour calibration).
    """
    cls = pa.classification or {}
    saf = float(cls.get("skin_area_fraction", 0.0) or 0.0)
    n_ref, n_opp = len(pa.ref_faces), len(pa.opp_faces)
    n_flank, n_failed = len(pa.flank_faces), len(pa.failed_faces)
    n_faces = n_ref + n_opp + n_flank
    # équilibre des deux peaux : le SC8R a besoin de deux peaux opposées franches
    side_balance = min(n_ref, n_opp) / max(n_ref, n_opp, 1)
    failed_frac = n_failed / max(n_faces, 1)

    signals = dict(kind=pa.kind, skin_area_fraction=round(saf, 4),
                   n_ref_faces=n_ref, n_opp_faces=n_opp, n_flank_faces=n_flank,
                   n_failed_faces=n_failed, side_balance=round(side_balance, 4),
                   failed_frac=round(failed_frac, 4))

    # cas tranchés : pas de peaux -> pas de SC8R possible
    if pa.kind == "massive" or n_ref == 0 or n_opp == 0:
        return dict(score=0.0, verdict="tet",
                    reason="pièce massive ou sans deux peaux opposées", signals=signals)

    # score lisible : couverture de peau, pondérée par l'équilibre des deux peaux,
    # pénalisée par les faces dont l'analyse a échoué.
    score = saf * (0.5 + 0.5 * side_balance) * (1.0 - min(failed_frac, 0.5))
    score = float(max(0.0, min(1.0, score)))
    if score >= 0.55:
        verdict, reason = "sc8r", "deux peaux franches, forte couverture"
    elif score >= 0.40:
        verdict, reason = "hard", "couverture ou équilibre des peaux moyens"
    else:
        verdict, reason = "tet", "peaux peu franches ou couverture faible"
    return dict(score=round(score, 4), verdict=verdict, reason=reason, signals=signals)
