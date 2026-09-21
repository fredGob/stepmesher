"""Construction de l'état envoyé au LLM : résumé de la pièce + historique des essais.

Taille bornée par construction (nombre d'essais et de faces tronqués) pour qu'un contexte
de 8192 tokens suffise même sur une pièce qui enchaîne les échecs.
"""
from __future__ import annotations

import json

CHARS_PER_TOKEN = 3.3        # JSON compact + français, ordre de grandeur


def estimate_tokens(state: dict) -> int:
    return round(len(json.dumps(state, ensure_ascii=False, separators=(",", ":"))) / CHARS_PER_TOKEN)


def _recipe_brief(d: dict) -> dict:
    """Recette réduite à ce qui distingue un essai d'un autre."""
    out = dict(strategy=d.get("strategy"), size_mult=d.get("size_mult"),
               structured=bool(d.get("structured", True)))
    for k in ("free_faces", "face_alg", "merge"):
        n = len(d.get(k) or ())
        if n:
            out[k] = n          # seul le nombre compte pour le LLM
    return out


def build_state(pa, attempts: list, geometry: str, max_attempts: int = 10,
                max_faces: int = 12, max_reasons: int = 3, memory: list | None = None) -> dict:
    """État courant d'une pièce pour le conseiller LLM.

    `attempts` : entrées du rapport (recipe, label, status, reasons, bad_faces...).
    `memory` : recettes gagnantes de pièces voisines (jalon mémoire), optionnel.
    """
    c = pa.classification
    st = dict(
        piece=dict(
            nature=pa.kind,
            epaisseur_mm=round(float(c.get("t_median", 0.0)), 3),
            paliers_mm=[round(float(z), 2) for z in (pa.thickness_zones or [])][:12],
            diagonale_mm=round(float(pa.diag), 1),
            faces=dict(reference=len(pa.ref_faces), opposee=len(pa.opp_faces), chants=len(pa.flank_faces)),
            plis=len(pa.bends),
            trous=dict(conserves=len(pa.holes_kept), bouches=len(pa.holes_filled)),
            geometrie=geometry,            # "brute" ou "micro-arêtes supprimées"
        ),
        essais=[],
    )
    for a in attempts[-max_attempts:]:
        bad = list(a.get("bad_faces") or a.get("faces_not_meshed") or [])
        e = dict(recette=_recipe_brief(a.get("recipe") or {}),
                 statut=a.get("status"),
                 raisons=[str(r)[:160] for r in (a.get("reasons") or [])[:max_reasons]])
        if bad:
            e["faces_fautives"] = [int(f) for f in bad[:max_faces]]
            if len(bad) > max_faces:
                e["faces_fautives_total"] = len(bad)
        if a.get("sj_min") is not None:
            e["jacobien_min"] = round(float(a["sj_min"]), 3)
        if a.get("internal_jacobian_min") is not None:
            e["jacobien_interne_min"] = round(float(a["internal_jacobian_min"]), 6)
        st["essais"].append(e)
    if len(attempts) > max_attempts:
        st["essais_omis"] = len(attempts) - max_attempts
    if memory:
        st["pieces_voisines"] = memory[:5]
    return st


def user_message(state: dict) -> str:
    return ("Etat courant :\n" + json.dumps(state, ensure_ascii=False, indent=1)
            + "\n\nQuel est le prochain essai ?")
