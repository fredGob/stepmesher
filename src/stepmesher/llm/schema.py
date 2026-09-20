"""Actions autorisées et schéma JSON de la réponse du LLM."""
from __future__ import annotations

ACTIONS = {
    "alg": "remailler les faces fautives avec un autre algorithme 2D gmsh",
    "free": "libérer les plis structurés impliqués (maillage libre au lieu de transfini)",
    "merge": "fusionner les faces étroites fautives avec leur voisine tangente",
    "size": "relancer avec une taille de maille différente (size_mult)",
    "subdiv": "stratégie subdiv : triangles puis subdivision, 100 % quads garanti",
    "microfix": "passer à la géométrie corrigée (suppression des micro-arêtes à l'import)",
    "tet": "abandonner le SC8R et passer au repli tétraédrique",
}

DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "raison"],
    "properties": {
        "action": {"type": "string", "enum": sorted(ACTIONS)},
        "size_mult": {"type": "number", "minimum": 0.25, "maximum": 3.0},
        "faces": {"type": "array", "items": {"type": "integer"}, "maxItems": 64},
        "alg": {"type": "integer", "enum": [1, 5, 6, 8]},
        "raison": {"type": "string", "maxLength": 300},
    },
}

SYSTEM_PROMPT = """Tu aides un mailleur de pièces aéronautiques (STEP CATIA -> Abaqus).
La pièce est maillée en coques volumiques SC8R (un élément dans l'épaisseur) sur une peau de
référence décalée vers la peau opposée. Quand un essai échoue, tu choisis LE prochain essai.

Règles :
- réponds uniquement par un JSON conforme au schéma ;
- une seule action par réponse, choisie dans la liste ;
- "faces" ne peut contenir que des numéros de faces présents dans "faces_fautives" de
  l'historique ; laisse la liste vide pour laisser le code choisir ;
- "tet" seulement si plusieurs essais échouent sur des éléments retournés ou sur des faces
  non maillables, ce qui trahit une géométrie hors du modèle deux peaux (jonction en T,
  nervures, chape épaisse) ;
- "microfix" seulement si des faces ne sont pas maillées du tout, ce qui évoque des
  micro-arêtes CATIA ;
- "raison" : une phrase courte en français.

Actions :
""" + "\n".join(f"- {k} : {v}" for k, v in ACTIONS.items())
