"""Candidats déterministes : stratégies x tailles x (plis structurés / maillage libre)."""
from __future__ import annotations

from .recipe import Recipe


def deterministic_candidates(cfg, first: Recipe | None = None) -> list[Recipe]:
    """Ordre : pour chaque stratégie et chaque taille, plis structurés puis maillage
    libre (les plis structurés donnent des quads alignés, mais la transition vers une
    face en biseau peut être mauvaise ; le maillage libre s'y adapte mieux)."""
    m = cfg["mesh"]
    out: list[Recipe] = []
    if first is not None:
        out.append(first)
    for s in m["strategies"]:
        for mult in m["size_variants"]:
            for structured in ((True, False) if s in ("conform", "blossom") else (True,)):
                r = Recipe(strategy=s, size_mult=float(mult), n_per_bend=int(m["n_per_bend"]),
                           n_per_hole=int(m["n_per_hole"]), tangent_angle_deg=float(m["tangent_angle_deg"]),
                           structured=structured).clamp()
                if r not in out:
                    out.append(r)
    return out[: int(cfg["general"]["max_attempts"])]
