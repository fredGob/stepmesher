"""Leviers de recette : une modification élémentaire d'une recette en échec.

Utilisés par la montée gloutonne déterministe (`process._sc8r_pass`) et par le conseiller
LLM, qui ne fait que *choisir* le levier : la construction de la recette reste ici, donc
une proposition du LLM ne peut pas produire de recette invalide.
"""
from __future__ import annotations

from .recipe import BOUNDS, Recipe

LEVERS = ("alg", "free", "merge", "size", "subdiv")
ALG_ORDER = (6, 1, 5)


def apply_lever(lever: str, base: Recipe, res: dict, bend_faces: set, tried_alg: set,
                faces: list | None = None, alg: int | None = None,
                size_mult: float | None = None) -> Recipe | None:
    """Nouvelle recette obtenue en appliquant un levier, ou None si le levier n'apporte rien.

    `faces` restreint le levier à ces faces (proposition LLM) ; elles sont filtrées sur les
    faces réellement fautives de l'essai. `tried_alg` est mis à jour (couples déjà essayés).
    """
    bad = list(res.get("bad_faces") or res.get("faces_not_meshed") or [])
    if faces:
        bad = [f for f in bad if f in set(faces)]
    d = base.to_dict()

    if lever == "alg":
        algs = dict(base.face_alg)
        ch = {}
        for f in bad:
            cur = algs.get(f, 8)
            nxt = alg if (alg is not None and alg != cur and (f, alg) not in tried_alg) else \
                next((a for a in ALG_ORDER if a != cur and (f, a) not in tried_alg), None)
            if nxt is not None:
                ch[f] = nxt
        if not ch:
            return None
        tried_alg.update(ch.items())
        algs.update(ch)
        d["face_alg"] = tuple(sorted(algs.items()))

    elif lever == "free":
        impl = set(res.get("implicated_structured") or ())
        if faces:
            impl &= set(faces)
        if not base.structured or not impl:
            return None
        free = set(base.free_faces) | (impl & bend_faces)
        if free == set(base.free_faces) or free == bend_faces:
            return None
        d["free_faces"] = tuple(sorted(free))

    elif lever == "merge":
        pairs = [tuple(p) for p in (res.get("sliver_merges") or ())]
        if faces:
            pairs = [p for p in pairs if p[0] in set(faces)]
        new_m = set(map(tuple, base.merge)) | set(pairs)
        if new_m == set(map(tuple, base.merge)):
            return None
        d["merge"] = tuple(sorted(new_m))

    elif lever == "size":
        lo, hi = BOUNDS["size_mult"]
        m = float(size_mult) if size_mult else base.size_mult * 0.7
        m = min(max(m, lo), hi)
        if abs(m - base.size_mult) < 1e-6:
            return None
        d["size_mult"] = m

    elif lever == "subdiv":
        if base.strategy == "subdiv":
            return None
        d["strategy"] = "subdiv"

    else:
        return None
    return Recipe(**d).clamp()
