"""Recette de maillage : paramètres relatifs, donc transposables d'une pièce à une autre."""
from __future__ import annotations

from dataclasses import asdict, dataclass

STRATEGIES = ("conform", "compound", "blossom", "subdiv", "qqs")

# bornes appliquées à toute recette (y compris proposée par un LLM au jalon 4)
BOUNDS = dict(size_mult=(0.25, 3.0), n_per_bend=(2, 12), n_per_hole=(6, 48), tangent_angle_deg=(1.0, 30.0))


@dataclass
class Recipe:
    strategy: str = "conform"
    size_mult: float = 1.0          # multiplicateur de la taille cible issue de la config
    n_per_bend: int = 4
    n_per_hole: int = 12
    tangent_angle_deg: float = 10.0
    structured: bool = True         # plis (et petites faces à 4 côtés) en maillage transfini
    free_faces: tuple = ()          # plis laissés en maillage libre (recette adaptative)
    face_alg: tuple = ()            # ((face, algorithme 2D gmsh), ...) : remaillage local adaptatif
    merge: tuple = ()               # ((face étroite, voisine tangente), ...) : surfaces composites locales

    def clamp(self) -> "Recipe":
        if self.strategy not in STRATEGIES:
            raise ValueError(f"stratégie inconnue : {self.strategy}")
        for k, (lo, hi) in BOUNDS.items():
            v = getattr(self, k)
            setattr(self, k, type(v)(min(max(v, lo), hi)))
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d["free_faces"] = tuple(d.get("free_faces", ()))
        d["face_alg"] = tuple(tuple(x) for x in d.get("face_alg", ()))
        d["merge"] = tuple(tuple(x) for x in d.get("merge", ()))
        return cls(**d).clamp()

    def label(self) -> str:
        mode = "" if self.structured else "-libre"
        if self.structured and self.free_faces:
            mode = f"-mixte({len(self.free_faces)} plis libres)"
        if self.face_alg:
            mode += f"+{len(self.face_alg)} faces remaillées"
        if self.merge:
            mode += f"+{len(self.merge)} fusions"
        return f"{self.strategy}{mode}×{self.size_mult:g}"
