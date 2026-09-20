"""Conseiller : interroge le LLM local et traduit sa réponse en recette valide.

Garanties : une action hors liste, des faces inventées, une taille aberrante ou un serveur
absent ne produisent jamais de recette — l'appelant retombe alors sur ses règles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..strategy.levers import LEVERS, apply_lever
from ..strategy.recipe import Recipe
from .client import LlamaClient
from .payload import estimate_tokens, user_message
from .schema import ACTIONS, DECISION_SCHEMA, SYSTEM_PROMPT

log = logging.getLogger("stepmesher.llm")


@dataclass
class Decision:
    action: str | None = None           # levier appliqué, "tet", "microfix" ou None
    recipe: Recipe | None = None
    reason: str = ""
    raw: dict | None = None
    rejected: str = ""                  # pourquoi la proposition n'a pas été retenue
    seconds: float | None = None
    tokens_in: int | None = None

    def log_entry(self) -> dict:
        return dict(action=self.action, recette=self.recipe.label() if self.recipe else None,
                    raison=self.reason, rejet=self.rejected or None, secondes=self.seconds,
                    tokens_estimes=self.tokens_in, reponse=self.raw)


@dataclass
class Advisor:
    """Enveloppe du client LLM : désactivable, silencieuse en cas de panne."""
    client: LlamaClient | None = None
    enabled: bool = False
    decisions: list = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg) -> "Advisor":
        c = cfg["llm"]
        if not c.get("enabled"):
            return cls(enabled=False)
        return cls(enabled=True, client=LlamaClient(
            base_url=c["base_url"], model=c["model"], timeout_s=float(c["timeout_s"]),
            temperature=float(c["temperature"]), max_tokens=int(c["max_tokens"])))

    def propose(self, state: dict, base: Recipe, res: dict, bend_faces: set,
                tried_alg: set, allow_microfix: bool = False) -> Decision:
        if not self.enabled or self.client is None:
            return Decision(rejected="LLM désactivé")
        d = Decision(tokens_in=estimate_tokens(state))
        raw = self.client.ask_json(SYSTEM_PROMPT, user_message(state), DECISION_SCHEMA)
        d.seconds, d.raw = self.client.last_seconds, raw
        if raw is None:
            d.rejected = self.client.last_error or "pas de réponse"
        else:
            self._interpret(raw, d, base, res, bend_faces, tried_alg, allow_microfix)
        self.decisions.append(d.log_entry())
        if d.action:
            log.info("LLM -> %s (%s) en %.1fs", d.action, d.reason[:80], d.seconds or 0.0)
        elif d.rejected:
            log.info("LLM ignoré : %s", d.rejected)
        return d

    @staticmethod
    def _interpret(raw: dict, d: Decision, base: Recipe, res: dict, bend_faces: set,
                   tried_alg: set, allow_microfix: bool):
        action = str(raw.get("action", "")).strip().lower()
        d.reason = str(raw.get("raison", ""))[:300]
        if action not in ACTIONS:
            d.rejected = f"action inconnue : {action[:30]}"
            return
        if action == "tet":
            d.action = "tet"
            return
        if action == "microfix":
            if not allow_microfix:
                d.rejected = "microfix indisponible (pas de correction de micro-arêtes acceptée)"
                return
            d.action = "microfix"
            return
        if action not in LEVERS:
            d.rejected = f"levier non applicable : {action}"
            return
        faces = [int(f) for f in (raw.get("faces") or [])][:64]
        rc = apply_lever(action, base, res, bend_faces, tried_alg, faces=faces or None,
                         alg=raw.get("alg"), size_mult=raw.get("size_mult"))
        if rc is None:
            d.rejected = f"levier {action} sans effet sur cette recette"
            return
        d.action, d.recipe = action, rc
