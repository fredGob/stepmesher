"""Conseiller LLM local (llama-server), optionnel et jamais bloquant.

Le LLM ne choisit rien lui-même : il propose *une* action parmi une liste fermée, la
proposition est revalidée et bornée par le code, et tout échec (serveur absent, délai,
JSON invalide, action inconnue) fait repartir sur les règles déterministes.
"""
from .advisor import Advisor, Decision            # noqa: F401
from .client import LlamaClient                   # noqa: F401
from .payload import build_state, estimate_tokens  # noqa: F401
