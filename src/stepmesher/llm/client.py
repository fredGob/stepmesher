"""Client llama-server minimal (urllib, aucune dépendance supplémentaire).

Sortie contrainte par schéma JSON : `response_format` sur /v1/chat/completions, repli sur
`json_schema` de /completion pour les versions plus anciennes de llama.cpp. Toute erreur
renvoie None : l'appelant repart sur ses règles.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

log = logging.getLogger("stepmesher.llm")


class LlamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", model: str = "local",
                 timeout_s: float = 60.0, temperature: float = 0.2, max_tokens: int = 256):
        self.base = base_url.rstrip("/")
        self.model, self.timeout, self.temperature, self.max_tokens = model, timeout_s, temperature, max_tokens
        self.last_error: str | None = None
        self.last_seconds: float | None = None

    # -- transport ---------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict | None:
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:      # noqa: S310 (URL locale)
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return None

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=min(self.timeout, 5)) as r:  # noqa: S310
                return r.status == 200
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            return False

    # -- décision ----------------------------------------------------------
    def ask_json(self, system: str, user: str, schema: dict) -> dict | None:
        """Renvoie le JSON produit par le modèle, ou None (serveur absent, délai, JSON cassé)."""
        t0 = time.time()
        self.last_error = None
        out = self._chat(system, user, schema)
        if out is None and self.last_error and "HTTP Error" in self.last_error:
            out = self._completion(system, user, schema)
        self.last_seconds = time.time() - t0
        if out is None:
            log.info("LLM indisponible (%s) -> règles déterministes", self.last_error)
        return out

    def _chat(self, system: str, user: str, schema: dict) -> dict | None:
        r = self._post("/v1/chat/completions", dict(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[dict(role="system", content=system), dict(role="user", content=user)],
            response_format=dict(type="json_schema",
                                 json_schema=dict(name="decision", strict=True, schema=schema))))
        if not r:
            return None
        try:
            return json.loads(r["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as e:
            self.last_error = f"réponse illisible: {type(e).__name__}"
            return None

    def _completion(self, system: str, user: str, schema: dict) -> dict | None:
        r = self._post("/completion", dict(prompt=f"{system}\n\n{user}\n", json_schema=schema,
                                           temperature=self.temperature, n_predict=self.max_tokens))
        if not r:
            return None
        try:
            return json.loads(r["content"])
        except (KeyError, TypeError, ValueError) as e:
            self.last_error = f"réponse illisible: {type(e).__name__}"
            return None
