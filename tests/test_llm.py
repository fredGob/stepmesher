"""Conseiller LLM : charge utile bornée, décisions valides appliquées, tout le reste ignoré.

Un faux llama-server (http.server) répond ce que le test veut ; aucun modèle n'est requis.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from stepmesher.llm.advisor import Advisor
from stepmesher.llm.client import LlamaClient
from stepmesher.llm.payload import build_state, estimate_tokens
from stepmesher.llm.schema import ACTIONS, DECISION_SCHEMA
from stepmesher.strategy.recipe import Recipe


class _Handler(BaseHTTPRequestHandler):
    reply = {"action": "alg", "raison": "faces fautives isolées"}
    status = 200

    def log_message(self, *a):
        pass

    def do_GET(self):                      # /health
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        type(self).last_request = body
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        out = dict(choices=[dict(message=dict(content=json.dumps(type(self).reply)))])
        self.wfile.write(json.dumps(out).encode())


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _advisor(url, **kw):
    return Advisor(enabled=True, client=LlamaClient(base_url=url, timeout_s=5, **kw))


def _pa(n_attempts=0):
    return SimpleNamespace(kind="variable", classification=dict(t_median=3.2), thickness_zones=[3.2, 5.9],
                           diag=4000.0, ref_faces=[1, 2], opp_faces=list(range(50)), flank_faces=list(range(30)),
                           bends=[dict(faces=[7, 8])], holes_kept=[], holes_filled=[{}])


def _attempts(n, bad=(4, 10, 43)):
    return [dict(index=i, recipe=Recipe().to_dict(), status="failed",
                 reasons=[f"jacobien normalisé min 0.0{i} < 0.1 (critère dur, {i + 1} élément(s))"],
                 bad_faces=list(bad), sj_min=0.05) for i in range(n)]


def _res(bad=(4, 10, 43)):
    return dict(bad_faces=list(bad), implicated_structured=[7], sliver_merges=[[4, 5]])


# --- charge utile ---------------------------------------------------------

def test_payload_is_bounded():
    st = build_state(_pa(), _attempts(30, bad=tuple(range(60))), "brute", max_attempts=10, max_faces=12)
    assert len(st["essais"]) == 10
    assert st["essais_omis"] == 20
    assert all(len(e["faces_fautives"]) == 12 for e in st["essais"])
    assert estimate_tokens(st) < 5000        # tient largement dans un contexte de 8192


def test_payload_contains_part_summary():
    st = build_state(_pa(), _attempts(2), "brute")
    assert st["piece"]["nature"] == "variable"
    assert st["piece"]["paliers_mm"] == [3.2, 5.9]
    assert st["piece"]["geometrie"] == "brute"


# --- décisions ------------------------------------------------------------

def test_valid_decision_gives_recipe(server):
    _Handler.reply = {"action": "alg", "faces": [4, 10], "alg": 6, "raison": "deux faces en biseau"}
    d = _advisor(server).propose(build_state(_pa(), _attempts(2), "brute"), Recipe(), _res(), {7, 8}, set())
    assert d.action == "alg"
    assert dict(d.recipe.face_alg) == {4: 6, 10: 6}
    assert "biseau" in d.reason


def test_free_lever_limited_to_bends(server):
    _Handler.reply = {"action": "free", "raison": "pli structuré fautif"}
    d = _advisor(server).propose(build_state(_pa(), _attempts(2), "brute"), Recipe(), _res(), {7, 8}, set())
    assert d.recipe.free_faces == (7,)


def test_invented_faces_are_filtered(server):
    _Handler.reply = {"action": "alg", "faces": [9999], "raison": "face inventée"}
    d = _advisor(server).propose(build_state(_pa(), _attempts(2), "brute"), Recipe(), _res(), {7, 8}, set())
    assert d.recipe is None and "sans effet" in d.rejected


def test_unknown_action_rejected(server):
    _Handler.reply = {"action": "rm -rf", "raison": "n'importe quoi"}
    d = _advisor(server).propose(build_state(_pa(), _attempts(2), "brute"), Recipe(), _res(), {7, 8}, set())
    assert d.action is None and "inconnue" in d.rejected


def test_size_is_clamped(server):
    _Handler.reply = {"action": "size", "size_mult": 99.0, "raison": "beaucoup trop gros"}
    d = _advisor(server).propose(build_state(_pa(), _attempts(2), "brute"), Recipe(), _res(), {7, 8}, set())
    assert d.recipe.size_mult == 3.0         # borne haute de BOUNDS


def test_tet_and_microfix_signals(server):
    _Handler.reply = {"action": "tet", "raison": "jonction en T"}
    d = _advisor(server).propose(build_state(_pa(), _attempts(4), "brute"), Recipe(), _res(), {7, 8}, set())
    assert d.action == "tet" and d.recipe is None
    _Handler.reply = {"action": "microfix", "raison": "faces non maillées"}
    a = _advisor(server)
    assert a.propose(build_state(_pa(), _attempts(2), "brute"), Recipe(), _res(), {7, 8}, set(),
                     allow_microfix=True).action == "microfix"
    assert a.propose(build_state(_pa(), _attempts(2), "brute"), Recipe(), _res(), {7, 8}, set(),
                     allow_microfix=False).action is None


def test_schema_is_sent_to_server(server):
    _Handler.reply = {"action": "alg", "raison": "ok"}
    _advisor(server).propose(build_state(_pa(), _attempts(2), "brute"), Recipe(), _res(), {7, 8}, set())
    rf = _Handler.last_request["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == DECISION_SCHEMA
    assert set(DECISION_SCHEMA["properties"]["action"]["enum"]) == set(ACTIONS)


# --- pannes ---------------------------------------------------------------

def test_server_down_is_not_blocking():
    a = Advisor(enabled=True, client=LlamaClient(base_url="http://127.0.0.1:1", timeout_s=2))
    d = a.propose(build_state(_pa(), _attempts(1), "brute"), Recipe(), _res(), {7, 8}, set())
    assert d.action is None and d.recipe is None and d.rejected
    assert a.decisions and a.decisions[0]["action"] is None


def test_disabled_advisor_does_nothing():
    d = Advisor(enabled=False).propose(build_state(_pa(), _attempts(1), "brute"), Recipe(), _res(), {7, 8}, set())
    assert d.action is None and "désactivé" in d.rejected


def test_broken_json_is_ignored(server):
    a = _advisor(server)
    _Handler.reply = {"action": "alg"}       # "raison" manquante : toléré, champ optionnel côté code
    d = a.propose(build_state(_pa(), _attempts(2), "brute"), Recipe(), _res(), {7, 8}, set())
    assert d.action == "alg" and d.reason == ""
