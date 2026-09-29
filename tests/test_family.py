"""Famille de pièce : les 44 pièces de parts_stp_echelle classées par Fred (vérité terrain),
caractéristiques relevées sur la campagne result_v5 (OBB, épaisseur, plis)."""
import pytest

from stepmesher.analyze.family import classify_family
from stepmesher.config import load_config

CASES = [
    ("000", (179, 139, 30), 1.86, "constant", [85], "clip"),
    ("001", (590, 160, 84), 3.74, "variable", [46, 35, 35, 35, 35, 51, 48], "clip_autre"),
    ("002", (617, 133, 30), 3.41, "constant", [82], "clip"),
    ("003", (617, 126, 31), 3.41, "constant", [97], "clip"),
    ("004", (286, 138, 81), 2.17, "constant", [90, 105, 90], "clip"),
    ("005", (161, 139, 71), 1.86, "constant", [83], "clip"),
    ("006", (164, 138, 29), 3.41, "constant", [82], "clip"),
    ("007", (163, 130, 30), 3.41, "constant", [96], "clip"),
    ("008", (138, 49, 38), 2.17, "constant", [90, 91], "stabilo"),
    ("009", (140, 49, 39), 1.86, "constant", [89, 90], "stabilo"),
    ("010", (4440, 1056, 43), 2.39, "variable", [162, 180, 180, 91], "frame"),
    ("011", (182, 159, 69), 2.17, "constant", [84, 173, 90], "clip"),
    ("012", (245, 134, 70), 2.17, "constant", [90, 91, 83], "clip"),
    ("013", (182, 155, 71), 2.17, "constant", [90, 84, 92], "clip"),
    ("014", (330, 132, 71), 1.86, "constant", [83, 89, 90], "clip"),
    ("015", (150, 136, 27), 2.79, "constant", [82], "clip"),
    ("016", (168, 54, 41), 3.80, "constant", [90, 91], "stabilo"),
    ("017", (166, 54, 41), 3.80, "constant", [91, 91], "stabilo"),
    ("018", (215, 166, 76), 3.80, "constant", [121], "clip"),
    ("019", (181, 117, 34), 3.50, "variable", [], "fitting"),
    ("020", (202, 119, 35), 3.80, "variable", [30, 30], "fitting"),
    ("021", (566, 145, 44), 3.08, "constant", [90, 180], "intercostale"),
    ("022", (179, 89, 73), 6.00, "variable", [], "fitting"),
    ("023", (568, 162, 41), 3.08, "constant", [117, 90], "intercostale"),
    ("024", (170, 86, 71), 6.00, "variable", [35], "fitting"),
    ("025", (570, 157, 37), 3.08, "constant", [162, 90], "intercostale"),
    ("026", (166, 85, 70), 4.01, "variable", [], "fitting"),
    ("027", (560, 154, 38), 2.71, "constant", [174, 90], "intercostale"),
    ("028", (163, 86, 70), 4.00, "variable", [], "fitting"),
    ("029", (540, 148, 30), 2.71, "constant", [180, 90], "intercostale"),
    ("030", (161, 86, 69), 4.01, "variable", [], "fitting"),
    ("031", (514, 149, 42), 3.08, "constant", [90, 159], "intercostale"),
    ("032", (165, 89, 70), 6.00, "variable", [], "fitting"),
    ("033", (500, 149, 42), 3.08, "constant", [169, 90], "intercostale"),
    ("034", (166, 89, 71), 6.00, "variable", [], "fitting"),
    ("035", (175, 112, 34), 3.80, "variable", [30, 30], "fitting"),
    ("036", (4383, 1064, 133), 6.05, "variable", [92, 91, 152, 131, 109, 104], "frame"),
    ("037", (110, 67, 28), 3.08, "constant", [95], "stabilo"),
    ("038", (110, 67, 28), 3.08, "constant", [95], "stabilo"),
    ("039", (114, 65, 28), 2.71, "constant", [94], "stabilo"),
    ("040", (108, 66, 28), 2.71, "constant", [94], "stabilo"),
    ("041", (108, 66, 28), 2.71, "constant", [93], "stabilo"),
    ("042", (110, 65, 28), 2.71, "constant", [93], "stabilo"),
    ("043", (129, 70, 44), 2.71, "constant", [88], "stabilo"),
]


@pytest.mark.parametrize("part,dims,t,kind,bends,expected", CASES, ids=[c[0] for c in CASES])
def test_family_matches_fred(part, dims, t, kind, bends, expected):
    expected = "clip" if expected == "stabilo" else expected     # stabilo = clip pour le maillage (Fred)
    assert classify_family(dims, t, kind, bends, load_config()["family"])[0] == expected


def test_flat_sheet_is_skin():
    assert classify_family((400, 300, 4), 2.0, "constant", [], load_config()["family"])[0] == "peau"
