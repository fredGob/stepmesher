"""Famille de pièce (vocabulaire de Fred : frame, intercostale, fitting, clip, clip_autre,
peau ; les stabilos sont des clips pour le maillage). Pilote la voie de maillage via
`[family] tet_direct`."""
from __future__ import annotations

FAMILIES = ("frame", "intercostale", "fitting", "clip", "clip_autre", "peau")


def classify_family(dims, t: float, thickness_kind: str, bend_angles, cfg: dict) -> tuple[str, str]:
    """Renvoie (famille, raison). `dims` : dimensions OBB (mm), `bend_angles` : degrés."""
    L, W, H = sorted((float(x) for x in dims), reverse=True)
    nb = sum(1 for a in bend_angles if a >= cfg["min_bend_deg"])
    if L >= cfg["frame_min_length_mm"]:
        return "frame", f"longueur {L:.0f} mm >= {cfg['frame_min_length_mm']:g}"
    if thickness_kind in ("variable", "massive"):
        if L < cfg["fitting_max_length_mm"]:
            return "fitting", f"usinée (épaisseur {thickness_kind}), longueur {L:.0f} mm"
        return "clip_autre", f"usinée (épaisseur {thickness_kind}), longueur {L:.0f} mm"
    if nb == 0 and t > 0 and H <= cfg["peau_max_height_t"] * t:
        return "peau", f"sans pli, hauteur {H:.0f} mm <= {cfg['peau_max_height_t']:g} t"
    if L >= cfg["intercostale_min_length_mm"] and nb >= 2:
        return "intercostale", f"longueur {L:.0f} mm, {nb} plis"
    return "clip", f"longueur {L:.0f} mm, largeur {W:.0f} mm, {nb} pli(s)"
