"""Chargement de la configuration TOML (défauts + surcharge utilisateur)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

DEFAULT_PATH = Path(__file__).with_name("default.toml")


def _deep_update(base: dict, over: dict, path: str = "") -> dict:
    for k, v in over.items():
        if k not in base:
            raise KeyError(f"clé de configuration inconnue : {path}{k}")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise TypeError(f"{path}{k} doit être une section")
            _deep_update(base[k], v, f"{path}{k}.")
        else:
            base[k] = v
    return base


@dataclass
class Config:
    data: dict = field(default_factory=dict)

    def __getitem__(self, section: str) -> dict:
        return self.data[section]

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)

    def hole_threshold(self, diag: float) -> float:
        """Diamètre sous lequel un trou est bouché (0 = pas de bouchage)."""
        h = self.data["holes"]
        if not h["fill"]:
            return 0.0
        vals = [v for v in (h["max_diameter_mm"], h["max_diameter_frac"] * diag) if v > 0]
        return min(vals) if vals else 0.0

    def target_size(self, diag: float, thickness: float) -> float:
        m = self.data["mesh"]
        h = m["size_frac"] * diag
        if thickness > 0:
            h = min(h, m["thickness_factor"] * thickness)
        return float(min(max(h, m["min_size_mm"]), m["max_size_mm"]))


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    with open(DEFAULT_PATH, "rb") as f:
        data = tomllib.load(f)
    if path:
        with open(path, "rb") as f:
            _deep_update(data, tomllib.load(f))
    if overrides:
        _deep_update(data, overrides)
    return Config(data)
