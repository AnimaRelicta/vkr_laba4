"""Загрузка конфигов и защита протокола хешем (протокол не меняется после заморозки)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs"
LOCKED_FILES = ["protocol.yaml", "grid.yaml", "mutations.yaml", "scenarios.yaml"]
LOCK_PATH = ROOT / "protocol.lock.json"


def load_yaml(name: str, config_dir: Path | None = None) -> dict:
    with open((config_dir or CONFIG_DIR) / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Config:
    def __init__(self, config_dir: Path | None = None):
        self.dir = config_dir or CONFIG_DIR
        self.protocol = load_yaml("protocol.yaml", self.dir)
        self.grid = load_yaml("grid.yaml", self.dir)
        self.mutations = load_yaml("mutations.yaml", self.dir)
        self.scenarios = load_yaml("scenarios.yaml", self.dir)

    @property
    def backend(self) -> str:
        return self.protocol["model"]["backend"]

    @property
    def results_dir(self) -> Path:
        """Результаты разных моделей не смешиваются: results/<backend>/."""
        return ROOT / self.protocol["paths"]["results"] / self.backend

    @property
    def artifacts_dir(self) -> Path:
        return ROOT / "artifacts"


def _normalized(path: Path) -> bytes:
    """Хеш не должен зависеть от флага frozen и переводов строк."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.name == "protocol.yaml":
        data = {k: v for k, v in data.items() if k != "frozen"}
    return json.dumps(data, sort_keys=True, ensure_ascii=False).encode()


def protocol_hash(config_dir: Path | None = None) -> str:
    d = config_dir or CONFIG_DIR
    h = hashlib.sha256()
    for name in LOCKED_FILES:
        h.update(name.encode())
        h.update(_normalized(d / name))
    return h.hexdigest()


def write_lock(config_dir: Path | None = None, lock_path: Path | None = None) -> str:
    digest = protocol_hash(config_dir)
    (lock_path or LOCK_PATH).write_text(json.dumps({"sha256": digest, "files": LOCKED_FILES}, indent=1), encoding="utf-8")
    return digest


def check_lock(config_dir: Path | None = None, lock_path: Path | None = None) -> None:
    """Бросает RuntimeError, если протокол заморожен и после этого менялся."""
    lp = lock_path or LOCK_PATH
    if not lp.exists():
        raise RuntimeError("Протокол не заморожен: сначала `uv run lab4 freeze --steps N --target F`.")
    locked = json.loads(lp.read_text(encoding="utf-8"))["sha256"]
    if locked != protocol_hash(config_dir):
        raise RuntimeError(
            "configs/*.yaml изменены после заморозки протокола (protocol.lock.json). "
            "Верните значения или заморозьте заново и явно отметьте это в отчёте.")
