"""Описание одного запуска (без torch, чтобы планы запусков строились и без него)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunSpec:
    phase: str                  # pilot | tune | main | sens
    optimizer: str
    lr: float
    seed: int
    batch_size: int = 64
    weight_decay: float | None = None   # None: значение из grid.yaml
    steps: int = 0

    @property
    def run_id(self) -> str:
        wd = "def" if self.weight_decay is None else f"{self.weight_decay:g}"
        return f"{self.phase}-{self.optimizer}-lr{self.lr:.4g}-bs{self.batch_size}-wd{wd}-s{self.seed}"
