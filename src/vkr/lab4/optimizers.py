"""Фабрика 8 оптимизаторов (torch.optim, без внешних зависимостей) по configs/grid.yaml."""
from __future__ import annotations

import numpy as np

OPTIMIZERS = ["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adam", "adamw", "radam"]


def lr_grid(grid_cfg: dict, optimizer: str) -> list[float]:
    """7 значений LR на логарифмической сетке для семейства оптимизатора."""
    g = grid_cfg["lr_grid"][grid_cfg["optimizers"][optimizer]["family"]]
    return [float(x) for x in np.logspace(np.log10(g["min"]), np.log10(g["max"]), g["n"])]


def make_optimizer(name: str, params, lr: float, grid_cfg: dict, weight_decay: float | None = None):
    import torch

    o = grid_cfg["optimizers"][name]
    wd = o.get("weight_decay", 0.0) if weight_decay is None else weight_decay
    if name in ("sgd", "momentum", "nesterov"):
        return torch.optim.SGD(params, lr=lr, momentum=o["momentum"], nesterov=o["nesterov"], weight_decay=wd)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=o.get("alpha", 0.99), weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=tuple(o["betas"]), weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=tuple(o["betas"]), weight_decay=wd)
    if name == "radam":
        return torch.optim.RAdam(params, lr=lr, betas=tuple(o["betas"]), weight_decay=wd)
    raise KeyError(name)
