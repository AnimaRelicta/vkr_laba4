"""Бюджет времени и режим smoke (без torch, поэтому тестируется на любой машине)."""
from __future__ import annotations

from .optimizers import OPTIMIZERS, lr_grid
from .runner import main_specs, pilot_specs, sens_specs, tune_specs

_TOP_GUESS = ["adamw", "adam", "nesterov", "momentum"]


def plan_specs(cfg, lrs: dict | None = None) -> dict:
    """Все фазы с реальными списками запусков. LR до `select-lr` неизвестны: подставляется 4-я точка сетки (на число запусков не влияет)."""
    lrs = lrs or {o: lr_grid(cfg.grid, o)[min(3, len(lr_grid(cfg.grid, o)) - 1)] for o in OPTIMIZERS}
    top = _TOP_GUESS[: cfg.grid["sensitivity"]["optimizers_top"]] + ["sgd"]
    return {"pilot": pilot_specs(cfg), "tune": tune_specs(cfg), "main": main_specs(cfg, lrs), "sens": sens_specs(cfg, top, lrs)}


def plan_cost(cfg, per_step_s: float, overhead_s, lrs: dict | None = None) -> dict:
    """Фаза -> (запусков, шагов в единицах batch=64, секунд). Стоимость шага пропорциональна batch size.

    overhead_s: число (одинаково для всех фаз) или dict {фаза: секунд на запуск}: в pilot и tune нет оценки сценариев S0-S5, там накладные расходы меньше."""
    t = cfg.protocol["train"]
    out = {}
    for phase, specs in plan_specs(cfg, lrs).items():
        over = overhead_s[phase] if isinstance(overhead_s, dict) else overhead_s
        units = sum((s.steps or t["steps"]) * s.batch_size / t["batch_size"] for s in specs)
        out[phase] = (len(specs), units, units * per_step_s + len(specs) * over)
    return out


def smoke_config(cfg, steps: int = 8):
    """Уменьшенный протокол для сквозной проверки кода (несколько минут). Результаты идут в results_smoke/, не в results/."""
    t = cfg.protocol["train"]
    t.update(steps=steps, eval_every=max(1, steps // 2), log_every=max(1, steps // 4), tuning_seeds=[100],
             eval_seeds=[0, 1], sens_seeds=[0], time_limit_min=5)
    for fam in cfg.grid["lr_grid"].values():
        fam["n"] = 2                     # концы сетки: проверяется и расходящийся запуск (SGD lr=1)
    cfg.grid["sensitivity"].update(optimizers_top=2, lr_multipliers=[1.0, 3.0], batch_sizes=[32, 64], weight_decays=[0.0, 0.01])
    cfg.protocol["paths"]["results"] = "results_smoke"
    return cfg
