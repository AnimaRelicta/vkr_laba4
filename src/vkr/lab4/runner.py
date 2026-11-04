"""Планы запусков (pilot / tune / main / sens), запись runs.csv, возобновление после обрыва."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import pandas as pd

from .optimizers import OPTIMIZERS, lr_grid
from .spec import RunSpec

SCEN_COLS = ["S0", "S1", "S2", "S3", "S4", "MAIN"]
FIELDS = (["run_id", "phase", "optimizer", "lr", "seed", "batch_size", "weight_decay", "steps", "steps_done", "backend",
           "n_trainable", "val_f1_final", "val_f1_best", "steps_to_target", "unstable", "unstable_reason", "final_loss",
           "train_time_s", "peak_mem_mb"]
          + [f"test_f1_{s}" for s in SCEN_COLS] + ["test_f1_S5", "test_f1_S5_sd"] + [f"test_ece_{s}" for s in SCEN_COLS]
          + ["test_f1_main", "test_ece_main"])


def runs_path(cfg, fname: str = "runs.csv") -> Path:
    return cfg.results_dir / fname


def read_runs(cfg, fname: str = "runs.csv") -> pd.DataFrame:
    p = runs_path(cfg, fname)
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else pd.DataFrame(columns=FIELDS)


def _append(cfg, row: dict, fname: str = "runs.csv") -> None:
    p = runs_path(cfg, fname)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow(row)


def execute(specs: list[RunSpec], ctx, eval_scenarios: bool, save_probs: bool = False, fname: str = "runs.csv") -> None:
    from .train import train_run

    cfg = ctx.cfg
    done = set(read_runs(cfg, fname)["run_id"]) if runs_path(cfg, fname).exists() else set()
    todo = [s for s in specs if s.run_id not in done]
    print(f"Запусков в плане: {len(specs)}, уже выполнено: {len(specs) - len(todo)}, осталось: {len(todo)}")
    t0 = time.time()
    for i, spec in enumerate(todo, 1):
        try:
            row = train_run(spec, ctx, cfg.results_dir / "logs" / spec.phase, eval_scenarios,
                            cfg.results_dir / "probs" if save_probs else None)
        except RuntimeError as e:   # CUDA OOM и подобное: фиксируем как нестабильный запуск, идём дальше
            if "out of memory" not in str(e).lower():
                raise
            import torch
            torch.cuda.empty_cache()
            row = {**spec.__dict__, "run_id": spec.run_id, "unstable": 1, "unstable_reason": "oom", "backend": ctx.backend.name}
        _append(cfg, row, fname)
        if row.get("unstable"):
            up = cfg.results_dir / "unstable_runs.csv"
            new = not up.exists()
            with open(up, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["run_id", "optimizer", "seed", "lr", "unstable_reason", "steps_done"])
                w.writerow([row["run_id"], spec.optimizer, spec.seed, spec.lr, row.get("unstable_reason"), row.get("steps_done", "")])
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {spec.run_id}: val_f1={row.get('val_f1_final', float('nan')):.4f} "
              f"unstable={row.get('unstable_reason') or '-'}  (осталось ~{el / i * (len(todo) - i) / 60:.0f} мин)")


# ------------------------------------------------------------------ планы
def pilot_specs(cfg, seed: int = 0) -> list[RunSpec]:
    """8 оптимизаторов, один seed, LR из середины сетки (4-я из 7 точек)."""
    bs = cfg.protocol["train"]["batch_size"]
    return [RunSpec("pilot", o, lr_grid(cfg.grid, o)[3], seed, bs) for o in OPTIMIZERS]


def tune_specs(cfg) -> list[RunSpec]:
    t = cfg.protocol["train"]
    return [RunSpec("tune", o, lr, s, t["batch_size"]) for o in OPTIMIZERS for lr in lr_grid(cfg.grid, o) for s in t["tuning_seeds"]]


def select_lr(cfg) -> dict:
    """Лучший LR на оптимизатор по среднему val Macro-F1 на tuning-seed; NaN считается за 0; при равенстве берём меньший LR."""
    runs = read_runs(cfg)
    runs = runs[runs["phase"] == "tune"].copy()
    runs["val_f1_final"] = pd.to_numeric(runs["val_f1_final"], errors="coerce").fillna(0.0)
    chosen, table = {}, []
    for o, g in runs.groupby("optimizer"):
        m = g.groupby("lr")["val_f1_final"].mean().sort_index()
        best = m.max()
        chosen[o] = float(m[m >= best - 1e-9].index.min())
        table += [{"optimizer": o, "lr": lr, "val_f1_mean": v, "selected": lr == chosen[o]} for lr, v in m.items()]
    d = cfg.results_dir / "tuning"
    d.mkdir(parents=True, exist_ok=True)
    (d / "selected_lr.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    pd.DataFrame(table).to_csv(d / "tuning_table.csv", index=False, encoding="utf-8-sig")
    return chosen


def load_selected_lr(cfg) -> dict:
    p = cfg.results_dir / "tuning" / "selected_lr.json"
    if not p.exists():
        raise SystemExit("Нет results/tuning/selected_lr.json: сначала `lab4 tune` и `lab4 select-lr`.")
    return json.loads(p.read_text(encoding="utf-8"))


def main_specs(cfg) -> list[RunSpec]:
    t, lrs = cfg.protocol["train"], load_selected_lr(cfg)
    return [RunSpec("main", o, lrs[o], s, t["batch_size"]) for o in OPTIMIZERS for s in t["eval_seeds"]]


def sens_specs(cfg, optimizers: list[str]) -> list[RunSpec]:
    """Один фактор за раз: LR x{0.1, 1/3, 1, 3, 10}, batch size, weight decay; 5 seed."""
    t, g, lrs = cfg.protocol["train"], cfg.grid["sensitivity"], load_selected_lr(cfg)
    specs: dict[str, RunSpec] = {}
    for o in optimizers:
        for s in t["sens_seeds"]:
            for m in g["lr_multipliers"]:
                sp = RunSpec("sens", o, lrs[o] * m, s, t["batch_size"])
                specs[sp.run_id] = sp
            for bs in g["batch_sizes"]:
                sp = RunSpec("sens", o, lrs[o], s, bs)
                specs[sp.run_id] = sp
            for wd in g["weight_decays"]:
                sp = RunSpec("sens", o, lrs[o], s, t["batch_size"], weight_decay=wd)
                specs[sp.run_id] = sp
    return list(specs.values())


def top_for_sensitivity(cfg) -> list[str]:
    from . import stats as S
    runs = read_runs(cfg)
    main = runs[runs["phase"] == "main"]
    if main.empty:
        raise SystemExit("Нет запусков main: sensitivity строится после основного прогона.")
    sel = cfg.protocol["selection"]
    summary = S.summarize(main, sel, cfg.protocol["train"]["steps"])
    score = S.composite(summary, sel["weights"]).sort_values(ascending=False)
    top = list(score.index[: cfg.grid["sensitivity"]["optimizers_top"]])
    return top + ([] if "sgd" in top else ["sgd"])
