"""Командная строка ЛР4: `uv run lab4 <команда>`. Порядок команд описан в README.md."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .config import CONFIG_DIR, ROOT, Config, check_lock, write_lock


def _cfg(args) -> Config:
    cfg = Config()
    if getattr(args, "backend", None):
        cfg.protocol["model"]["backend"] = args.backend
    if getattr(args, "data", None):
        cfg.protocol["data"]["dir"] = args.data
    return cfg


def _data_dir(cfg: Config) -> Path:
    p = Path(cfg.protocol["data"]["dir"])
    return p if p.is_absolute() else ROOT / p


def _context(cfg: Config, steps: int | None = None):
    """Тяжёлые импорты (torch) только здесь, чтобы passport/stats/status работали без GPU."""
    import torch

    from .data import load_observations, split_frames
    from .models import make_backend
    from .train import Context

    if steps:
        cfg.protocol["train"]["steps"] = steps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and cfg.backend == "qwen":
        print("ВНИМАНИЕ: CUDA недоступна, Qwen будет считаться на CPU очень медленно. Проверьте установку torch (README, «Проверка GPU»).")
    print(f"Устройство: {device}; модель: {cfg.backend}; данные: {_data_dir(cfg)}")
    df = load_observations(_data_dir(cfg))
    return Context(cfg, split_frames(df), make_backend(cfg.protocol), device)


# ------------------------------------------------------------------ команды
def cmd_demo_data(args):
    from .demo_data import make_demo
    out = make_demo(args.out)
    print(f"Демо-данные (НЕ реальные!) записаны в {out}. Запуск: uv run lab4 passport --data {out}")


def cmd_passport(args):
    from .data import load_observations, write_passport
    cfg = _cfg(args)
    df = load_observations(_data_dir(cfg))
    leaks = write_passport(df, cfg.artifacts_dir)
    print(f"Записано: {cfg.artifacts_dir}/data_passport.md, split.json. Утечек нет: {leaks['clean']}")
    if not leaks["clean"]:
        print("ВНИМАНИЕ: проверка утечек не пройдена, подробности в split.json")
        sys.exit(1)


def cmd_check_data(args):
    from .verify import build_check
    cfg = _cfg(args)
    dirs = [Path(x) for x in (args.dir or [str(_data_dir(cfg))])]
    dirs = [d if d.is_absolute() else ROOT / d for d in dirs]
    llm = [Path(x) if Path(x).is_absolute() else ROOT / x for x in args.llm_dir]
    md, summ = build_check(dirs, llm, args.claim_headers, args.claim_classes)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.artifacts_dir / "data_check.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nЗаписано: {out}")


def cmd_freeze(args):
    p = CONFIG_DIR / "protocol.yaml"
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"(?m)^(  steps:\s*)\d+", rf"\g<1>{args.steps}", s, count=1)
    s = re.sub(r"(?m)^(  target_val_f1:\s*)[0-9.]+", rf"\g<1>{args.target}", s, count=1)
    s = re.sub(r"(?m)^frozen:\s*\w+", "frozen: true", s, count=1)
    p.write_text(s, encoding="utf-8")
    print(f"Протокол заморожен: steps={args.steps}, target_val_f1={args.target}, sha256={write_lock()[:16]}…")
    print("Дальше configs/*.yaml менять нельзя; запишите значения в PROTOCOL.md.")


def cmd_bench(args):
    import time

    import torch

    from .optimizers import OPTIMIZERS
    cfg = _cfg(args)
    ctx = _context(cfg)
    n, plan_default = args.steps, cfg.protocol["train"]["steps"]
    cfg.protocol["train"].update({"steps": n, "eval_every": 10**9, "target_val_f1": 2.0})
    from .spec import RunSpec
    from .train import train_run
    t0 = time.perf_counter()
    row = train_run(RunSpec("bench", "adamw", 1e-3, 0, steps=n), ctx, None, eval_scenarios=False)
    dt = time.perf_counter() - t0
    per = row["train_time_s"] / max(1, row["steps_done"])
    t = cfg.protocol["train"]
    plan = {"tuning (112)": len(OPTIMIZERS) * 7 * len(t["tuning_seeds"]), "main (80)": len(OPTIMIZERS) * len(t["eval_seeds"]), "sensitivity (~180)": 180}
    print(f"{per:.3f} с/шаг, пиковая память {row['peak_mem_mb']:.0f} МБ, обучаемых параметров {row['n_trainable']:,} (всего {dt:.1f} с на {n} шагов)")
    steps = args.plan_steps or plan_default
    for k, v in plan.items():
        print(f"  {k}: {v} запусков x {steps} шагов ≈ {v * steps * per / 3600:.1f} ч (без оценки на val/test)")


def cmd_pilot(args):
    from .runner import execute, pilot_specs, runs_path
    cfg = _cfg(args)
    ctx = _context(cfg, args.steps)
    runs_path(cfg, "pilot_runs.csv").unlink(missing_ok=True)
    execute(pilot_specs(cfg), ctx, eval_scenarios=False, fname="pilot_runs.csv")
    print(f"Готово: {runs_path(cfg, 'pilot_runs.csv')}, кривые в results/{cfg.backend}/logs/pilot. "
          "Выберите steps и target_val_f1 и выполните `uv run lab4 freeze --steps N --target F`.")


def cmd_tune(args):
    from .runner import execute, tune_specs
    cfg = _cfg(args)
    check_lock()
    execute(tune_specs(cfg), _context(cfg), eval_scenarios=False)


def cmd_select_lr(args):
    from .runner import select_lr
    cfg = _cfg(args)
    for o, lr in select_lr(cfg).items():
        print(f"{o:9s} lr={lr:.4g}")


def cmd_main(args):
    from .runner import execute, main_specs
    cfg = _cfg(args)
    check_lock()
    execute(main_specs(cfg), _context(cfg), eval_scenarios=True, save_probs=True)


def cmd_sens(args):
    from .runner import execute, sens_specs, top_for_sensitivity
    cfg = _cfg(args)
    check_lock()
    opts = top_for_sensitivity(cfg)
    print("Sensitivity для:", opts)
    execute(sens_specs(cfg, opts), _context(cfg), eval_scenarios=True)


def cmd_stats(args):
    from .report import build_report
    from .runner import read_runs
    cfg = _cfg(args)
    print("Отчёт:", build_report(cfg, read_runs(cfg)))


def cmd_status(args):
    from .runner import read_runs, runs_path
    cfg = _cfg(args)
    runs = read_runs(cfg)
    print(f"{runs_path(cfg)}: {len(runs)} запусков")
    if len(runs):
        print(runs.groupby(["phase", "optimizer"]).size().unstack(fill_value=0).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lab4", description=__doc__)
    ap.add_argument("--backend", choices=["qwen", "ngram"], help="модель (по умолчанию из configs/protocol.yaml)")
    ap.add_argument("--data", help="папка с observations.csv (по умолчанию из protocol.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        return p

    p = add("demo-data", cmd_demo_data, "создать СИНТЕТИЧЕСКИЕ данные для проверки конвейера")
    p.add_argument("--out", default="data/demo/out")
    add("passport", cmd_passport, "этап 2: паспорт данных, split.json, проверка утечек")
    p = add("check-data", cmd_check_data, "сверка чисел ЛР1 (255/276 заголовков, 14/16 классов) по всем определениям")
    p.add_argument("--dir", action="append", help="папка с observations*.csv (можно несколько, например out и out_hard)")
    p.add_argument("--llm-dir", action="append", default=[], help="папка логов LLM ЛР1 с preds/*.jsonl (можно несколько)")
    p.add_argument("--claim-headers", type=int, nargs="*", default=[255, 276])
    p.add_argument("--claim-classes", type=int, nargs="*", default=[14, 16])
    p = add("freeze", cmd_freeze, "этап 3: заморозить число шагов и целевой F1, защитить configs хешем")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--target", type=float, required=True)
    p = add("bench", cmd_bench, "замер секунд/шаг и оценка времени всех этапов")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--plan-steps", type=int, default=0)
    p = add("pilot", cmd_pilot, "этап 3: 8 оптимизаторов на одном seed")
    p.add_argument("--steps", type=int, default=0, help="переопределить число шагов только для пилота")
    add("tune", cmd_tune, "этап 4: подбор LR (8 x 7 x 2 запуска)")
    add("select-lr", cmd_select_lr, "выбрать LR по результатам tune")
    add("main", cmd_main, "этап 5: 8 оптимизаторов x 10 seed + сценарии S0-S5")
    add("sens", cmd_sens, "этап 7: sensitivity (LR, batch size, weight decay)")
    add("stats", cmd_stats, "этап 6: статистика, сводный балл, гипотезы, графики")
    add("status", cmd_status, "сколько запусков выполнено")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
