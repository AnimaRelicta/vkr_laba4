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
    """Замер скорости: прогрев, затем один запуск на каждый micro_batch (с оценками), оценка часов по реальным планам запусков."""
    import time

    import numpy as np
    import torch

    from .budget import plan_cost
    from .spec import RunSpec
    from .train import train_run

    cfg = _cfg(args)
    ctx = _context(cfg)
    t = cfg.protocol["train"]
    plan_steps, n = args.plan_steps or t["steps"], args.steps
    evals = max(1, plan_steps // t["eval_every"])          # столько же оценок на val, сколько в настоящем запуске
    t.update({"steps": n, "eval_every": max(1, n // evals), "target_val_f1": 2.0})
    be = ctx.backend
    if be.name == "qwen":
        tr = ctx.splits["train"]
        L = np.array([len(be.encode_one(h, v)) for h, v in zip(tr["header_text"], tr["values_sample"])])
        print(f"Длина входа, токенов: среднее {L.mean():.1f}, p95 {np.percentile(L, 95):.0f}, макс {L.max()}; "
              f"обрезано по max_len={be.q['max_len']}: {100 * (L >= be.q['max_len']).mean():.1f} %")
    micros = sorted(args.micro or ([8, 16, 24] if be.name == "qwen" else [be.micro_batch]), key=lambda x: (x is None, x))
    vram = torch.cuda.get_device_properties(0).total_memory if ctx.device == "cuda" else 0
    train_run(RunSpec("bench", "adamw", 1e-3, 0, steps=3), ctx, None, eval_scenarios=False)     # прогрев CUDA
    res = []
    for m in micros:
        be.micro_batch = m
        t0 = time.perf_counter()
        try:
            row = train_run(RunSpec("bench", "adamw", 1e-3, 0, steps=n), ctx, None, eval_scenarios=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            import torch
            torch.cuda.empty_cache()
            print(f"micro_batch={m}: не хватило памяти")
            continue
        dt = time.perf_counter() - t0
        per, over = row["train_time_s"] / max(1, row["steps_done"]), dt - row["train_time_s"]
        reserved = torch.cuda.max_memory_reserved() if vram else 0
        too_big = bool(vram) and reserved > 0.85 * vram      # Windows (WDDM) при нехватке уводит память в общую (RAM) и всё замедляется
        print(f"micro_batch={m}: {per:.3f} с/шаг, пик памяти {row['peak_mem_mb']:.0f} МБ"
              + (f", зарезервировано {reserved / 2**20:.0f} из {vram / 2**20:.0f} МБ" if vram else "")
              + f", обучаемых параметров {row['n_trainable']:,}, накладные расходы на запуск {over:.1f} с"
              + ("  <- СЛИШКОМ БЛИЗКО К ЛИМИТУ ПАМЯТИ, не использовать" if too_big else ""))
        if too_big:
            break                                            # большие micro_batch заведомо хуже
        res.append((per, over, m))
    if not res:
        raise SystemExit("Ни один вариант не отработал")
    per, over, m = min(res)
    be.micro_batch = m
    t["steps"] = plan_steps
    print(f"\nЛучший micro_batch: {m}. Впишите его в configs/protocol.yaml (model.qwen.micro_batch).")
    t0 = time.perf_counter()
    row = train_run(RunSpec("bench", "adamw", 1e-3, 0, steps=n), ctx, None, eval_scenarios=False)
    over_tune = max(0.0, time.perf_counter() - t0 - row["train_time_s"])
    print(f"накладные расходы на запуск: с оценкой сценариев {over:.1f} с (main, sens), без неё {over_tune:.1f} с (pilot, tune)")
    cost = plan_cost(cfg, per, {"pilot": over_tune, "tune": over_tune, "main": over, "sens": over})
    total = 0.0
    for phase, (runs, units, sec) in cost.items():
        print(f"  {phase:6s}: {runs:3d} запусков, {units:7.0f} шагов (в единицах batch {t['batch_size']}) ≈ {sec / 3600:.2f} ч")
        total += sec if phase != "pilot" else 0
    print(f"  Итого tune + main + sens ≈ {total / 3600:.2f} ч (пилот отдельно: {cost['pilot'][2] / 60:.0f} мин)")


def cmd_smoke(args):
    """Сквозной тестовый прогон всего конвейера на уменьшенном протоколе: tune -> select-lr -> main -> sens -> stats."""
    import shutil

    from .budget import smoke_config
    from .report import build_report
    from .runner import execute, main_specs, read_runs, select_lr, sens_specs, top_for_sensitivity, tune_specs

    cfg = smoke_config(_cfg(args), args.steps)
    shutil.rmtree(cfg.results_dir, ignore_errors=True)
    ctx = _context(cfg)
    print("== tune")
    execute(tune_specs(cfg), ctx, eval_scenarios=False)
    print("выбранные LR:", select_lr(cfg))
    print("== main")
    execute(main_specs(cfg), ctx, eval_scenarios=True, save_probs=True)
    print("== sens")
    opts = top_for_sensitivity(cfg)
    print("sensitivity для:", opts)
    execute(sens_specs(cfg, opts), ctx, eval_scenarios=True)
    print("== noise")
    from .noise import run_noise
    run_noise(cfg, ctx, ["adam", "sgd"], [0])
    print("== stats")
    print("Отчёт:", build_report(cfg, read_runs(cfg)))
    print("== stoch")
    from .stoch_report import build_stoch_report
    print("Отчёт:", build_stoch_report(cfg, read_runs(cfg), None, 200))
    print(f"SMOKE OK. Результаты проверки лежат в {cfg.results_dir} (в основные results/ не попадают, можно удалить).")


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


def cmd_noise(args):
    from .noise import run_noise
    cfg = _cfg(args)
    check_lock()
    run_noise(cfg, _context(cfg), args.optimizers, args.seeds, args.n_batches)


def cmd_stoch(args):
    from .runner import read_runs
    from .stoch_report import build_stoch_report
    cfg = _cfg(args)
    print("Отчёт:", build_stoch_report(cfg, read_runs(cfg), args.only, args.bootstrap))


def cmd_stats(args):
    from .report import build_report
    from .runner import read_runs
    cfg = _cfg(args)
    print("Отчёт:", build_report(cfg, read_runs(cfg)))


def cmd_transfer(args):
    from .runner import read_runs
    from .transfer import build_transfer
    cfg = _cfg(args)
    lab1 = {"f1": args.lab1_f1, "n": args.lab1_n, "n4": args.lab4_n, "name": args.lab1_name} if args.lab1_f1 else None
    print("Отчёт:", build_transfer(cfg, read_runs(cfg), args.final, tuple(args.baselines), lab1))


def cmd_check_report(args):
    from .check_report import run_check
    cfg = _cfg(args)
    out, res = run_check(cfg, Path(args.file), args.allow)
    print(f"Отчёт: {out}. Подтверждено: {len(res['ok'])}, без источника: {len(res['bad'])} десятичных, {len(res['ints'])} целых")
    for i in res["bad"][:20]:
        print(f"  ? {i['число']}: …{i['контекст']}…")
    if res["bad"]:
        sys.exit(1)


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
    p = add("bench", cmd_bench, "замер секунд/шаг, выбор micro_batch и оценка часов всех этапов")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--plan-steps", type=int, default=0)
    p.add_argument("--micro", type=int, nargs="*", help="micro_batch для сравнения, например: --micro 16 32 64")
    p = add("smoke", cmd_smoke, "тестовый прогон всего конвейера на уменьшенном протоколе (минуты)")
    p.add_argument("--steps", type=int, default=8)
    p = add("pilot", cmd_pilot, "этап 3: 8 оптимизаторов на одном seed")
    p.add_argument("--steps", type=int, default=0, help="переопределить число шагов только для пилота")
    add("tune", cmd_tune, "этап 4: подбор LR (8 x 7 x 2 запуска)")
    add("select-lr", cmd_select_lr, "выбрать LR по результатам tune")
    add("main", cmd_main, "этап 5: 8 оптимизаторов x 10 seed + сценарии S0-S5")
    add("sens", cmd_sens, "этап 7: запуски sensitivity (LR, batch size, weight decay)")
    p = add("noise", cmd_noise, "этап 7: шум градиента на чекпоинте (нужен torch и GPU)")
    p.add_argument("--optimizers", nargs="*", help="по умолчанию: финалисты (топ-3 по баллу) + sgd")
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--n-batches", type=int, default=0, help="только для проверки: переопределить grid.yaml (отступление нужно отметить в отчёте)")
    p = add("stoch", cmd_stoch, "этап 7: sensitivity-таблицы, плато, bootstrap по документам, LOCO, шум, карта случайности (без torch)")
    p.add_argument("--only", nargs="*", choices=["sens", "noise", "robust", "map"], help="только эти разделы")
    p.add_argument("--bootstrap", type=int, default=2000, help="число перевыборок")
    add("stats", cmd_stats, "этап 6: статистика, сводный балл, гипотезы, графики")
    p = add("transfer", cmd_transfer, "задание 4: выбранный оптимизатор против SGD/Adam и сопоставление с ЛР1 (из runs.csv)")
    p.add_argument("--final", help="по умолчанию итоговый выбор из stats")
    p.add_argument("--baselines", nargs="*", default=["sgd", "adam"])
    p.add_argument("--lab1-f1", type=float, help="Macro-F1 LLM из ЛР1, например 0.9640")
    p.add_argument("--lab1-n", type=int, help="строк в тесте ЛР1 (out_hard/test), например 255")
    p.add_argument("--lab4-n", type=int, help="строк в исходном тесте ЛР4 (out/test), например 216")
    p.add_argument("--lab1-name", default="qwen/qwen3.8-flash, few-shot k=3")
    p = add("check-report", cmd_check_report, "проверка: каждое число отчёта (.md/.txt/.docx) есть в results/<backend>/")
    p.add_argument("file")
    p.add_argument("--allow", type=float, nargs="*", default=[], help="внешние числа, которых нет в results (например 0.9640 255 216)")
    add("status", cmd_status, "сколько запусков выполнено")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
