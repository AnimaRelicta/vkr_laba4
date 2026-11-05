"""Задание 4: «перенос лучшего метода». Сравнение выбранного оптимизатора с базовыми (SGD, Adam) в модели ЛР4 и сопоставление с ЛР1.

Без torch, только runs.csv. Каскад с LLM в объём задания не входит. Числа ЛР1 пользователь задаёт сам и они явно помечены
как несопоставимые напрямую (другая модель и другой тестовый набор).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

from . import stats as S

METRICS = [("test_f1_S0", "F1 S0 (исходные заголовки)"), ("test_f1_main", "F1 MAIN (мутации)"),
           ("test_ece_main", "ECE MAIN"), ("steps_to_target", "шагов до цели (усечено бюджетом)")]


def _col(main: pd.DataFrame, o: str, c: str, budget: int) -> pd.Series:
    g = main[main["optimizer"] == o].set_index(pd.to_numeric(main[main["optimizer"] == o]["seed"]))
    x = pd.to_numeric(g[c], errors="coerce")
    return x.fillna(budget) if c == "steps_to_target" else x


def paired(main: pd.DataFrame, a: str, b: str, col: str, budget: int, n_boot: int) -> dict:
    """Разность a - b по общим seed: среднее, bootstrap-ДИ по seed, точный парный Wilcoxon (при n=5 минимум p = 0,0625)."""
    d = (_col(main, a, col, budget) - _col(main, b, col, budget)).dropna()
    m, lo, hi = S.bootstrap_ci(d.to_numpy(), n_boot)
    p = 1.0 if len(d) == 0 or np.allclose(d, 0) else float(sst.wilcoxon(d, zero_method="wilcox", method="auto").pvalue)
    return {"n_seeds": int(len(d)), "diff": m, "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": p,
            "min_possible_p": float(2 / 2 ** len(d)) if len(d) else float("nan")}


def build_transfer(cfg, runs: pd.DataFrame, final: str | None = None, baselines=("sgd", "adam"), lab1: dict | None = None,
                   out: Path | None = None) -> Path:
    out = out or cfg.results_dir / "transfer"
    out.mkdir(parents=True, exist_ok=True)
    main = runs[runs["phase"] == "main"]
    if main.empty:
        raise SystemExit("Нет запусков main.")
    sel, budget = cfg.protocol["selection"], cfg.protocol["train"]["steps"]
    summary = S.summarize(main, sel, budget)
    score = S.composite(summary, sel["weights"])
    choice = S.choose_winner(summary, score, sel["max_unstable_share"])
    final = final or choice["final"]
    if final not in summary.index:
        raise SystemExit(f"Оптимизатор {final} не найден в main.")
    scen_cols = {"test_f1_S0": "F1 S0", "test_f1_S3": "F1 S3", "test_f1_S4": "F1 S4", "test_f1_main": "F1 MAIN", "test_f1_S5": "F1 S5"}
    rows = []
    for o in [final] + [b for b in baselines if b != final and b in summary.index]:
        r = {"optimizer": o, "роль": "выбранный" if o == final else "базовый"}
        for c, name in scen_cols.items():
            if c in main.columns:
                m, lo, hi = S.bootstrap_ci(pd.to_numeric(main[main.optimizer == o][c], errors="coerce").dropna(), sel["bootstrap_resamples"])
                r[name] = m
                if c in ("test_f1_S0", "test_f1_main"):
                    r[name + " ДИ"] = f"[{lo:.3f}; {hi:.3f}]"
        s = summary.loc[o]
        r.update({"ECE MAIN": s["ece_mean"], "val F1": s["val_f1_mean"], "шагов до цели": s["steps_to_target_mean"],
                  "достигла цели": f"{int(s['reached_target'])} из {int(s['n_runs'])}", "время, мин": s["train_time_min"],
                  "память, МБ": s["peak_mem_mb"], "нестабильных": s["unstable_share"]})
        rows.append(r)
    tbl = pd.DataFrame(rows).set_index("optimizer")
    tbl.to_csv(out / "transfer_table.csv", encoding="utf-8-sig")
    pr = []
    for b in baselines:
        if b == final or b not in summary.index:
            continue
        for col, name in METRICS:
            if col in main.columns:
                pr.append({"против": b, "метрика": name, **paired(main, final, b, col, budget, sel["bootstrap_resamples"])})
    pairs = pd.DataFrame(pr)
    pairs.to_csv(out / "transfer_pairs.csv", index=False, encoding="utf-8-sig")
    L = ["# Перенос лучшего метода (авто-отчёт из runs.csv)\n",
         f"Выбранный оптимизатор (правило раздела 6 PROTOCOL.md): **{final}**; вердикт выбора: {choice['verdict']}. Базовые: {', '.join(b for b in baselines if b != final)}.\n",
         "## Сводка\n", S.fmt_table(tbl), "\n",
         "## Парные разности (выбранный минус базовый, по общим seed)\n",
         "Для steps_to_target и ECE меньше значит лучше. p_wilcoxon при n seed не может быть меньше min_possible_p: при 5 seed это 0,0625, "
         "значимости 0,05 не достичь в принципе, поэтому вывод строится по размеру эффекта и ДИ, а формулировать надо «различие не подтверждено», "
         "а не «различия нет».\n", S.fmt_table(pairs.set_index(["против", "метрика"])) if len(pairs) else "нет базовых оптимизаторов\n", "\n"]
    L += ["## Сопоставление с ЛР1\n"]
    if lab1:
        f4 = float(summary.loc[final, "f1_mean"])
        L += [f"| Показатель | Значение | Тестовый набор | Модель |", "| --- | --- | --- | --- |",
              f"| ЛР1: {lab1.get('name', 'LLM')} | {lab1['f1']:.4f} | {lab1.get('test', 'out_hard/test')}"
              + (f", {lab1['n']} строк" if lab1.get("n") else "") + " | LLM, few-shot |",
              f"| ЛР4: {final}, F1 MAIN | {f4:.4f} | out/test с мутациями" + (f", {lab1['n4']} строк (исходных)" if lab1.get("n4") else "")
              + " | Qwen3-0.6B + LoRA |", "",
              "**Напрямую числа не сравнимы:** разные модели и разные тестовые наборы (out_hard/test в ЛР1 против out/test в ЛР4), "
              "в ЛР4 добавлены искусственные мутации заголовков. Сопоставление показывает порядок величины и положение метода относительно "
              "результата ЛР1, но не доказывает превосходства одного подхода.\n"]
    else:
        L.append("Числа ЛР1 не заданы (`--lab1-f1 0.9640 --lab1-n 255 --lab4-n 216`).\n")
    (out / "transfer.md").write_text("\n".join(L), encoding="utf-8")
    return out / "transfer.md"
