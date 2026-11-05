"""Сборка results/stats/*: таблицы, проверка гипотез, графики. Читает только runs.csv и logs/."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import stats as S


def _plots(main: pd.DataFrame, logs_dir: Path, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    opts = sorted(main["optimizer"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([pd.to_numeric(main.loc[main.optimizer == o, "test_f1_main"], errors="coerce").dropna() for o in opts], tick_labels=opts)
    ax.set_ylabel("Macro-F1, test с мутациями")
    ax.set_title("Распределение итоговой метрики по seed")
    fig.tight_layout()
    fig.savefig(out / "f1_boxplot.png", dpi=150)
    plt.close(fig)
    made.append("f1_boxplot.png")
    curves: dict = {o: [] for o in opts}
    for rid, o in zip(main["run_id"], main["optimizer"]):
        p = logs_dir / "main" / f"{rid}.jsonl"
        if p.exists():
            pts = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]
            curves[o].append(pd.Series({x["step"]: x["val_f1"] for x in pts if "val_f1" in x}))
    if any(curves.values()):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for o, lst in curves.items():
            if lst:
                m = pd.concat(lst, axis=1).mean(axis=1)
                ax.plot(m.index, m.values, label=o)
        ax.set_xlabel("шаг")
        ax.set_ylabel("val Macro-F1 (среднее по seed)")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "convergence.png", dpi=150)
        plt.close(fig)
        made.append("convergence.png")
    return made


def build_report(cfg, runs: pd.DataFrame) -> Path:
    out = cfg.results_dir / "stats"
    out.mkdir(parents=True, exist_ok=True)
    main = runs[runs["phase"] == "main"].copy()
    if main.empty:
        raise SystemExit("В runs.csv нет запусков фазы main.")
    sel, tcfg = cfg.protocol["selection"], cfg.protocol["train"]
    summary = S.summarize(main, sel, tcfg["steps"])
    score = S.composite(summary, sel["weights"])
    choice = S.choose_winner(summary, score, sel["max_unstable_share"])
    table = choice["table"].sort_values("score", ascending=False)
    scen_names = [k for k in cfg.scenarios["scenarios"]]
    scen = S.scenario_table(main, scen_names)
    hyp = S.hypotheses(summary, scen, choice["winner"], cfg.protocol["hypotheses"], main)
    table.to_csv(out / "summary.csv", encoding="utf-8-sig")
    scen.to_csv(out / "scenarios.csv", encoding="utf-8-sig")
    lines = ["# Сравнение оптимизаторов (авто-отчёт из runs.csv)\n",
             f"Запусков main: {len(main)}, оптимизаторов: {main['optimizer'].nunique()}, seed на оптимизатор: "
             f"{main.groupby('optimizer')['seed'].nunique().min()}-{main.groupby('optimizer')['seed'].nunique().max()}.\n",
             "## Сводная таблица (test с мутациями, сценарий MAIN)\n", S.fmt_table(table), "\n",
             f"## Выбор\n\nФильтр: не более {sel['max_unstable_share']:.0%} нестабильных запусков.\n",
             f"Победитель по сводному баллу: **{choice['winner']}**. Вердикт: {choice['verdict']}.\n",
             f"Итоговый выбор оптимизатора: **{choice.get('final')}**.\n"]
    if choice["winner"]:
        w = S.pairwise_wilcoxon(main, choice["winner"])
        w.to_csv(out / "wilcoxon_holm.csv", encoding="utf-8-sig")
        lines += [f"Парный Wilcoxon {choice['winner']} против остальных (по seed, Holm):\n", S.fmt_table(w.rename_axis("optimizer")), "\n"]
    lines += ["## Сценарии S0-S5 (средний Macro-F1, ранг, падение к S0)\n", S.fmt_table(scen), "\n",
              "## Гипотезы\n"] + [f"- {h['h']}: {h['value']} -> {'подтверждена' if h['ok'] else 'не подтверждена' if h['ok'] is False else 'не проверяется (вне объёма задания)'}" for h in hyp]
    unstable = main[main["unstable"].astype(str).isin(["1", "True", "1.0"])]
    keep = [c for c in ("run_id", "optimizer", "seed", "lr", "unstable_reason", "steps_done") if c in unstable.columns]
    unstable[keep].to_csv(out / "unstable_main.csv", index=False, encoding="utf-8-sig")   # полный журнал всех фаз: results/<backend>/unstable_runs.csv
    figs = _plots(main, cfg.results_dir / "logs", out)
    lines += ["\n## Графики\n"] + [f"- stats/{f}" for f in figs]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "report.md"
