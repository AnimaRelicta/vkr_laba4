"""Сборка results/<backend>/stochastic/: таблицы, графики и report.md этапа 7. Без torch.

Каждый раздел строится, только если есть его входные данные; чего не хватает, отмечено в отчёте.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stochastic as X
from .optimizers import OPTIMIZERS

_COLORS = {o: c for o, c in zip(OPTIMIZERS, ["#6b7280", "#8c564b", "#e377c2", "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd"])}


def md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df is None or not len(df):
        return "_нет данных_\n"
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows) + "\n"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------ sensitivity
def _plot_sens(t: pd.DataFrame, out: Path) -> list[str]:
    plt = _plt()
    made = []
    for kind, fname, ylab in (("f1", "sens_curves.png", "test Macro-F1 (MAIN), среднее по seed"), ("rank", "sens_ranks.png", "ранг (1 = лучший)")):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=(kind == "rank"))
        for ax, f in zip(axes, X.FACTORS):
            g = t[t["factor"] == f]
            xs = sorted(g["x"].unique())
            for o, go in g.groupby("optimizer"):
                go = go.set_index("x").reindex(xs)
                pos = np.arange(len(xs))
                if kind == "f1":
                    ax.errorbar(pos, go["f1"], yerr=go["f1_sd"].fillna(0), marker="o", capsize=3, label=o, color=_COLORS.get(o))
                else:
                    ax.plot(pos, go["rank"], marker="o", lw=2, label=o, color=_COLORS.get(o))
            ax.set_xticks(np.arange(len(xs)))
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.set_xlabel(X.FACTORS[f][1])
            if kind == "rank":
                ax.invert_yaxis()
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(ylab)
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)
        made.append(fname)
    return made


def _plot_tune(tune: pd.DataFrame, plat: pd.DataFrame, out: Path, grid: dict) -> str:
    plt = _plt()
    fams = {"adam": "Adam-семейство", "sgd": "SGD-семейство"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (fam, title) in zip(axes, fams.items()):
        for o, g in tune.groupby("optimizer"):
            if grid["optimizers"][o]["family"] != fam:
                continue
            g = g.sort_values("lr")
            ax.plot(g["lr"], g["val_f1_mean"], marker="o", label=o, color=_COLORS.get(o))
            s = g[g["selected"]]
            ax.scatter(s["lr"], s["val_f1_mean"], s=140, facecolors="none", edgecolors="k", zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("LR")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("val Macro-F1 (tune, один seed)")
    fig.tight_layout()
    fig.savefig(out / "tune_curves.png", dpi=150)
    plt.close(fig)
    return "tune_curves.png"


def section_sens(cfg, runs: pd.DataFrame, out: Path) -> list[str]:
    from .runner import load_selected_lr

    L = ["## Sensitivity (sens): LR, batch size, weight decay\n"]
    sens = runs[runs["phase"] == "sens"]
    if sens.empty:
        L.append("Запусков фазы sens нет: выполните `uv run lab4 sens`.\n")
    else:
        bs0 = cfg.protocol["train"]["batch_size"]
        lrs = load_selected_lr(cfg)
        t = X.sens_tables(runs, lrs, cfg.grid, bs0)
        t.to_csv(out / "sens_table.csv", index=False, encoding="utf-8-sig")
        st = X.rank_stability(t)
        st.to_csv(out / "rank_stability.csv", index=False, encoding="utf-8-sig")
        n = sens.groupby("optimizer")["seed"].nunique()
        L += [f"Оптимизаторы: {', '.join(sorted(sens['optimizer'].unique()))}; seed на оптимизатор: {n.min()}-{n.max()}. "
              "Один фактор за раз, остальные на базовых значениях; расходящийся запуск считается за F1 = 0. "
              "При таком числе seed `f1_sd` - грубая оценка, различия порядка 0,01-0.02 в пределах шума.\n",
              "### Средний F1 и ранг по условиям\n",
              md(t[["factor", "optimizer", "x", "f1", "f1_sd", "val", "n_seeds", "unstable", "rank"]].round(4)),
              "### Устойчивость порядка оптимизаторов\n", md(st)]
        rep = X.sens_reproducibility(runs, lrs, bs0)
        if len(rep):
            rep.to_csv(out / "sens_vs_main.csv", index=False, encoding="utf-8-sig")
            mx = float(rep["abs_diff"].max())
            L += ["### Воспроизводимость: базовый sens-запуск против main (тот же оптимизатор, LR, batch, seed)\n",
                  md(rep.round(6)),
                  f"Максимальная разница F1: {mx:.3g}. " + ("Запуски воспроизводятся бит-в-бит.\n" if mx < 1e-9 else
                  "Ненулевая разница: один seed на GPU (bf16, недетерминированные ядра) даёт слегка разный результат; "
                  "это отдельный источник случайности, указать в карте.\n")]
        plat = X.plateau_table(t[t["factor"] == "lr"], "f1", True).assign(source="sens, test F1")
        plat.to_csv(out / "plateau_sens.csv", index=False, encoding="utf-8-sig")
        L += ["### Ширина плато по LR (sens, test F1)\n",
              "x - log10 множителя к выбранному LR (сетка множителей 0,1...10 даёт максимум 2 декады). "
              "`left/right_censored` = плато упирается в край проверенного диапазона, тогда ширина - нижняя оценка.\n",
              md(plat[plat["delta"] == 0.02].drop(columns=["delta", "source"]).round(3)),
              "Ширины при допусках 0,01 и 0,05: `plateau_sens.csv`.\n"]
        L += [f"![sens](stochastic/{f})\n" for f in _plot_sens(t, out)]
    # плато по сетке tune (все 8 оптимизаторов, val F1, один seed)
    tp = cfg.results_dir / "tuning" / "tuning_table.csv"
    if tp.exists():
        tune = pd.read_csv(tp, encoding="utf-8-sig")
        plat = X.plateau_table(tune.rename(columns={"lr": "x"}), "val_f1_mean", True)
        plat["best_lr"] = 10 ** plat["best_x"]
        plat.to_csv(out / "plateau_tune.csv", index=False, encoding="utf-8-sig")
        sub = plat[plat["delta"] == 0.02][["optimizer", "best_lr", "best_y", "width_decades", "n_points", "left_censored", "right_censored"]]
        L += ["### Ширина плато по сетке tune (8 оптимизаторов, val F1, один seed, допуск 0,02)\n",
              "Сетка шагает по половине декады, поэтому плато из одной точки даёт нулевую ширину; обрыв до F1 около 0,02 "
              "(расходимость) интерполируется линейно между соседними точками, реальная граница устойчивости лежит где-то в этом интервале.\n",
              md(sub.round(4)), f"![tune](stochastic/{_plot_tune(tune, plat, out, cfg.grid)})\n"]
    else:
        L.append("tuning_table.csv не найден: плато по сетке tune не построено.\n")
    return L


# ------------------------------------------------------------------ шум градиента
def section_noise(cfg, out: Path) -> list[str]:
    L = ["## Шум градиента на чекпоинте\n"]
    files = sorted((cfg.results_dir / "noise").glob("noise_*.json"))
    if not files:
        L.append("Файлов noise_*.json нет: выполните `uv run lab4 noise`.\n")
        return L
    rows, fits = [], []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        r = X.noise_rows(d["moments"])
        r.insert(0, "seed", d["seed"])
        r.insert(0, "optimizer", d["optimizer"])
        rows.append(r)
        fits.append({"optimizer": d["optimizer"], "seed": d["seed"], "lr": d["lr"], **X.noise_scale(d["moments"])})
    t, fit = pd.concat(rows, ignore_index=True), pd.DataFrame(fits)
    t.to_csv(out / "noise_by_batch.csv", index=False, encoding="utf-8-sig")
    fit.to_csv(out / "noise_scale.csv", index=False, encoding="utf-8-sig")
    ck = json.loads(files[0].read_text(encoding="utf-8"))
    L += [f"Чекпоинт шага {ck['checkpoint_step']}, {ck['n_batches']} независимых батчей на каждый размер, градиент до clipping, "
          "модель в train-режиме (dropout включён). `trace_cov_x_B` должно быть почти постоянным (шум ~ 1/B); "
          "`noise_to_signal` = tr(Cov) / |E g|^2 при данном batch size; `B_simple` = S/|G|^2 - размер батча, при котором шум и сигнал равны.\n",
          "### По batch size\n", md(t.round(5), "{:.4g}"),
          "### Масштаб шума (МНК по всем batch size и по паре крайних)\n", md(fit, "{:.4g}")]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (o, s), g in t.groupby(["optimizer", "seed"]):
        ax.plot(g["batch_size"], g["trace_cov"], marker="o", label=f"{o} s{s}", color=_COLORS.get(o))
    b = np.array(sorted(t["batch_size"].unique()), float)
    g0 = t.groupby("batch_size")["trace_cov"].mean()
    ax.plot(b, g0.iloc[0] * b[0] / b, "k--", lw=1, label="наклон -1")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size")
    ax.set_ylabel("tr Cov градиента")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "noise_scaling.png", dpi=150)
    plt.close(fig)
    L.append("![noise](stochastic/noise_scaling.png)\n")
    return L


# ------------------------------------------------------------------ bootstrap по документам и LOCO
def section_robust(cfg, runs: pd.DataFrame, out: Path, fin: list[str], B: int = 2000) -> list[str]:
    from .data import load_observations, split_frames
    from .mutations import scenario_test_sets

    L = ["## Bootstrap по документам и leave-one-cluster-out (test)\n"]
    if len(fin) < 2:
        return L + ["Меньше двух финалистов: анализ пропущен.\n"]
    main = runs[runs["phase"] == "main"]
    pdir = cfg.results_dir / "probs"
    if not pdir.exists():
        return L + ["Нет results/<backend>/probs/: выполните `uv run lab4 main`.\n"]
    data_dir = Path(cfg.protocol["data"]["dir"])
    from .config import ROOT
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    try:
        test = split_frames(load_observations(data_dir))["test"]
    except FileNotFoundError as e:
        return L + [f"Данные не найдены ({e}): передайте `--data <папка с observations.csv>`. Анализ пропущен.\n"]
    sets = {k: v for k, v in scenario_test_sets(cfg.scenarios, cfg.mutations, cfg.protocol["train"]["copies_test"], test).items()
            if k in X.SAVED_SCENARIOS}
    boot_rows, diff_rows, loco_rows, unit_rows, check = [], [], [], [], []
    for scen, df in sets.items():
        y = X.labels_of(df)
        cnt_doc, cnt_cl = {}, {}
        doc_codes, doc_names = pd.factorize(df["doc_ref"])
        cl_of_doc = df.drop_duplicates("doc_ref").set_index("doc_ref")["cluster_id"].reindex(doc_names).to_numpy()
        cl_codes, cl_names = pd.factorize(df["cluster_id"])
        for o in fin:
            rs = main[main["optimizer"] == o].sort_values("seed")
            per_doc, per_cl = [], []
            for r in rs.itertuples():
                f = pdir / f"{r.run_id}.npz"
                if not f.exists():
                    return L + [f"Нет {f.name}: анализ пропущен."]
                P = np.load(f)[scen]
                if len(P) != len(df):
                    return L + [f"{f.name}[{scen}]: {len(P)} строк, а пересобранный test даёт {len(df)}: данные или конфиг отличаются от прогона main."]
                pred = P.argmax(1)
                diff = X.check_against_runs(getattr(r, f"test_f1_{scen}"), y, pred)
                check.append(diff)
                if diff > 1e-3:
                    return L + [f"Пересчитанный F1 расходится с runs.csv на {diff:.4f} ({f.name}, {scen}): порядок строк или данные другие. Анализ остановлен."]
                per_doc.append(X.count_arrays(y, pred, doc_codes, len(doc_names)))
                per_cl.append(X.count_arrays(y, pred, cl_codes, len(cl_names)))
            cnt_doc[o], cnt_cl[o] = np.stack(per_doc), np.stack(per_cl)
        unit_rows.append({"scenario": scen, "rows": len(df), "docs": len(doc_names), "clusters": len(cl_names)})
        levels = {"docs": (cnt_doc, len(doc_names))}
        if len(cl_names) >= 3:
            levels["clusters"] = (cnt_cl, len(cl_names))
        for lvl, (cnt, nu) in levels.items():
            boot = X.bootstrap_units(cnt, nu, B)
            for o in fin:
                pe = X.point_estimate(cnt[o])
                seeds_f1 = [float(X.f1_from_weights(cnt[o][s], np.ones(nu))) for s in range(cnt[o].shape[0])]
                lo, hi = X.ci(boot[o]["units"])
                lo2, hi2 = X.ci(boot[o]["units_seeds"])
                boot_rows.append({"scenario": scen, "level": lvl, "optimizer": o, "f1": pe, "sd_seeds": float(np.std(seeds_f1, ddof=1)),
                                  "ci_lo": lo, "ci_hi": hi, "sd_boot": float(boot[o]["units"].std()),
                                  "ci_lo_with_seeds": lo2, "ci_hi_with_seeds": hi2})
            dd = X.paired_diffs(boot, "units_seeds")
            dd.insert(0, "level", lvl)
            dd.insert(0, "scenario", scen)
            diff_rows.append(dd)
        lc = X.loco(cnt_doc, cl_of_doc)
        if len(lc):
            lc.insert(0, "scenario", scen)
            loco_rows.append(lc)
    bt, units = pd.DataFrame(boot_rows), pd.DataFrame(unit_rows)
    bt.to_csv(out / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    L += [f"Финалисты (топ-{len(fin)} по сводному баллу среди прошедших фильтр): {', '.join(fin)}. "
          f"Перевыборка {B} раз; единица - документ (все колонки документа идут вместе) и, отдельно, кластер. "
          f"Пересчитанный F1 совпал с runs.csv (макс. расхождение {max(check):.1e}), то есть порядок строк test восстановлен верно. "
          "Модели не переобучаются: bootstrap показывает неопределённость оценки по выборке test при данных моделях, "
          "`ci_with_seeds` добавляет перевыборку seed обучения.\n",
          "### Размеры test по сценариям\n", md(units),
          "### F1 и доверительные интервалы\n", md(bt.round(4))]
    if diff_rows:
        d = pd.concat(diff_rows, ignore_index=True)
        d.to_csv(out / "bootstrap_diffs.csv", index=False, encoding="utf-8-sig")
        L += ["### Парные разности (документы + seed; p_boot - двусторонняя доля реплик по разные стороны нуля)\n", md(d[d["level"] == "docs"].round(4))]
    if loco_rows:
        lc = pd.concat(loco_rows, ignore_index=True)
        lc.to_csv(out / "loco.csv", index=False, encoding="utf-8-sig")
        L += ["### Leave-one-cluster-out (без переобучения: кластер убирается из оценки)\n",
              "`delta` = F1 без кластера минус F1 на всём test; `f1_only_cluster` - F1 только на нём. Меняется ли лидер при удалении кластера, видно по `rank_without`.\n",
              md(lc.round(4))]
        sub = lc[lc["scenario"] == "MAIN"]
        if len(sub):
            lead = {k: g.sort_values("f1_without").iloc[-1]["optimizer"] for k, g in sub.groupby("left_out_cluster")}
            L.append("Лидер (MAIN) при удалении кластера: " + ", ".join(f"{k}: {v}" for k, v in lead.items()) + ".\n")
    else:
        L.append("В test меньше двух кластеров: LOCO не определён.\n")
    _plot_boot(bt, out)
    L.append("![boot](stochastic/bootstrap_ci.png)\n")
    return L


def _plot_boot(bt: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    g = bt[(bt["scenario"] == "MAIN") & (bt["level"] == "docs")]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, r in enumerate(g.itertuples()):
        ax.errorbar(r.f1, i + 0.12, xerr=[[r.f1 - r.ci_lo], [r.ci_hi - r.f1]], fmt="o", color="C0", capsize=3, label="документы" if i == 0 else None)
        ax.errorbar(r.f1, i - 0.12, xerr=[[r.f1 - r.ci_lo_with_seeds], [r.ci_hi_with_seeds - r.f1]], fmt="s", color="C1", capsize=3,
                    label="документы + seed" if i == 0 else None)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g["optimizer"])
    ax.set_xlabel("test Macro-F1 (MAIN), 95 % bootstrap-ДИ")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "bootstrap_ci.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ карта источников случайности
def section_randomness(cfg, runs: pd.DataFrame, out: Path, fin: list[str]) -> list[str]:
    """Таблица источников случайности: где фиксирован, где меняется, чем оценён вклад. Числа берутся из файлов, если они есть."""
    mc, t = cfg.mutations["seeds"], cfg.protocol["train"]
    main = runs[runs["phase"] == "main"]
    sd_seed = float("nan")
    if len(main) and fin:
        sd_seed = float(np.mean([pd.to_numeric(main[main["optimizer"] == o]["test_f1_main"], errors="coerce").std(ddof=1) for o in fin]))

    def read(name):
        p = out / name
        return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

    boot, loco, plat, nscale = read("bootstrap_ci.csv"), read("loco.csv"), read("plateau_sens.csv"), read("noise_scale.csv")
    sd_doc = float(boot[(boot["scenario"] == "MAIN") & (boot["level"] == "docs") & boot["optimizer"].isin(fin)]["sd_boot"].mean()) \
        if boot is not None and len(boot) else float("nan")
    loco_rng = float(loco[loco["scenario"] == "MAIN"].groupby("optimizer")["delta"].apply(lambda s: s.max() - s.min()).mean()) \
        if loco is not None and len(loco) else float("nan")
    width = float(plat[plat["delta"] == 0.02]["width_decades"].mean()) if plat is not None and len(plat) else float("nan")
    bsimple = float(nscale["B_simple_fit"].median()) if nscale is not None and len(nscale) else float("nan")

    def f(v, fmt="{:.4f}"):
        return fmt.format(v) if np.isfinite(v) else "нет данных"

    rows = [
        ("Разбиение train/val/test", f"из ЛР1, не пересчитывается (split.json); {t['copies_train']} копий train", "нет (фиксировано)",
         f"LOCO: размах F1 при удалении кластера, среднее по финалистам = {f(loco_rng)}"),
        ("Выборка документов test", "фиксирована", "нет",
         f"bootstrap по документам: sd F1 (MAIN) = {f(sd_doc)}"),
        ("Мутации train", f"seed = {mc['train_base']} + seed запуска", "да, между seed",
         "входит в разброс по seed"),
        ("Мутации val/test", f"seed val = {mc['val']}, test = {mc['test']}", "нет, одинаковы для всех оптимизаторов",
         "сценарии S0-S5 разделяют F1 при одинаковых мутациях"),
        ("Инициализация LoRA и головы", "torch.manual_seed(seed) в build_model", "да, между seed", "разброс по seed"),
        ("Порядок и состав батчей", "np.random.default_rng(seed).permutation", "да, между seed", "разброс по seed; шум градиента (ниже)"),
        ("Dropout (LoRA 0,05, голова 0,2)", "torch RNG после seed", "да, между seed", "разброс по seed; входит в замер шума градиента"),
        ("Разброс по seed обучения", f"{len(t['eval_seeds'])} seed в main ({t['eval_seeds']})", "-",
         f"sd test F1 (MAIN), среднее по финалистам = {f(sd_seed)}"),
        ("Выбор LR", f"tune: seed {t['tuning_seeds']}, один запуск на точку сетки", "нет (одна реализация)",
         f"плато по LR: средняя ширина {f(width, '{:.2f}')} декады (sens, допуск 0,02)"),
        ("Шум градиента", f"шаг {cfg.grid['noise']['checkpoint_step']}, {cfg.grid['noise']['n_batches']} батчей, batch {cfg.grid['noise']['batch_sizes']}", "да (мини-батч)",
         f"B_simple (медиана по запускам) = {f(bsimple, '{:.0f}')}; рабочий batch = {t['batch_size']}"),
        ("S5: цепь Маркова версий шаблона", f"{cfg.scenarios['scenarios']['S5']['markov']['draws']} розыгрышей, seed test", "да, внутри запуска",
         "test_f1_S5_sd в runs.csv"),
        ("Bootstrap ДИ", "seed 0, 10000 ресэмплов (по seed обучения)", "нет", "воспроизводим"),
        ("Недетерминизм CUDA и bf16", "не контролируется", "да, слабо",
         "не оценивался отдельно; повторный запуск того же seed даёт близкие, но не обязательно побитово равные метрики"),
    ]
    df = pd.DataFrame(rows, columns=["Источник", "Как задан", "Меняется между запусками", "Чем оценён вклад"])
    return ["## Карта источников случайности\n", md(df)]


# ------------------------------------------------------------------ сборка
def build_stoch_report(cfg, runs: pd.DataFrame, only: list[str] | None = None, B: int = 2000) -> Path:
    out = cfg.results_dir / "stochastic"
    out.mkdir(parents=True, exist_ok=True)
    want = set(only or ["sens", "noise", "robust", "map"])
    main = runs[runs["phase"] == "main"]
    fin: list[str] = []
    if len(main):
        fin = X.finalists(runs, cfg.protocol["selection"], cfg.protocol["train"]["steps"])
    L = ["# Стохастический анализ (этап 7)\n",
         f"Бэкенд: {cfg.backend}. Финалисты: {', '.join(fin) if fin else 'нет данных main'}.\n"]
    if "sens" in want:
        L += section_sens(cfg, runs, out)
    if "noise" in want:
        L += section_noise(cfg, out)
    if "robust" in want:
        L += section_robust(cfg, runs, out, fin, B)
    if "map" in want:
        L += section_randomness(cfg, runs, out, fin)
    p = out / "report.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p
