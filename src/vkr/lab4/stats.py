"""Статистика, сводный балл, Pareto и проверка гипотез (ТЗ, п. 3.6). Без torch: работает по runs.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sst

DIRECTION = {"f1": +1, "std": -1, "ece": -1, "steps_to_target": -1, "memory": -1}


def bootstrap_ci(x, n: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Среднее и перцентильный bootstrap-ДИ по seed."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals: dict) -> dict:
    """Поправка Холма-Бонферрони."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run_max = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run_max = max(run_max, min(1.0, (m - i) * p))
        out[k] = run_max
    return out


def _censored_steps(df: pd.DataFrame, budget: int) -> pd.Series:
    s = pd.to_numeric(df["steps_to_target"], errors="coerce")
    return s.fillna(budget)


def summarize(main: pd.DataFrame, sel: dict, budget_steps: int, metric: str = "test_f1_main", ece_col: str = "test_ece_main") -> pd.DataFrame:
    rows = []
    for opt, g in main.groupby("optimizer"):
        f1 = pd.to_numeric(g[metric], errors="coerce")
        m, lo, hi = bootstrap_ci(f1.dropna(), sel["bootstrap_resamples"])
        rows.append({
            "optimizer": opt, "n_runs": len(g), "f1_mean": m, "f1_std": float(f1.std(ddof=1)) if f1.notna().sum() > 1 else float("nan"),
            "f1_ci_lo": lo, "f1_ci_hi": hi,
            "val_f1_mean": pd.to_numeric(g["val_f1_final"], errors="coerce").mean(),
            "ece_mean": pd.to_numeric(g[ece_col], errors="coerce").mean(),
            "steps_to_target_mean": _censored_steps(g, budget_steps).mean(),
            "reached_target": int(pd.to_numeric(g["steps_to_target"], errors="coerce").notna().sum()),
            "train_time_min": pd.to_numeric(g["train_time_s"], errors="coerce").mean() / 60,
            "peak_mem_mb": pd.to_numeric(g["peak_mem_mb"], errors="coerce").mean(),
            "unstable_share": pd.to_numeric(g["unstable"], errors="coerce").mean(),
        })
    return pd.DataFrame(rows).set_index("optimizer").sort_index()


def _minmax(x: pd.Series, direction: int) -> pd.Series:
    lo, hi = x.min(), x.max()
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return pd.Series(1.0, index=x.index)      # метрики равны у всех: никого не штрафуем
    z = (x - lo) / (hi - lo)
    return z if direction > 0 else 1 - z


def composite(summary: pd.DataFrame, weights: dict) -> pd.Series:
    """40 % F1, 20 % разброс, 15 % ECE, 15 % шаги до цели, 10 % память; нормировка min-max между оптимизаторами."""
    cols = {"f1": "f1_mean", "std": "f1_std", "ece": "ece_mean", "steps_to_target": "steps_to_target_mean", "memory": "peak_mem_mb"}
    score = sum(weights[k] * _minmax(summary[c].fillna(summary[c].max()), DIRECTION[k]) for k, c in cols.items())
    return score.rename("score")


def pareto_nondominated(summary: pd.DataFrame) -> pd.Series:
    """Недоминируемость по F1 (больше), времени (меньше) и разбросу (меньше)."""
    pts = pd.DataFrame({"f1": -summary["f1_mean"], "t": summary["train_time_min"], "sd": summary["f1_std"].fillna(0)})
    ok = {}
    for a in pts.index:
        dom = ((pts <= pts.loc[a]).all(axis=1) & (pts < pts.loc[a]).any(axis=1)).drop(a).any()
        ok[a] = not dom
    return pd.Series(ok, name="pareto_ok")


def pairwise_wilcoxon(main: pd.DataFrame, winner: str, metric: str = "test_f1_main") -> pd.DataFrame:
    """Парный Wilcoxon по seed: победитель против каждого, поправка Холма."""
    piv = main.pivot_table(index="seed", columns="optimizer", values=metric, aggfunc="mean")
    raw = {}
    for o in piv.columns:
        if o == winner:
            continue
        d = (piv[winner] - piv[o]).dropna()
        raw[o] = 1.0 if len(d) == 0 or np.allclose(d, 0) else float(sst.wilcoxon(d, zero_method="wilcox", method="auto").pvalue)
    adj = holm(raw)
    return pd.DataFrame({"p_raw": raw, "p_holm": adj, "mean_diff": {o: float((piv[winner] - piv[o]).mean()) for o in raw}})


def choose_winner(summary: pd.DataFrame, score: pd.Series, max_unstable: float) -> dict:
    res = summary.join(score).join(pareto_nondominated(summary))
    elig = res[res["unstable_share"] <= max_unstable + 1e-12]
    if elig.empty:
        return {"winner": None, "verdict": "ни один оптимизатор не прошёл фильтр нестабильности", "table": res}
    ranked = elig.sort_values("score", ascending=False)
    nd = ranked[ranked["pareto_ok"]]
    winner = (nd.index[0] if len(nd) else ranked.index[0])
    note = "" if ranked.index[0] == winner else f" ({ranked.index[0]} лидирует по баллу, но доминируется по Pareto)"
    second = ranked.index[1] if len(ranked) > 1 and ranked.index[1] != winner else (ranked.index[2] if len(ranked) > 2 else None)
    verdict, final = "победитель по сводному баллу" + note, winner
    if second is not None:
        a, b = res.loc[winner], res.loc[second]
        if a["f1_ci_lo"] <= b["f1_ci_hi"] and b["f1_ci_lo"] <= a["f1_ci_hi"]:
            verdict = f"{winner} и {second} статистически неразличимы (ДИ пересекаются): выбор по стабильности и ресурсам" + note
            final = min([winner, second], key=lambda o: (res.loc[o, "unstable_share"], res.loc[o, "peak_mem_mb"], res.loc[o, "train_time_min"]))
    return {"winner": winner, "second": second, "final": final, "verdict": verdict, "table": res}


def scenario_table(main: pd.DataFrame, scenarios: list[str]) -> pd.DataFrame:
    """Средний test Macro-F1 по сценариям, ранг, падение относительно S0."""
    cols = {s: f"test_f1_{s}" for s in scenarios if f"test_f1_{s}" in main.columns}
    mean = pd.DataFrame({s: main.groupby("optimizer")[c].apply(lambda x: pd.to_numeric(x, errors="coerce").mean()) for s, c in cols.items()})
    out = mean.copy()
    out.columns = [f"F1_{c}" for c in mean.columns]
    for s in mean.columns:
        out[f"rank_{s}"] = mean[s].rank(ascending=False, method="min")
        if "S0" in mean.columns and s != "S0":
            out[f"drop_{s}"] = mean["S0"] - mean[s]
    return out


def hypotheses(summary: pd.DataFrame, scen: pd.DataFrame, winner: str | None, hyp: dict, main: pd.DataFrame) -> list[dict]:
    out = []
    # H1: Adam и AdamW достигают цели минимум на 30 % быстрее SGD по шагам
    if "sgd" in summary.index:
        base = summary.loc["sgd", "steps_to_target_mean"]
        for o in ("adam", "adamw"):
            if o in summary.index:
                gain = 1 - summary.loc[o, "steps_to_target_mean"] / base
                out.append({"h": f"H1 ({o} vs sgd)", "value": f"выигрыш по шагам {gain:.1%}", "ok": bool(gain >= hyp["H1_min_speedup"])})
    # H2: разрыв F1 лучший-худший >= 0.02 и ДИ лучшего и SGD не пересекаются
    best, worst = summary["f1_mean"].idxmax(), summary["f1_mean"].idxmin()
    gap = summary.loc[best, "f1_mean"] - summary.loc[worst, "f1_mean"]
    ci_ok = None
    if "sgd" in summary.index and best != "sgd":
        ci_ok = bool(summary.loc[best, "f1_ci_lo"] > summary.loc["sgd", "f1_ci_hi"])
    out.append({"h": "H2", "value": f"разрыв {gap:.4f} ({best} - {worst}); ДИ лучшего и SGD не пересекаются: {ci_ok}",
                "ok": bool(gap >= hyp["H2_min_gap"] and (ci_ok is None or ci_ok))})
    # H3: победитель в топ-3 на S3-S5
    if winner is not None:
        ranks = {s: int(scen.loc[winner, f"rank_{s}"]) for s in ("S3", "S4", "S5") if f"rank_{s}" in scen.columns}
        out.append({"h": "H3", "value": f"ранги {winner}: {ranks}", "ok": bool(ranks) and all(r <= 3 for r in ranks.values())})
    out.append({"h": "H4", "value": "вне объёма задания ЛР4 (каскад с LLM не выполняется); перенос: `lab4 transfer`", "ok": None})
    return out


def fmt_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    d = df.reset_index()
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v) for v in r) + " |"
            for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)
