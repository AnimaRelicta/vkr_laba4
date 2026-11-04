"""Этап 7: стохастический анализ без torch (работает по runs.csv, tuning_table.csv, probs/*.npz и json шума градиента).

Что здесь считается:
  * sensitivity: таблицы по LR / batch size / weight decay, ранги, ширина «плато» по LR;
  * bootstrap по документам (и по кластерам) для финалистов, парные разности;
  * leave-one-cluster-out на test (без переобучения: убираем кластер из оценки, модели те же);
  * оценка масштаба шума градиента по моментам, собранным в noise.py.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sst

from . import stats as S
from .labels import LABEL2ID, N_CLASSES
from .metrics import macro_f1

DELTAS = (0.01, 0.02, 0.05)             # допуски «плато»: падение F1 от максимума
SAVED_SCENARIOS = ("S0", "S3", "S4", "MAIN")   # сценарии, для которых train_run сохраняет вероятности
FACTORS = {"lr": ("lr_mult", "LR, множитель к выбранному"), "bs": ("batch_size", "batch size"), "wd": ("wd", "weight decay")}


# ------------------------------------------------------------------ финалисты
def finalists(runs: pd.DataFrame, sel: dict, steps: int, k: int = 3) -> list[str]:
    """Топ-k по сводному баллу среди оптимизаторов, прошедших фильтр нестабильности (то же правило, что в stats)."""
    main = runs[runs["phase"] == "main"]
    summ = S.summarize(main, sel, steps)
    score = S.composite(summ, sel["weights"])
    ok = summ.index[summ["unstable_share"] <= sel["max_unstable_share"] + 1e-12]
    return list(score.loc[ok].sort_values(ascending=False).index[:k])


# ------------------------------------------------------------------ sensitivity
def sens_tidy(runs: pd.DataFrame, lrs: dict, grid: dict) -> pd.DataFrame:
    """Запуски фазы sens с эффективными значениями факторов. Пустой weight_decay в runs.csv = значение по умолчанию из grid.yaml."""
    d = runs[runs["phase"] == "sens"].copy()
    for c in ("lr", "batch_size", "weight_decay", "val_f1_final", "test_f1_main", "unstable", "seed"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["lr_mult"] = (d["lr"] / d["optimizer"].map(lrs)).round(3)
    base_wd = d["optimizer"].map(lambda o: float(grid["optimizers"][o].get("weight_decay", 0.0)))
    d["wd_is_base"] = d["weight_decay"].isna()
    d["wd"] = d["weight_decay"].fillna(base_wd)
    d["f1"] = d["test_f1_main"].fillna(0.0)       # расходящийся запуск = 0, как в select_lr
    d["val"] = d["val_f1_final"].fillna(0.0)
    return d


def factor_table(d: pd.DataFrame, factor: str, bs0: int) -> pd.DataFrame:
    """Один фактор за раз: остальные на базовых значениях. Базовая точка входит в таблицу каждого фактора."""
    x = FACTORS[factor][0]
    if factor == "lr":
        m = (d["batch_size"] == bs0) & d["wd_is_base"]
    elif factor == "bs":
        m = (d["lr_mult"] == 1.0) & d["wd_is_base"]
    else:
        m = (d["lr_mult"] == 1.0) & (d["batch_size"] == bs0)
    g = (d[m].groupby(["optimizer", x])
         .agg(f1=("f1", "mean"), f1_sd=("f1", "std"), val=("val", "mean"), n_seeds=("seed", "nunique"), unstable=("unstable", "mean"))
         .reset_index().rename(columns={x: "x"}))
    g.insert(0, "factor", factor)
    return g


def sens_reproducibility(runs: pd.DataFrame, lrs: dict, bs0: int) -> pd.DataFrame:
    """Базовый sens-запуск (LR x1, batch bs0, weight decay по умолчанию) совпадает по конфигурации с main при том же seed:
    F1 должны совпасть. Ненулевая разница - недетерминизм железа."""
    s = runs[runs["phase"] == "sens"].copy()
    m = runs[runs["phase"] == "main"].copy()
    for c in ("lr", "batch_size", "weight_decay", "test_f1_main", "seed"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    m["seed"], m["f1_main"] = pd.to_numeric(m["seed"], errors="coerce"), pd.to_numeric(m["test_f1_main"], errors="coerce")
    s = s[(s["lr"] / s["optimizer"].map(lrs)).round(6).eq(1.0) & (s["batch_size"] == bs0) & s["weight_decay"].isna()]
    j = s.merge(m[["optimizer", "seed", "f1_main"]], on=["optimizer", "seed"])
    j["f1_sens"] = j["test_f1_main"]
    j["abs_diff"] = (j["f1_sens"] - j["f1_main"]).abs()
    return j[["optimizer", "seed", "f1_main", "f1_sens", "abs_diff"]].sort_values(["optimizer", "seed"]).reset_index(drop=True)


def sens_tables(runs: pd.DataFrame, lrs: dict, grid: dict, bs0: int) -> pd.DataFrame:
    d = sens_tidy(runs, lrs, grid)
    t = pd.concat([factor_table(d, f, bs0) for f in FACTORS], ignore_index=True)
    t["rank"] = t.groupby(["factor", "x"])["f1"].rank(ascending=False, method="average")
    return t


def rank_stability(t: pd.DataFrame) -> pd.DataFrame:
    """Насколько порядок оптимизаторов меняется между условиями: число разных победителей и средний Spearman между условиями."""
    rows = []
    for f, g in t.groupby("factor"):
        pf = g.pivot(index="x", columns="optimizer", values="f1")
        if pf.shape[1] < 2 or pf.shape[0] < 2:
            continue
        winners = pf.idxmax(axis=1)
        rho = []
        xs = list(pf.index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    r = float(sst.spearmanr(pf.loc[xs[i]], pf.loc[xs[j]])[0])
                    if np.isfinite(r):
                        rho.append(r)
        rows.append({"factor": f, "n_conditions": len(xs), "n_distinct_winners": int(winners.nunique()),
                     "winner_by_condition": "; ".join(f"{x:g}: {w}" for x, w in winners.items()),
                     "mean_pairwise_spearman": float(np.mean(rho)) if rho else float("nan")})
    return pd.DataFrame(rows)


def plateau(x, y, delta: float) -> dict:
    """Ширина плато: непрерывный участок вокруг максимума, где y >= max - delta; x - log10 LR (или множителя).

    Границы уточняются линейной интерполяцией между соседними точками. Если плато упирается в край сетки,
    ширина - нижняя оценка (left_censored / right_censored)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], np.nan_to_num(y[order], nan=0.0)
    i = int(np.argmax(y))
    thr = y[i] - delta
    lo = hi = i
    while lo > 0 and y[lo - 1] >= thr:
        lo -= 1
    while hi < len(x) - 1 and y[hi + 1] >= thr:
        hi += 1

    def cross(a, b):    # a - точка ниже порога, b - выше
        return x[a] + (thr - y[a]) * (x[b] - x[a]) / (y[b] - y[a])

    xl = x[lo] if lo == 0 else cross(lo - 1, lo)
    xr = x[hi] if hi == len(x) - 1 else cross(hi + 1, hi)
    return {"best_x": float(x[i]), "best_y": float(y[i]), "x_lo": float(xl), "x_hi": float(xr), "width_decades": float(xr - xl),
            "n_points": hi - lo + 1, "left_censored": lo == 0, "right_censored": hi == len(x) - 1}


def plateau_table(t_lr: pd.DataFrame, value: str, log10_x: bool, deltas=DELTAS) -> pd.DataFrame:
    """t_lr: колонки optimizer, x (LR или множитель), value. Плато считается отдельно для каждого оптимизатора и допуска."""
    rows = []
    for o, g in t_lr.groupby("optimizer"):
        xs = np.log10(g["x"].to_numpy(float)) if log10_x else g["x"].to_numpy(float)
        for dl in deltas:
            rows.append({"optimizer": o, "delta": dl, **plateau(xs, g[value].to_numpy(float), dl)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ bootstrap по документам / кластерам и LOCO
def count_arrays(y, pred, unit_idx, n_units: int, n_classes: int = N_CLASSES) -> np.ndarray:
    """Ошибки по единицам (документ или кластер): массив (3, U, C) = tp, fp, fn."""
    y, pred, unit_idx = np.asarray(y), np.asarray(pred), np.asarray(unit_idx)
    cnt = np.zeros((3, n_units, n_classes))
    hit = y == pred
    np.add.at(cnt[0], (unit_idx[hit], y[hit]), 1)
    np.add.at(cnt[1], (unit_idx[~hit], pred[~hit]), 1)
    np.add.at(cnt[2], (unit_idx[~hit], y[~hit]), 1)
    return cnt


def f1_from_weights(cnt: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Macro-F1 по классам, встречающимся в gold (как metrics.macro_f1), для набора весов единиц.

    cnt: (3, U, C); w: (U,) или (B, U). Вес = сколько раз единица попала в выборку."""
    tp, fp, fn = (w @ cnt[0]), (w @ cnt[1]), (w @ cnt[2])
    den = 2 * tp + fp + fn
    f = np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)
    present = (tp + fn) > 0
    return (f * present).sum(-1) / np.maximum(present.sum(-1), 1)


def bootstrap_units(cnt: dict, n_units: int, B: int = 2000, seed: int = 0) -> dict:
    """Одна общая перевыборка единиц на всех оптимизаторов (так разности парные).

    cnt[opt]: (S, 3, U, C) по seed обучения. Возвращает opt -> {"units": (B,), "units_seeds": (B,)}:
    среднее по seed при фиксированных моделях и то же с перевыборкой seed внутри каждой реплики."""
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=B).astype(float)
    n_seeds = {c.shape[0] for c in cnt.values()}
    if len(n_seeds) != 1:
        raise ValueError(f"Разное число seed у оптимизаторов: {sorted(n_seeds)}")
    ns = n_seeds.pop()
    pick = rng.integers(0, ns, size=(B, ns))
    out = {}
    for o, c in cnt.items():
        F = np.stack([f1_from_weights(c[s], W) for s in range(ns)], axis=1)       # (B, S)
        out[o] = {"units": F.mean(1), "units_seeds": np.take_along_axis(F, pick, 1).mean(1)}
    return out


def point_estimate(c: np.ndarray) -> float:
    """Среднее по seed F1 на всех единицах (без перевыборки)."""
    return float(np.mean([f1_from_weights(c[s], np.ones(c.shape[2])) for s in range(c.shape[0])]))


def ci(x, alpha: float = 0.05) -> tuple[float, float]:
    return float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))


def paired_diffs(boot: dict, key: str = "units") -> pd.DataFrame:
    """Разность средних F1 для каждой пары оптимизаторов по общим перевыборкам; p - двусторонняя доля реплик по разные стороны нуля."""
    rows, opts = [], sorted(boot)
    for i, a in enumerate(opts):
        for b in opts[i + 1:]:
            d = boot[a][key] - boot[b][key]
            lo, hi = ci(d)
            rows.append({"a": a, "b": b, "diff_mean": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                         "p_boot": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
    return pd.DataFrame(rows)


def loco(cnt: dict, unit_cluster: np.ndarray) -> pd.DataFrame:
    """Leave-one-cluster-out на test: F1 на документах без кластера k (модели не переобучаются)."""
    clusters = sorted(set(unit_cluster))
    rows = []
    for o, c in cnt.items():
        full = point_estimate(c)
        for k in clusters:
            w = (unit_cluster != k).astype(float)
            if w.sum() == 0:
                continue
            rest = float(np.mean([f1_from_weights(c[s], w) for s in range(c.shape[0])]))
            only = float(np.mean([f1_from_weights(c[s], 1.0 - w) for s in range(c.shape[0])]))
            rows.append({"optimizer": o, "left_out_cluster": k, "n_docs_left_out": int((unit_cluster == k).sum()),
                         "f1_full": full, "f1_without": rest, "delta": rest - full, "f1_only_cluster": only})
    t = pd.DataFrame(rows)
    if len(t):
        t["rank_without"] = t.groupby("left_out_cluster")["f1_without"].rank(ascending=False, method="min")
    return t


def check_against_runs(runs_f1: float, y, pred) -> float:
    """Расхождение пересчитанного F1 с записанным в runs.csv: ловит неверный порядок строк или другие данные."""
    return abs(macro_f1(y, pred) - float(runs_f1))


def labels_of(df: pd.DataFrame) -> np.ndarray:
    return df["label"].map(LABEL2ID).to_numpy()


# ------------------------------------------------------------------ шум градиента (моменты собирает noise.py)
def noise_scale(moments: dict) -> dict:
    """Масштаб шума: E|g_B|^2 = |G|^2 + S/B (S - след ковариации градиента на один пример).

    Оценка МНК по всем batch size и по паре (минимальный, максимальный). B_simple = S / |G|^2 (McCandlish и др., 2018):
    при batch size заметно меньше B_simple шаг определяется шумом, заметно больше - сигналом."""
    moments = {int(k): v for k, v in moments.items()}      # json хранит ключи строками
    Bs = np.array(sorted(moments), float)
    y = np.array([moments[int(b)]["mean_sq_norm"] for b in Bs])
    X = np.column_stack([np.ones(len(Bs)), 1.0 / Bs])
    G2, Sn = np.linalg.lstsq(X, y, rcond=None)[0]
    B0, B1, y0, y1 = Bs[0], Bs[-1], y[0], y[-1]
    G2p = (B1 * y1 - B0 * y0) / (B1 - B0)
    Sp = (y0 - y1) / (1 / B0 - 1 / B1)
    return {"G2_fit": float(G2), "S_fit": float(Sn), "B_simple_fit": float(Sn / G2) if G2 > 0 else float("nan"),
            "G2_pair": float(G2p), "S_pair": float(Sp), "B_simple_pair": float(Sp / G2p) if G2p > 0 else float("nan")}


def noise_rows(moments: dict) -> pd.DataFrame:
    """По batch size: след ковариации, B*trace (должно быть ~константой), оценка сигнала без смещения и шум/сигнал."""
    moments = {int(k): v for k, v in moments.items()}
    rows = []
    for b in sorted(moments):
        m = moments[b]
        signal = m["mean_norm_sq"] - m["trace_cov"] / m["n"]       # |mean|^2 смещён вверх на tr(Cov)/n
        rows.append({"batch_size": b, "n_batches": m["n"], "trace_cov": m["trace_cov"], "trace_cov_x_B": m["trace_cov"] * b,
                     "mean_sq_norm": m["mean_sq_norm"], "signal_sq_debiased": signal,
                     "noise_to_signal": m["trace_cov"] / signal if signal > 0 else float("nan"),
                     "cos_to_ref": m.get("cos_to_ref", float("nan")), "mean_loss": m.get("mean_loss", float("nan"))})
    return pd.DataFrame(rows)
