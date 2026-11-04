import numpy as np
import pandas as pd

from vkr.lab4 import stats as S
from vkr.lab4.optimizers import OPTIMIZERS


def synthetic_runs(seed=0, n_seeds=10):
    rng = np.random.default_rng(seed)
    quality = dict(zip(OPTIMIZERS, [0.80, 0.84, 0.845, 0.82, 0.835, 0.90, 0.905, 0.895]))
    rows = []
    for o in OPTIMIZERS:
        for s in range(n_seeds):
            f1 = quality[o] + rng.normal(0, 0.01)
            rows.append({"run_id": f"main-{o}-{s}", "phase": "main", "optimizer": o, "seed": s, "val_f1_final": f1,
                         "test_f1_main": f1, "test_ece_main": 0.05 + rng.normal(0, 0.005),
                         "steps_to_target": 300 if o in ("adam", "adamw", "radam") else (500 if o != "sgd" else ""),
                         "train_time_s": 100 + rng.normal(0, 3), "peak_mem_mb": 500 + 10 * OPTIMIZERS.index(o),
                         "unstable": int(o == "sgd" and s < 3), "test_f1_S0": 0.99, "test_f1_S3": f1 - 0.05,
                         "test_f1_S4": f1 - 0.03, "test_f1_S5": f1 - 0.02})
    return pd.DataFrame(rows)


SEL = {"bootstrap_resamples": 2000, "weights": {"f1": 0.4, "std": 0.2, "ece": 0.15, "steps_to_target": 0.15, "memory": 0.10},
       "max_unstable_share": 0.2}


def test_bootstrap_ci_contains_mean():
    m, lo, hi = S.bootstrap_ci([0.8, 0.82, 0.81, 0.79, 0.83], 2000)
    assert lo <= m <= hi


def test_holm_monotone_and_bounded():
    adj = S.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == 0.03 and adj["b"] <= adj["c"] <= 1.0


def test_composite_and_choice():
    runs = synthetic_runs()
    summ = S.summarize(runs, SEL, 600)
    score = S.composite(summ, SEL["weights"])
    assert score.between(0, 1).all()
    res = S.choose_winner(summ, score, 0.2)
    assert res["winner"] in {"adam", "adamw", "radam"}
    assert res["table"].loc["sgd", "unstable_share"] == 0.3      # SGD отсеян фильтром
    assert res["winner"] != "sgd"


def test_minmax_ties_do_not_penalise():
    assert (S._minmax(pd.Series([1.0, 1.0, 1.0]), -1) == 1.0).all()


def test_wilcoxon_and_hypotheses():
    runs = synthetic_runs()
    summ = S.summarize(runs, SEL, 600)
    w = S.pairwise_wilcoxon(runs, "adamw")
    assert "sgd" in w.index and (w["p_holm"] >= w["p_raw"] - 1e-12).all()
    scen = S.scenario_table(runs, ["S0", "S3", "S4", "S5"])
    hyp = S.hypotheses(summ, scen, "adamw", {"H1_min_speedup": 0.3, "H2_min_gap": 0.02}, runs)
    h = {x["h"]: x for x in hyp}
    assert h["H2"]["ok"] is True
    assert h["H1 (adam vs sgd)"]["ok"] is True      # sgd не достигает цели: цензурируется бюджетом шагов


def test_pareto_flags_dominated():
    summ = pd.DataFrame({"f1_mean": [0.9, 0.8], "train_time_min": [1.0, 2.0], "f1_std": [0.01, 0.02]}, index=["a", "b"])
    ok = S.pareto_nondominated(summ)
    assert ok["a"] and not ok["b"]
