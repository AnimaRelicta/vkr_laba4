"""Этап 7 без torch: плато, bootstrap по документам, LOCO, оценка шума, sensitivity-таблицы, сборка отчёта на демо-данных."""
import json

import numpy as np
import pandas as pd
import pytest

from vkr.lab4 import stochastic as X
from vkr.lab4.config import Config
from vkr.lab4.data import load_observations, split_frames
from vkr.lab4.demo_data import make_demo
from vkr.lab4.labels import LABEL2ID, N_CLASSES
from vkr.lab4.metrics import macro_f1
from vkr.lab4.mutations import build_set
from vkr.lab4.optimizers import OPTIMIZERS


def test_plateau_interpolates_and_flags_edges():
    x, y = [-2, -1, 0, 1, 2], [0.0, 0.5, 0.9, 0.95, 0.2]
    p = X.plateau(x, y, 0.1)                       # порог 0,85: точки 0 и 1
    assert p["best_x"] == 1 and p["n_points"] == 2 and not p["left_censored"] and not p["right_censored"]
    assert p["x_lo"] == pytest.approx(-1 + (0.85 - 0.5) / (0.9 - 0.5))
    assert p["x_hi"] == pytest.approx(1 + (0.95 - 0.85) / (0.95 - 0.2))
    q = X.plateau([0, 1, 2], [0.9, 0.9, 0.9], 0.01)
    assert q["left_censored"] and q["right_censored"] and q["width_decades"] == 2


def _rand_preds(rng, n=300, n_docs=20):
    y = rng.integers(0, N_CLASSES, n)
    pred = np.where(rng.random(n) < 0.7, y, rng.integers(0, N_CLASSES, n))
    doc = np.sort(rng.integers(0, n_docs, n))
    return y, pred, doc


def test_f1_from_weights_matches_macro_f1():
    rng = np.random.default_rng(0)
    y, pred, doc = _rand_preds(rng)
    cnt = X.count_arrays(y, pred, doc, 20)
    assert X.f1_from_weights(cnt, np.ones(20)) == pytest.approx(macro_f1(y, pred))
    w = rng.multinomial(20, np.full(20, 0.05)).astype(float)
    rows = np.repeat(np.arange(len(y)), w[doc].astype(int))
    assert X.f1_from_weights(cnt, w) == pytest.approx(macro_f1(y[rows], pred[rows]))


def test_bootstrap_shared_resamples_and_ci():
    rng = np.random.default_rng(1)
    y, pa, doc = _rand_preds(rng)
    pb = np.where(rng.random(len(y)) < 0.4, y, rng.integers(0, N_CLASSES, len(y)))      # заметно хуже
    cnt = {"a": np.stack([X.count_arrays(y, pa, doc, 20)] * 3), "b": np.stack([X.count_arrays(y, pb, doc, 20)] * 3)}
    boot = X.bootstrap_units(cnt, 20, B=500, seed=3)
    lo, hi = X.ci(boot["a"]["units"])
    assert lo < X.point_estimate(cnt["a"]) < hi
    d = X.paired_diffs(boot)
    assert d.loc[0, "diff_mean"] > 0 and d.loc[0, "p_boot"] < 0.05
    assert boot["a"]["units"].shape == (500,)


def test_loco_equals_filtered_f1():
    rng = np.random.default_rng(2)
    y, pred, doc = _rand_preds(rng)
    cluster = np.arange(20) % 4
    t = X.loco({"a": X.count_arrays(y, pred, doc, 20)[None]}, cluster)
    row = t[t["left_out_cluster"] == 1].iloc[0]
    keep = cluster[doc] != 1
    assert row["f1_without"] == pytest.approx(macro_f1(y[keep], pred[keep]))
    assert row["n_docs_left_out"] == 5


def test_noise_scale_recovers_parameters():
    G2, S = 0.5, 40.0
    mom = {str(B): {"n": 200, "mean_sq_norm": G2 + S / B, "mean_norm_sq": G2 + S / B / 200 * 1.0, "trace_cov": S / B} for B in (16, 32, 64, 128)}
    r = X.noise_scale(mom)
    assert r["G2_fit"] == pytest.approx(G2) and r["S_fit"] == pytest.approx(S) and r["B_simple_fit"] == pytest.approx(80)
    assert r["B_simple_pair"] == pytest.approx(80)
    rows = X.noise_rows(mom)
    assert rows["trace_cov_x_B"].nunique() == 1 or np.allclose(rows["trace_cov_x_B"], S)


def _sens_runs(lrs, bs0=64):
    rows = []
    quality = {"adam": 0.80, "rmsprop": 0.78, "sgd": 0.70}
    for o, q in quality.items():
        for s in (0, 1):
            def add(mult=1.0, bs=bs0, wd=None, f1=q):
                rows.append({"run_id": f"sens-{o}-{mult}-{bs}-{wd}-{s}", "phase": "sens", "optimizer": o, "lr": lrs[o] * mult, "seed": s,
                             "batch_size": bs, "weight_decay": wd if wd is not None else np.nan, "val_f1_final": f1 + 0.1,
                             "test_f1_main": f1, "unstable": 0})
            for m, drop in ((0.1, 0.2), (0.333, 0.05), (1.0, 0.0), (3.0, 0.03), (10.0, 0.7)):
                add(mult=m, f1=q - drop + 0.01 * s)
            add(bs=32, f1=q + 0.01)
            add(bs=128, f1=q - 0.01)
            add(wd=0.0, f1=q)
            add(wd=1e-4, f1=q - 0.005)
    return pd.DataFrame(rows)


def test_sens_tables_ranks_and_plateau():
    cfg = Config()
    lrs = {"adam": 1e-3, "rmsprop": 3e-4, "sgd": 0.3}
    t = X.sens_tables(_sens_runs(lrs), lrs, cfg.grid, 64)
    assert set(t["factor"]) == {"lr", "bs", "wd"}
    lr = t[t["factor"] == "lr"]
    assert sorted(lr["x"].unique()) == [0.1, 0.333, 1.0, 3.0, 10.0]
    base = lr[(lr["x"] == 1.0)].set_index("optimizer")["rank"]
    assert base["adam"] == 1 and base["sgd"] == 3
    st = X.rank_stability(t)
    assert (st["n_distinct_winners"] == 1).all()
    pt = X.plateau_table(lr, "f1", True, deltas=(0.06,))
    adam = pt[pt["optimizer"] == "adam"].iloc[0]
    assert adam["best_x"] == pytest.approx(0.0, abs=1e-6) and adam["n_points"] >= 3 and adam["width_decades"] > 0.5


def test_adamw_default_wd_is_effective_value():
    cfg = Config()
    r = pd.DataFrame([{"run_id": "a", "phase": "sens", "optimizer": "adamw", "lr": 1e-3, "seed": 0, "batch_size": 64, "weight_decay": np.nan,
                       "val_f1_final": 0.9, "test_f1_main": 0.8, "unstable": 0}])
    d = X.sens_tidy(r, {"adamw": 1e-3}, cfg.grid)
    assert d.loc[0, "wd"] == 0.01 and d.loc[0, "wd_is_base"]


def test_report_end_to_end_on_demo_data(tmp_path):
    from vkr.lab4.stoch_report import build_stoch_report
    cfg = Config()
    cfg.protocol["paths"]["results"] = str(tmp_path / "res")
    cfg.protocol["data"]["dir"] = str(make_demo(tmp_path / "demo"))
    cfg.protocol["train"]["eval_seeds"] = [0, 1, 2]
    df = load_observations(cfg.protocol["data"]["dir"])
    test = split_frames(df)["test"]
    t, mc, sc = cfg.protocol["train"], cfg.mutations, cfg.scenarios
    sets = {n: build_set(test, s["mutations"], "test", mc, mc["seeds"]["test"], 1 if n == "S0" else t["copies_test"], keep_original=(n == "S0"))
            for n, s in sc["scenarios"].items() if "mutations" in s and n in X.SAVED_SCENARIOS}
    rng = np.random.default_rng(0)
    pdir = cfg.results_dir / "probs"
    pdir.mkdir(parents=True)
    lrs, rows = {}, []
    for k, o in enumerate(OPTIMIZERS):
        lrs[o] = 1e-3
        for s in range(3):
            rid = f"main-{o}-lr0.001-bs64-wddef-s{s}"
            row = {"run_id": rid, "phase": "main", "optimizer": o, "seed": s, "lr": 1e-3, "batch_size": 64, "steps_to_target": 40 + 5 * k,
                   "unstable": 0, "train_time_s": 70, "peak_mem_mb": 4000, "val_f1_final": 0.9}
            out = {}
            for n, d in sets.items():
                y = d["label"].map(LABEL2ID).to_numpy()
                P = rng.random((len(d), N_CLASSES)) * 0.2
                ok = rng.random(len(d)) < (0.9 - 0.02 * k)
                P[np.arange(len(d)), np.where(ok, y, rng.integers(0, N_CLASSES, len(d)))] += 1.0
                out[n] = P.astype(np.float32)
                row[f"test_f1_{n}"] = macro_f1(y, P.argmax(1))
                row[f"test_ece_{n}"] = 0.05
            row["test_f1_main"], row["test_ece_main"] = row["test_f1_MAIN"], 0.05
            np.savez_compressed(pdir / f"{rid}.npz", **out)
            rows.append(row)
    runs = pd.DataFrame(rows)
    nd = cfg.results_dir / "noise"
    nd.mkdir(parents=True)
    mom = {str(B): {"n": 50, "mean_sq_norm": 0.5 + 40 / B, "mean_norm_sq": 0.51, "trace_cov": 40 / B, "mean_loss": 1.0, "cos_to_ref": 0.9} for B in (16, 32, 64, 128)}
    (nd / "noise_adam_s0.json").write_text(json.dumps({"optimizer": "adam", "seed": 0, "lr": 1e-3, "checkpoint_step": 40, "n_batches": 50, "moments": mom}))
    p = build_stoch_report(cfg, runs, None, 200)
    text = p.read_text(encoding="utf-8")
    out = cfg.results_dir / "stochastic"
    assert "Bootstrap по документам" in text and "Карта источников случайности" in text and "B_simple" in text
    assert "расходится" not in text and "анализ пропущен" not in text.lower()
    assert (out / "bootstrap_ci.csv").exists() and (out / "noise_scale.csv").exists() and (out / "bootstrap_ci.png").exists()
    bt = pd.read_csv(out / "bootstrap_ci.csv", encoding="utf-8-sig")
    assert ((bt["ci_lo"] <= bt["f1"]) & (bt["f1"] <= bt["ci_hi"])).all()


def test_sens_reproducibility_pairs_base_run_with_main():
    lrs = {"adam": 1e-3, "sgd": 0.3}
    rows = []
    for o in lrs:
        for s in (0, 1):
            rows.append({"run_id": f"m{o}{s}", "phase": "main", "optimizer": o, "lr": lrs[o], "seed": s, "batch_size": 64,
                         "weight_decay": np.nan, "test_f1_main": 0.8 + s / 100, "unstable": 0})
            rows.append({"run_id": f"b{o}{s}", "phase": "sens", "optimizer": o, "lr": lrs[o], "seed": s, "batch_size": 64,
                         "weight_decay": np.nan, "test_f1_main": 0.8 + s / 100 + (0.003 if o == "sgd" else 0), "unstable": 0})
            rows.append({"run_id": f"x{o}{s}", "phase": "sens", "optimizer": o, "lr": lrs[o] * 3, "seed": s, "batch_size": 64,
                         "weight_decay": np.nan, "test_f1_main": 0.5, "unstable": 0})     # не базовый: не должен попасть
    rep = X.sens_reproducibility(pd.DataFrame(rows), lrs, 64)
    assert len(rep) == 4
    assert rep[rep.optimizer == "adam"]["abs_diff"].max() < 1e-12
    assert rep[rep.optimizer == "sgd"]["abs_diff"].max() == pytest.approx(0.003)


def test_scenario_test_sets_match_context_logic(splits):
    from vkr.lab4.mutations import scenario_test_sets
    cfg = Config()
    sets = scenario_test_sets(cfg.scenarios, cfg.mutations, 2, splits["test"])
    assert set(sets) == {"S0", "S1", "S2", "S3", "S4", "MAIN"}          # S5 строится отдельно (цепь Маркова)
    mc = cfg.mutations
    manual = build_set(splits["test"], cfg.scenarios["scenarios"]["MAIN"]["mutations"], "test", mc, mc["seeds"]["test"], 2, keep_original=False)
    assert sets["MAIN"].equals(manual) and len(sets["S0"]) == len(splits["test"])
