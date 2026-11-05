"""Задание 4 и проверка чисел отчёта (без torch)."""
import zipfile

import numpy as np
import pandas as pd
import pytest

from vkr.lab4 import check_report as C
from vkr.lab4.runner import FIELDS
from vkr.lab4.transfer import build_transfer, paired


def _runs(cfg):
    rng = np.random.default_rng(0)
    rows = []
    base = {"adagrad": 0.78, "rmsprop": 0.78, "sgd": 0.73, "adam": 0.76}
    for o, b in base.items():
        for s in range(5):
            rows.append({"run_id": f"main-{o}-{s}", "phase": "main", "optimizer": o, "lr": 0.01, "seed": s, "batch_size": 64,
                         "steps_to_target": 20 + (30 if o == "sgd" else 0) + s, "unstable": 0, "val_f1_final": 0.93,
                         "test_f1_S0": 0.99, "test_f1_S3": b + 0.02, "test_f1_S4": 0.9, "test_f1_S5": 0.91, "test_f1_main": b + rng.normal(0, 0.02),
                         "test_ece_main": 0.07, "train_time_s": 75.0, "peak_mem_mb": 4372.2})
    return pd.DataFrame(rows, columns=FIELDS)


def test_paired_diff_and_minimum_p(cfg):
    runs = _runs(cfg)
    r = paired(runs, "adagrad", "sgd", "test_f1_main", 80, 500)
    assert r["n_seeds"] == 5 and r["diff"] > 0.03 and r["min_possible_p"] == pytest.approx(0.0625)
    assert r["p_wilcoxon"] >= 0.0625 - 1e-12                       # при n=5 значимости нет в принципе


def test_transfer_report(cfg, tmp_path):
    p = build_transfer(cfg, _runs(cfg), final="adagrad", lab1={"f1": 0.964, "n": 255, "n4": 216, "name": "LLM"}, out=tmp_path)
    t = p.read_text(encoding="utf-8")
    assert "adagrad" in t and "напрямую числа не сравнимы" in t.lower() and "0.9640" in t
    pairs = pd.read_csv(tmp_path / "transfer_pairs.csv", encoding="utf-8-sig")
    assert set(pairs["против"]) == {"sgd", "adam"}


def test_checker_confirms_and_flags(cfg, tmp_path):
    res = tmp_path / "res"
    (res / "stats").mkdir(parents=True)
    pd.DataFrame({"optimizer": ["a", "b"], "f1_mean": [0.77512, 0.7371], "steps": [28, 54], "diff": [-0.0527, 0.0]}).to_csv(res / "stats" / "summary.csv")
    pool = C.collect_pool(res)
    text = ("RMSprop дал F1 0,775, SGD 0,737 [12]. Разрыв 0,0527 (5,27 %). Шагов 28 и 54 в 2026 году. Выдуманное число 0,801. "
            "Ещё 1234 запуска. 1. Раздел S3, H2.")
    r = C.check_text(text, pool)
    bad = [i["число"] for i in r["bad"]]
    assert bad == ["0,801"]
    assert any(i["число"].startswith("5,27") for i in r["ok"]) and any(i["число"] == "0,775" for i in r["ok"])
    assert [i["число"] for i in r["ints"]] == ["1234"]
    assert not C.check_text("ЛР1: 0,9640 и 255 строк", pool, allow=[0.9640, 255])["bad"]


def test_docx_text_is_read(tmp_path):
    p = tmp_path / "r.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", '<w:document><w:body><w:p><w:r><w:t>F1 равен 0,775</w:t></w:r></w:p></w:body></w:document>')
    assert "0,775" in C.read_text(p)
