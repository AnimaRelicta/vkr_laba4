import shutil

import pytest

from vkr.lab4 import config as C
from vkr.lab4.features import ngram_ids, normalize, value_features
from vkr.lab4.metrics import ece, macro_f1
from vkr.lab4.optimizers import OPTIMIZERS, lr_grid
from vkr.lab4.runner import FIELDS, pilot_specs, tune_specs
from vkr.lab4.spec import RunSpec

import numpy as np


def test_lock_detects_changes(tmp_path):
    d = tmp_path / "configs"
    shutil.copytree(C.CONFIG_DIR, d)
    lock = tmp_path / "lock.json"
    C.write_lock(d, lock)
    C.check_lock(d, lock)
    (d / "grid.yaml").write_text((d / "grid.yaml").read_text(encoding="utf-8").replace("n: 7", "n: 8", 1), encoding="utf-8")
    with pytest.raises(RuntimeError):
        C.check_lock(d, lock)


def test_lock_ignores_frozen_flag(tmp_path):
    d = tmp_path / "configs"
    shutil.copytree(C.CONFIG_DIR, d)
    lock = tmp_path / "lock.json"
    C.write_lock(d, lock)
    p = d / "protocol.yaml"
    p.write_text(p.read_text(encoding="utf-8").replace("frozen: false", "frozen: true"), encoding="utf-8")
    C.check_lock(d, lock)


def test_missing_lock_blocks(tmp_path):
    with pytest.raises(RuntimeError):
        C.check_lock(lock_path=tmp_path / "none.json")


def test_plan_sizes(cfg):
    assert len(OPTIMIZERS) == 8
    n = 8 * 7 * len(cfg.protocol["train"]["tuning_seeds"])
    assert len(tune_specs(cfg)) == n == 56
    assert len(pilot_specs(cfg)) == 8
    assert len({s.run_id for s in tune_specs(cfg)}) == n


def test_lr_grids(cfg):
    a, s = lr_grid(cfg.grid, "adam"), lr_grid(cfg.grid, "sgd")
    assert len(a) == len(s) == 7 and abs(a[0] - 1e-4) < 1e-12 and abs(a[-1] - 0.1) < 1e-9
    assert abs(s[0] - 1e-3) < 1e-12 and abs(s[-1] - 1.0) < 1e-9
    assert lr_grid(cfg.grid, "adamw") == a and lr_grid(cfg.grid, "nesterov") == s


def test_weight_decay_only_adamw(cfg):
    wd = {o: cfg.grid["optimizers"][o]["weight_decay"] for o in OPTIMIZERS}
    assert wd["adamw"] == 0.01 and all(v == 0 for o, v in wd.items() if o != "adamw")


def test_run_id_unique_per_setting():
    a = RunSpec("main", "adam", 1e-3, 0)
    assert a.run_id != RunSpec("main", "adam", 1e-3, 1).run_id != RunSpec("sens", "adam", 1e-3, 1, weight_decay=0.0).run_id


def test_features_stable_and_normalised():
    assert ngram_ids("Количество", 1000) == ngram_ids("  КОЛИЧЕСТВО ", 1000)
    assert ngram_ids("Цена", 32768) == ngram_ids("Цена", 32768)
    f = value_features("12.50 | 3")
    assert len(f) == 3 and f[2] == 1.0 and 0 <= f[0] <= 1


def test_metrics():
    assert macro_f1([0, 1, 1], [0, 1, 1]) == 1.0
    assert macro_f1([0, 1], [1, 0]) == 0.0
    p = np.array([[0.9, 0.1], [0.9, 0.1]])
    assert abs(ece(p, [0, 0]) - 0.1) < 1e-9
    assert len(FIELDS) == len(set(FIELDS))
