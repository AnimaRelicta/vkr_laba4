import copy

from vkr.lab4.budget import plan_cost, plan_specs, smoke_config
from vkr.lab4.config import Config
from vkr.lab4.runner import sens_specs


def test_plan_sizes_and_budget(cfg):
    sp = plan_specs(cfg)
    assert {k: len(v) for k, v in sp.items()} == {"pilot": 8, "tune": 56, "main": 40, "sens": 54}
    cost = plan_cost(cfg, per_step_s=0.5, overhead_s=10)
    total_h = sum(v[2] for k, v in cost.items() if k != "pilot") / 3600
    assert total_h < 3          # при 0.5 с/шаг и 10 с накладных всё укладывается в несколько часов


def test_sens_has_no_duplicates(cfg):
    specs = sens_specs(cfg, ["adamw", "sgd"], {"adamw": 1e-3, "sgd": 0.1})
    assert len({s.run_id for s in specs}) == len(specs)
    # 5 LR + 2 нестандартных batch size + 2 нестандартных weight decay на каждый seed
    per_seed = len(specs) / (2 * len(cfg.protocol["train"]["sens_seeds"]))
    assert per_seed == 9


def test_smoke_config_is_small_and_isolated():
    c = smoke_config(Config(), steps=8)
    assert c.protocol["paths"]["results"] == "results_smoke"
    sp = plan_specs(c)
    assert len(sp["tune"]) == 16 and len(sp["main"]) == 16 and len(sp["sens"]) == 12
    assert all(s.steps in (0, 8) for v in sp.values() for s in v)
    assert Config().protocol["paths"]["results"] == "results"     # настоящий протокол не затронут
