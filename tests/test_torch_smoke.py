"""Требует torch (на машине с GPU/CPU после `uv sync`). Проверяет весь цикл обучения на компактной модели и демо-данных."""
import pytest

torch = pytest.importorskip("torch")

from vkr.lab4.models import make_backend  # noqa: E402
from vkr.lab4.optimizers import OPTIMIZERS, lr_grid  # noqa: E402
from vkr.lab4.spec import RunSpec  # noqa: E402
from vkr.lab4.train import Context, train_run  # noqa: E402


@pytest.fixture(scope="module")
def ctx(cfg, splits):
    cfg.protocol["train"].update({"steps": 100, "eval_every": 50, "log_every": 25, "copies_train": 2, "copies_val": 1, "copies_test": 1})
    cfg.protocol["model"]["backend"] = "ngram"
    return Context(cfg, splits, make_backend(cfg.protocol), "cpu")


def test_ngram_model_is_small(ctx):
    from vkr.lab4.models import n_trainable
    assert n_trainable(ctx.backend.build_model(0)) < 1_000_000


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_all_optimizers_train(ctx, opt):
    row = train_run(RunSpec("pilot", opt, lr_grid(ctx.cfg.grid, opt)[3], 0), ctx, None, eval_scenarios=True)
    assert row["steps_done"] == 100
    assert row["val_f1_final"] == row["val_f1_final"]       # не NaN
    for k in ("test_f1_S0", "test_f1_MAIN", "test_f1_S5", "test_ece_MAIN"):
        assert 0.0 <= row[k] <= 1.0


def test_same_seed_is_reproducible(ctx):
    a = train_run(RunSpec("pilot", "adam", 1e-3, 3), ctx, None, eval_scenarios=False)
    b = train_run(RunSpec("pilot", "adam", 1e-3, 3), ctx, None, eval_scenarios=False)
    assert a["val_f1_final"] == b["val_f1_final"] and a["final_loss"] == b["final_loss"]


def test_diverging_run_is_flagged(ctx):
    row = train_run(RunSpec("pilot", "sgd", 1e6, 0), ctx, None, eval_scenarios=False)
    assert row["unstable"] == 1


def test_step_hook_stops_training(ctx):
    calls = []
    row = train_run(RunSpec("noise", "adam", 1e-3, 0), ctx, None, eval_scenarios=False,
                    step_hook=lambda step, model: (calls.append(step), step == 20)[1])
    assert row["steps_done"] == 20 and calls[-1] == 20 and calls[0] == 1


def test_gradient_noise_decreases_with_batch(ctx):
    import numpy as np

    from vkr.lab4.noise import collect_moments
    from vkr.lab4.train import _labels
    tr, backend = ctx.train_set(0), ctx.backend
    items = [backend.encode_one(h, v) for h, v in zip(tr["header_text"], tr["values_sample"])]
    y = torch.tensor(_labels(tr), dtype=torch.long)
    model = backend.build_model(0)
    params = [p for p in model.parameters() if p.requires_grad]
    m = collect_moments(model, backend, items, y, params, [8, 32], 30, 32, "cpu", np.random.default_rng(0), log=lambda *a: None)
    assert m[8]["trace_cov"] > m[32]["trace_cov"] > 0
    assert abs(m[32]["cos_to_ref"] - 1.0) < 1e-5 and m[8]["n"] == 30
