"""Шум градиента на чекпоинте (этап 7, grid.yaml: noise). Нужен torch.

Обучаем оптимизатор с выбранным LR до шага checkpoint_step (полный протокол: тот же seed, то же расписание LR,
как в main), затем на замороженных весах собираем n_batches градиентов для каждого batch size и считаем моменты
потоково (Welford): след ковариации, квадрат нормы среднего градиента, средний квадрат нормы.
Модель остаётся в train-режиме: dropout LoRA и головы входит в шум, как и в обучении. Градиенты берутся до clipping.
Оценки масштаба шума по моментам - в stochastic.noise_scale (без torch).
"""
from __future__ import annotations

import json
import time

import numpy as np


def collect_moments(model, backend, items, y_all, params, batch_sizes, n_batches, micro, device, rng, log=print) -> dict:
    import torch
    from torch import nn

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.train()
    out, means = {}, {}
    for B in batch_sizes:
        if B > len(items):
            raise ValueError(f"batch size {B} больше обучающей выборки ({len(items)})")
        mean = M2 = None
        n, sq, losses, t0 = 0, [], [], time.perf_counter()
        for _ in range(n_batches):
            idx = rng.choice(len(items), size=B, replace=False)
            model.zero_grad(set_to_none=True)
            total = 0.0
            for j in range(0, B, micro or B):
                sub = idx[j:j + (micro or B)]
                batch = backend.collate([items[i] for i in sub], device)
                loss = loss_fn(model(**batch).float(), y_all[sub].to(device)) / B
                loss.backward()
                total += loss.item()
            g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params]).double()
            if not bool(torch.isfinite(g).all()):
                continue
            if mean is None:
                mean, M2 = torch.zeros_like(g), torch.zeros_like(g)
            n += 1
            d = g - mean
            mean += d / n
            M2 += d * (g - mean)
            sq.append(float(g.dot(g)))
            losses.append(total)
        if n < 2:
            raise RuntimeError(f"batch size {B}: меньше двух конечных градиентов из {n_batches}")
        out[B] = {"n": n, "mean_sq_norm": float(np.mean(sq)), "mean_norm_sq": float(mean.dot(mean)),
                  "trace_cov": float(M2.sum()) / (n - 1), "mean_loss": float(np.mean(losses)),
                  "seconds": round(time.perf_counter() - t0, 1)}
        means[B] = mean.float().cpu()
        log(f"  batch {B:4d}: tr(Cov)={out[B]['trace_cov']:.4g}  |mean|^2={out[B]['mean_norm_sq']:.4g}  ({out[B]['seconds']} с)")
    ref = means[max(means)]
    for B, m in means.items():
        out[B]["cos_to_ref"] = float(torch.nn.functional.cosine_similarity(m, ref, dim=0))
    model.zero_grad(set_to_none=True)
    return out


def run_noise(cfg, ctx, optimizers=None, seeds=(0,), n_batches=None) -> list:
    """Для каждого (оптимизатор, seed) пишет results/<backend>/noise/noise_<opt>_s<seed>.json. Возобновляемо."""
    import torch

    from .runner import load_selected_lr, read_runs
    from .spec import RunSpec
    from .stochastic import finalists
    from .train import _labels, train_run

    ncfg, tcfg = cfg.grid["noise"], cfg.protocol["train"]
    ckpt, bss = ncfg["checkpoint_step"], list(ncfg["batch_sizes"])
    nb = n_batches or ncfg["n_batches"]
    if nb != ncfg["n_batches"]:
        print(f"ВНИМАНИЕ: n_batches={nb} вместо {ncfg['n_batches']} из grid.yaml - отметьте отступление в отчёте.")
    if ckpt > tcfg["steps"]:
        raise SystemExit(f"checkpoint_step {ckpt} больше числа шагов {tcfg['steps']}")
    lrs = load_selected_lr(cfg)
    if optimizers is None:
        runs = read_runs(cfg)
        optimizers = finalists(runs, cfg.protocol["selection"], tcfg["steps"]) + ["sgd"]
        optimizers = list(dict.fromkeys(optimizers))
    out_dir = cfg.results_dir / "noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Шум градиента: {optimizers}, seed {list(seeds)}, чекпоинт {ckpt}, batch {bss}, {nb} батчей на размер "
          f"(~{nb * sum(bss) * 0.0135 / 60:.0f} мин на запуск при Qwen на RTX 4060)")
    done = []
    for o in optimizers:
        for seed in seeds:
            path = out_dir / f"noise_{o}_s{seed}.json"
            if path.exists():
                print(f"{path.name}: уже есть, пропуск")
                continue
            spec = RunSpec("noise", o, lrs[o], seed, tcfg["batch_size"])
            tr = ctx.train_set(seed)
            items = [ctx.backend.encode_one(h, v) for h, v in zip(tr["header_text"], tr["values_sample"])]
            y_all = torch.tensor(_labels(tr), dtype=torch.long)
            result: dict = {}

            def hook(step, model, _o=o, _seed=seed, _items=items, _y=y_all, _res=result):
                if step != ckpt:
                    return False
                print(f"{_o} seed {_seed}: чекпоинт шага {step}, сбор градиентов")
                params = [p for p in model.parameters() if p.requires_grad]
                rng = np.random.default_rng(10_000 + _seed)
                _res.update(collect_moments(model, ctx.backend, _items, _y, params, bss, nb, ctx.backend.micro_batch,
                                            ctx.device, rng))
                return True

            train_run(spec, ctx, None, eval_scenarios=False, step_hook=hook)
            if not result:
                print(f"{o} seed {seed}: чекпоинт {ckpt} не достигнут (обучение остановилось раньше, см. нестабильность), пропуск")
                continue
            path.write_text(json.dumps({"optimizer": o, "seed": seed, "lr": lrs[o], "checkpoint_step": ckpt,
                                        "n_batches": nb, "moments": {str(k): v for k, v in result.items()}},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
            done.append(path)
    return done
