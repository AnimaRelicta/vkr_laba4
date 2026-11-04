"""Один запуск обучения: общий цикл для всех оптимизаторов (ТЗ, п. 3.3-3.5)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .labels import LABEL2ID, N_CLASSES
from .metrics import ece, macro_f1
from .models import n_trainable
from .mutations import build_set, markov_frames, scenario_test_sets
from .optimizers import make_optimizer
from .spec import RunSpec


class Context:
    """Всё, что одинаково для всех запусков: данные, фиксированные val/test-мутации, бэкенд, устройство."""

    def __init__(self, cfg, splits: dict, backend, device: str):
        self.cfg, self.splits, self.backend, self.device = cfg, splits, backend, device
        t, mc, sc = cfg.protocol["train"], cfg.mutations, cfg.scenarios
        self.val = build_set(splits["val"], sc["val_mutations"], "val", mc, mc["seeds"]["val"], t["copies_val"], keep_original=True)
        self.test_sets: dict[str, pd.DataFrame] = scenario_test_sets(sc, mc, t["copies_test"], splits["test"])
        self.markov = list(markov_frames(splits["test"], sc["scenarios"]["S5"], mc, mc["seeds"]["test"])) \
            if "S5" in sc["scenarios"] else []

    def train_set(self, seed: int) -> pd.DataFrame:
        t, mc = self.cfg.protocol["train"], self.cfg.mutations
        return build_set(self.splits["train"], self.cfg.scenarios["train_mutations"], "train", mc,
                         mc["seeds"]["train_base"] + seed, t["copies_train"], keep_original=True)


def _labels(df: pd.DataFrame) -> np.ndarray:
    return df["label"].map(LABEL2ID).to_numpy()


@torch.no_grad()
def predict_probs(model, backend, df: pd.DataFrame, device, cache: dict | None = None, bs: int | None = None) -> np.ndarray:
    """Вероятности классов; одинаковые (заголовок, значения) считаются один раз."""
    model.eval()
    bs = bs or getattr(backend, "eval_bs", 256)
    keys = list(zip(df["header_text"], df["values_sample"]))
    cache = {} if cache is None else cache
    todo = [k for k in dict.fromkeys(keys) if k not in cache]
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        batch = backend.collate([backend.encode_one(h, v) for h, v in chunk], device)
        p = torch.softmax(model(**batch).float(), dim=1).cpu().numpy()
        for k, row in zip(chunk, p):
            cache[k] = row
    return np.stack([cache[k] for k in keys])


def _peak_mem_mb(device: str, rss_peak: float) -> float:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**20
    return rss_peak


def _rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def train_run(spec: RunSpec, ctx: Context, log_dir: Path | None = None, eval_scenarios: bool = True,
              save_probs_dir: Path | None = None, step_hook=None) -> dict:
    """step_hook(done_steps, model) -> bool вызывается после каждого шага; True останавливает обучение (используется для
    замера шума градиента на чекпоинте). Время хука не входит в train_time_s. Без хука поведение прежнее."""
    cfg, backend, device = ctx.cfg, ctx.backend, ctx.device
    tcfg = cfg.protocol["train"]
    steps = spec.steps or tcfg["steps"]
    target = tcfg["target_val_f1"]
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(spec.seed)
    torch.manual_seed(spec.seed)

    tr = ctx.train_set(spec.seed)
    y_all = torch.tensor(_labels(tr), dtype=torch.long)
    items = [backend.encode_one(h, v) for h, v in zip(tr["header_text"], tr["values_sample"])]
    val_y = _labels(ctx.val)

    model = backend.build_model(spec.seed).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(spec.optimizer, params, spec.lr, cfg.grid, spec.weight_decay)
    warm = max(1, int(round(tcfg["warmup_frac"] * steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    micro = backend.micro_batch or spec.batch_size
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    log, val_curve = [], []
    steps_to_target, reason = None, ""
    win_losses, win_means, running_min = [], [], math.inf
    rss_peak = _rss_mb()
    perm, ptr = rng.permutation(len(items)), 0
    eval_time, t_start = 0.0, time.perf_counter()
    done_steps = 0

    for step in range(steps):
        if ptr + spec.batch_size > len(perm):
            perm, ptr = rng.permutation(len(items)), 0
        idx = perm[ptr:ptr + spec.batch_size]
        ptr += spec.batch_size
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for j in range(0, len(idx), micro):
            sub = idx[j:j + micro]
            batch = backend.collate([items[i] for i in sub], device)
            logits = model(**batch)
            loss = loss_fn(logits.float(), y_all[sub].to(device)) / len(idx)
            loss.backward()
            total += loss.item()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"]))
        if not math.isfinite(total) or not math.isfinite(gnorm):
            reason = "nan_inf"
            done_steps = step
            break
        opt.step()
        sched.step()
        done_steps = step + 1
        win_losses.append(total)
        if done_steps % tcfg["log_every"] == 0:
            w = float(np.mean(win_losses))
            win_losses = []
            win_means.append(w)
            running_min = min(running_min, w)
            log.append({"step": done_steps, "train_loss": w, "grad_norm": gnorm})
            rss_peak = max(rss_peak, _rss_mb())
            if len(win_means) > 2 and w > 3 * running_min and not reason:
                reason = "loss_growth_3x"
        if done_steps % tcfg["eval_every"] == 0 and spec.phase != "tune":     # при подборе LR нужен только итоговый val F1
            _sync(device)
            t0 = time.perf_counter()
            f1 = macro_f1(val_y, predict_probs(model, backend, ctx.val, device).argmax(1))
            _sync(device)
            eval_time += time.perf_counter() - t0
            val_curve.append((done_steps, f1))
            log.append({"step": done_steps, "val_f1": f1})
            if steps_to_target is None and f1 >= target:
                steps_to_target = done_steps
        if step_hook is not None:
            _sync(device)
            t0 = time.perf_counter()
            stop = step_hook(done_steps, model)
            _sync(device)
            eval_time += time.perf_counter() - t0
            if stop:
                reason = reason or "stopped_by_hook"
                break
        if (time.perf_counter() - t_start - eval_time) / 60 > tcfg["time_limit_min"]:
            reason = reason or "time_limit"
            break
    _sync(device)
    train_time = time.perf_counter() - t_start - eval_time

    row = {**asdict(spec), "steps": steps, "run_id": spec.run_id, "backend": backend.name, "steps_done": done_steps,
           "n_trainable": n_trainable(model), "train_time_s": round(train_time, 2),
           "final_loss": win_means[-1] if win_means else float("nan")}
    if reason == "nan_inf":
        row.update(val_f1_final=float("nan"), val_f1_best=float("nan"))
    else:
        cache: dict = {}
        vf = macro_f1(val_y, predict_probs(model, backend, ctx.val, device, cache).argmax(1))
        val_curve.append((done_steps, vf))
        row.update(val_f1_final=vf, val_f1_best=max(f for _, f in val_curve))
        if steps_to_target is None:
            reason = reason or "target_not_reached"
    row["steps_to_target"] = steps_to_target if steps_to_target is not None else ""
    row["unstable"] = int(bool(reason))
    row["unstable_reason"] = reason

    if eval_scenarios and reason != "nan_inf":
        cache = {}
        probs_out = {}
        for name, df in ctx.test_sets.items():
            P = predict_probs(model, backend, df, device, cache)
            y = _labels(df)
            row[f"test_f1_{name}"] = macro_f1(y, P.argmax(1))
            row[f"test_ece_{name}"] = ece(P, y, cfg.protocol["selection"]["ece_bins"])
            if name in ("S0", "S3", "S4", "MAIN"):
                probs_out[name] = P.astype(np.float32)
        if ctx.markov:
            f1s = []
            for frame, _chain in ctx.markov:
                f1s.append(macro_f1(_labels(frame), predict_probs(model, backend, frame, device, cache).argmax(1)))
            row["test_f1_S5"] = float(np.mean(f1s))
            row["test_f1_S5_sd"] = float(np.std(f1s))
        main = cfg.scenarios["main_scenario"]
        row["test_f1_main"], row["test_ece_main"] = row.get(f"test_f1_{main}"), row.get(f"test_ece_{main}")
        if save_probs_dir is not None:
            save_probs_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_probs_dir / f"{spec.run_id}.npz", **probs_out)
    row["peak_mem_mb"] = round(_peak_mem_mb(device, max(rss_peak, _rss_mb())), 1)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{spec.run_id}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in log), encoding="utf-8")
    return row
