"""Проверка чисел отчёта: каждое число в тексте должно находиться в results/<backend>/ (runs.csv, stats, stochastic, tuning, noise, transfer).

Число считается подтверждённым, если округление какого-то значения из результатов до того же числа знаков даёт его (с учётом процентов).
Помимо отдельных значений в пул входят средние и SD по группам (фаза, оптимизатор) каждой числовой колонки runs.csv.
Без torch; читает .md, .txt и .docx (без внешних зависимостей).
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NUM = re.compile(r"(?<![\w.,/-])[-+−]?\d+(?:[.,]\d+)?(?:\s?%)?(?![\w])")


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
        xml = re.sub(r"</w:p>", "\n", xml)
        xml = re.sub(r"</w:tc>", " | ", xml)
        t = re.sub(r"<[^>]+>", "", xml)
        return (t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'"))
    return path.read_text(encoding="utf-8-sig")


def _flatten(o, acc: list):
    if isinstance(o, dict):
        for v in o.values():
            _flatten(v, acc)
    elif isinstance(o, list):
        for v in o:
            _flatten(v, acc)
    elif isinstance(o, (int, float)) and not isinstance(o, bool) and np.isfinite(o):
        acc.append(float(o))


def collect_pool(results_dir: Path, extra: list[float] | None = None) -> np.ndarray:
    vals: list[float] = list(extra or [])
    for p in sorted(results_dir.rglob("*.csv")):
        if "probs" in p.parts or "logs" in p.parts:
            continue
        try:
            df = pd.read_csv(p, encoding="utf-8-sig")
        except Exception:
            continue
        num = df.select_dtypes("number")
        vals += [float(x) for x in num.to_numpy().ravel() if np.isfinite(x)]
        if p.name in ("runs.csv", "pilot_runs.csv") and {"phase", "optimizer"} <= set(df.columns):
            for _, g in df.groupby(["phase", "optimizer"]):
                gn = g.select_dtypes("number")
                for stat in (gn.mean(), gn.std(ddof=1), gn.min(), gn.max(), gn.median()):
                    vals += [float(x) for x in stat if np.isfinite(x)]
            gn = df.select_dtypes("number")
            vals += [float(x) for x in gn.mean() if np.isfinite(x)]
    for p in sorted(results_dir.rglob("*.json")):
        try:
            _flatten(json.loads(p.read_text(encoding="utf-8")), vals)
        except Exception:
            pass
    return np.unique(np.abs(np.array(vals, dtype=float)))        # знак не сверяется: «падение 0,03» и разность -0,03 это одно число


def _strip(text: str) -> str:
    """Убирает то, что не является результатом: ссылки [12], даты, нумерацию строк и разделов, годы."""
    t = re.sub(r"\[\d+(?:[,\s–-]+\d+)*\]", " ", text)
    t = re.sub(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b", " ", t)
    t = re.sub(r"(?m)^\s*(?:#+\s*)?\d+(?:\.\d+)*[.)]?\s+", " ", t)
    t = re.sub(r"\b(?:19|20)\d{2}\b", " ", t)
    t = re.sub(r"(?i)\b(?:S|H|M|T|ЛР|лаб\.?|табл\.?|рис\.?|раздел\w*|п\.)\s?\d+\b", " ", t)
    return t


def parse(token: str) -> tuple[float, int, bool]:
    pct = token.rstrip().endswith("%")
    s = token.replace("%", "").replace("−", "-").replace(" ", "").replace(",", ".")
    dec = len(s.split(".")[1]) if "." in s else 0
    return float(s), dec, pct


def is_confirmed(v: float, dec: int, pct: bool, pool: np.ndarray) -> bool:
    """pool: отсортированные абсолютные значения. Совпадение после округления до тех же знаков (для процентов ещё и доля = v/100)."""
    for x, d in [(abs(v), dec)] + ([(abs(v) / 100, dec + 2)] if pct else []):
        tol = 0.5 * 10 ** -d + 1e-9
        i = np.searchsorted(pool, x - tol)
        if i < len(pool) and pool[i] <= x + tol:
            return True
    return False


def check_text(text: str, pool: np.ndarray, allow: list[float] | None = None, min_int: int = 21) -> dict:
    """Возвращает {'ok': [...], 'bad': [...], 'ints': [...]}: bad - десятичные числа без источника, ints - целые без источника."""
    allow = np.unique(np.abs(np.array(allow or [], dtype=float)))
    full = np.unique(np.concatenate([pool, allow])) if len(allow) else pool
    clean = _strip(text)
    res = {"ok": [], "bad": [], "ints": []}
    for m in NUM.finditer(clean):
        tok = m.group(0)
        v, dec, pct = parse(tok)
        ctx = clean[max(0, m.start() - 35):m.end() + 25].replace("\n", " ").strip()
        item = {"число": tok.strip(), "контекст": ctx}
        if is_confirmed(v, dec, pct, full):
            res["ok"].append(item)
        elif dec == 0 and not pct and abs(v) < min_int:
            continue                                   # маленькие целые (число seed, номера) не проверяем
        elif dec == 0 and not pct:
            res["ints"].append(item)
        else:
            res["bad"].append(item)
    return res


def run_check(cfg, path: Path, allow: list[float] | None = None, out: Path | None = None) -> tuple[Path, dict]:
    pool = collect_pool(cfg.results_dir)
    if len(pool) == 0:
        raise SystemExit(f"В {cfg.results_dir} нет результатов для сверки.")
    res = check_text(read_text(path), pool, allow)
    out = out or cfg.results_dir / "stats" / "check_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    def table(items):
        return "\n".join(["| число | контекст |", "| --- | --- |"] + [f"| {i['число']} | …{i['контекст']}… |" for i in items]) if items else "нет\n"

    L = [f"# Проверка чисел отчёта: {path.name}\n",
         f"Подтверждено значениями из results: {len(res['ok'])}. Без источника (десятичные): {len(res['bad'])}. "
         f"Целые от 21 и выше без источника: {len(res['ints'])}. В пуле {len(pool)} уникальных значений.\n",
         "## Десятичные числа без источника (исправить или объяснить)\n", table(res["bad"]), "\n",
         "## Целые числа без источника (проверить вручную)\n", table(res["ints"]), "\n",
         "Числа ЛР1 (например 0,9640, 255, 216) в results нет: передайте их через `--allow`. Числа внутри ссылок [n], даты, годы и "
         "нумерация разделов пропускаются.\n"]
    out.write_text("\n".join(L), encoding="utf-8")
    return out, res
