"""Сверка чисел ЛР1 (TODO: «255 и 276 заголовков в test, 14 и 16 классов»).

Идея: одно и то же слово «заголовков» можно посчитать по-разному (строки, уникальные header_raw,
уникальные header_text, пары (класс, заголовок) и т.д.). Скрипт считает ВСЕ варианты для каждой
папки данных и показывает, какой вариант даёт 255, а какой 276 (и 14 / 16 для классов).
Данные не меняет; пишет artifacts/data_check.md.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .labels import LABELS

SPLITS = ("train", "val", "test")


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.casefold()).strip()


def header_counts(rows: list[dict]) -> dict[str, int]:
    """Все разумные определения «числа заголовков» для набора наблюдений."""
    ht = lambda r: r.get("header_text", "")  # noqa: E731
    mapped = [r for r in rows if r.get("label") != "unmapped"]
    out = {
        "строк (наблюдений)": len(rows),
        "уникальных header_raw": len({r["header_raw"] for r in rows}) if rows and "header_raw" in rows[0] else -1,
        "уникальных header_text": len({ht(r) for r in rows}),
        "уникальных header_text (регистр и пробелы не важны)": len({_norm(ht(r)) for r in rows}),
        "уникальных пар (класс, header_text)": len({(r.get("label"), ht(r)) for r in rows}),
        "уникальных троек (класс, header_text, values_sample)": len({(r.get("label"), ht(r), r.get("values_sample")) for r in rows}),
        "строк без unmapped": len(mapped),
        "уникальных header_text без unmapped": len({ht(r) for r in mapped}),
        "уникальных пар (класс, header_text) без unmapped": len({(r.get("label"), ht(r)) for r in mapped}),
    }
    return {k: v for k, v in out.items() if v >= 0}


def class_counts(rows: list[dict]) -> dict[str, int]:
    labs = {r.get("label") for r in rows}
    return {"классов в gold": len(labs), "классов в gold без unmapped": len(labs - {"unmapped"})}


def load_dir(d: Path) -> dict:
    """observations.csv + observations_<split>.csv; проверяется, что файлы согласованы."""
    res: dict = {"dir": str(d), "splits": {}, "issues": []}
    full = d / "observations.csv"
    allrows = _read(full) if full.exists() else []
    if not allrows:
        res["issues"].append(f"нет {full}")
    for s in SPLITS:
        part = d / f"observations_{s}.csv"
        from_all = [r for r in allrows if r.get("split") == s]
        from_part = _read(part) if part.exists() else []
        if part.exists() and allrows and len(from_part) != len(from_all):
            res["issues"].append(f"{s}: в observations_{s}.csv {len(from_part)} строк, в observations.csv (split={s}) {len(from_all)}")
        res["splits"][s] = from_part or from_all
    res["all"] = allrows or [r for s in SPLITS for r in res["splits"][s]]
    return res


def llm_classes(d: Path) -> list[dict]:
    """Классы в preds/*.jsonl логов LLM из ЛР1: сколько уникальных gold и pred по запуску."""
    out = []
    for f in sorted((d / "preds").glob("*.jsonl")):
        gold, pred, n, hdr = set(), set(), 0, set()
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    n += 1
                    gold.add(r.get("gold"))
                    pred.add(r.get("pred"))
        out.append({"run": f.stem, "n": n, "gold": len(gold), "pred_valid": len(pred & set(LABELS)),
                    "pred_all": len(pred), "gold_not_in_labels": sorted(gold - set(LABELS))})
    return out


def grep_numbers(d: Path, numbers: list[int], limit: int = 12) -> list[str]:
    """Строки текстовых отчётов (md/txt/json/csv до 2 МБ), где встречаются заявленные числа."""
    pats = [re.compile(rf"(?<![\d.,]){n}(?![\d.,]\d)") for n in numbers]
    hits = []
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() not in {".md", ".txt", ".json"} or not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if any(rx.search(line) for rx in pats):
                    hits.append(f"{p.relative_to(d)}:{i}: {line.strip()[:160]}")
                    if len(hits) >= limit:
                        return hits
        except OSError:
            pass
    return hits


def _md(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    return "\n".join(out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def build_check(dirs: list[Path], llm_dirs: list[Path], claim_headers: list[int], claim_classes: list[int]) -> tuple[str, dict]:
    """-> (markdown, summary). summary['matches'][число] = список «где найдено»."""
    L = ["# Сверка чисел ЛР1\n",
         f"Проверяем: заголовков в test: {claim_headers}; классов: {claim_classes}. В проекте классов принято {len(LABELS)} (labels.py).\n"]
    matches: dict[int, list[str]] = {n: [] for n in claim_headers + claim_classes}

    def hit(val, where):
        if val in matches:
            matches[val].append(where)

    for d in dirs:
        info = load_dir(d)
        L.append(f"## Папка `{d}`\n")
        if info["issues"]:
            L += ["Проблемы: " + "; ".join(info["issues"]) + "\n"]
        if not info["all"]:
            continue
        rows_t = []
        for s in SPLITS:
            hc, cc = header_counts(info["splits"][s]), class_counts(info["splits"][s])
            for k, v in {**hc, **cc}.items():
                hit(v, f"{d.name}/{s}: {k} = {v}")
            rows_t.append((s, hc, cc))
        keys = list(rows_t[0][1])
        L.append(_md(["Определение"] + [s for s, *_ in rows_t] + ["все"],
                     [[k] + [hc.get(k, "") for _, hc, _ in rows_t] + [header_counts(info["all"]).get(k, "")] for k in keys]) + "\n")
        for k, v in header_counts(info["all"]).items():
            hit(v, f"{d.name}/все: {k} = {v}")
        ck = list(rows_t[0][2])
        L.append(_md(["Классы"] + [s for s, *_ in rows_t] + ["все"],
                     [[k] + [cc[k] for _, _, cc in rows_t] + [class_counts(info["all"])[k]] for k in ck]) + "\n")
        for k, v in class_counts(info["all"]).items():
            hit(v, f"{d.name}/все: {k} = {v}")
        lost = sorted(set(LABELS) - {r["label"] for r in info["all"]})
        extra = sorted({r["label"] for r in info["all"]} - set(LABELS))
        L.append(f"Классы из LABELS без наблюдений: {', '.join(lost) or 'нет'}. Классы в данных вне LABELS: {', '.join(extra) or 'нет'}.\n")
        found = grep_numbers(d, claim_headers + claim_classes)
        if found:
            L += ["Где эти числа встречаются в текстовых файлах папки:\n"] + [f"- `{x}`" for x in found] + [""]

    for d in llm_dirs:
        L.append(f"## Логи LLM `{d}`\n")
        if not d.exists():
            L.append(f"**Папки `{d}` не существует.** Проверьте путь (например, data/lab1/llm_out_hard).\n")
            continue
        rr = llm_classes(d)
        if not rr:
            L.append(f"Папка есть, но `{d / 'preds'}/*.jsonl` не найдены.\n")
            continue
        L.append(_md(["Запуск", "N", "gold", "pred (валидных)", "pred (всего)", "gold вне LABELS"],
                     [[r["run"][:60], r["n"], r["gold"], r["pred_valid"], r["pred_all"], r["gold_not_in_labels"] or ""] for r in rr]) + "\n")
        for r in rr:
            for key in ("n", "gold", "pred_valid", "pred_all"):
                hit(r[key], f"{d.name}/{r['run'][:40]}: {key} = {r[key]}")

    L.append("## Итог: где найдены заявленные числа\n")
    for n in claim_headers + claim_classes:
        L.append(f"- **{n}**: " + ("; ".join(matches[n][:8]) if matches[n] else "не найдено ни в одном определении") + (" …" if len(matches[n]) > 8 else ""))
    L.append("\nВывод делается вручную: выберите определение, которое использует отчёт ЛР1, и запишите его в паспорт данных. "
             "Если 255 и 276 отличаются, потому что один счёт ведётся по уникальным заголовкам, а другой по строкам или по `out_hard`, "
             "в отчёте ЛР4 нужно указывать оба числа с пояснением.\n")
    return "\n".join(L), {"matches": matches}
