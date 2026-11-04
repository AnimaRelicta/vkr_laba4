"""Данные ЛР1: загрузка observations.csv, split.json, проверка утечек, паспорт данных."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import LABELS

REQUIRED = ["split", "doc_ref", "cluster_id", "header_text", "values_sample", "label"]
OPTIONAL = ["contract_id", "seller_id", "buyer_id", "doc_date", "n_rows", "n_values", "n_numeric", "col_label"]


def load_observations(data_dir: str | Path) -> pd.DataFrame:
    """Читает out/observations.csv (результат prepare_dataset.py из ЛР1)."""
    p = Path(data_dir) / "observations.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} не найден. Положите папку out/ из ЛР1 в data/lab1/out "
            "(или создайте демо-данные: uv run lab4 demo-data).")
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"В observations.csv нет колонок: {missing}")
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = ""
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Неизвестные классы в данных: {unknown}. Список классов: vkr.lab4.labels.LABELS")
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["col_label"] = pd.to_numeric(df["col_label"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["split", "doc_ref", "col_label"], kind="stable").reset_index(drop=True)
    return df


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}


def doc_tokens(df: pd.DataFrame) -> dict[str, set]:
    """Токены документа для Jaccard (как в prepare_dataset: заголовки и значения с меткой класса)."""
    toks: dict[str, set] = defaultdict(set)
    for r in df.itertuples():
        toks[r.doc_ref].add(f"H|{r.label}|{r.header_text}")
        for v in filter(None, (x.strip().lower() for x in r.values_sample.split(" | "))):
            toks[r.doc_ref].add(f"V|{r.label}|{v[:200]}")
    return toks


def leak_report(df: pd.DataFrame, jaccard_thr: float = 0.90) -> dict:
    """Повторная проверка утечек между train/val/test (ТЗ, п. 3.1)."""
    doc_split = df.drop_duplicates("doc_ref").set_index("doc_ref")["split"]
    rep: dict = {}
    for key in ("contract_id", "seller_id", "buyer_id"):
        sub = df[df[key] != ""].drop_duplicates(["doc_ref"])
        where = sub.groupby(key)["split"].nunique()
        rep[key] = {"values": int(len(where)), "in_several_splits": int((where > 1).sum())}
    clusters = df.drop_duplicates("doc_ref").groupby("cluster_id")["split"].nunique()
    rep["cluster_id"] = {"values": int(len(clusters)), "in_several_splits": int((clusters > 1).sum())}
    toks = doc_tokens(df)
    docs = list(toks)
    cross = 0
    near = 0
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            a, b = toks[docs[i]], toks[docs[j]]
            if len(a & b) / max(len(a | b), 1) >= jaccard_thr:
                near += 1
                cross += doc_split[docs[i]] != doc_split[docs[j]]
    rep["near_duplicate_pairs"] = {"threshold": jaccard_thr, "pairs": near, "across_splits": int(cross)}
    rep["clean"] = all(rep[k]["in_several_splits"] == 0 for k in ("contract_id", "seller_id", "buyer_id", "cluster_id")) and cross == 0
    return rep


def split_json(df: pd.DataFrame, leaks: dict) -> dict:
    docs = df.drop_duplicates("doc_ref")
    out = {"source": "ЛР1 (observations.csv), split не пересчитывался", "leak_check": leaks, "parts": {}}
    for s in ("train", "val", "test"):
        d = docs[docs["split"] == s]
        out["parts"][s] = {
            "documents": int(len(d)), "observations": int((df["split"] == s).sum()),
            "clusters": sorted(d["cluster_id"].unique().tolist()), "doc_refs": sorted(d["doc_ref"].tolist()),
        }
    return out


def _share(a, b) -> str:
    return f"{100 * a / b:.1f} %" if b else "n/a"


def _md(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def passport(df: pd.DataFrame, leaks: dict) -> str:
    """Паспорт данных заново по ТЗ, п. 3.1: классы, длина заголовков, пропуски, дубликаты шаблонов, время."""
    n = len(df)
    docs = df.drop_duplicates("doc_ref")
    L = df["header_text"].str.len()
    P = ["# Паспорт данных (ЛР4)\n",
         f"Источник: observations.csv из ЛР1. Документов: {len(docs)}, колонок: {n}, "
         f"уникальных заголовков: {df['header_text'].nunique()}.\n",
         "## Классы\n",
         _md(["Класс", "Наблюдений", "Доля"], [[k, v, _share(v, n)] for k, v in Counter(df["label"]).most_common()]),
         "\nКлассы из LABELS без наблюдений: " + (", ".join(sorted(set(LABELS) - set(df["label"]))) or "нет") + "\n",
         "## Длина заголовка, символов\n",
         f"среднее {L.mean():.1f}, медиана {L.median():.1f}, мин {L.min()}, макс {L.max()}\n",
         "## Пропуски значений\n"]
    miss = []
    for lab, g in df.groupby("label"):
        cells = pd.to_numeric(g["n_rows"], errors="coerce").sum()
        vals = pd.to_numeric(g["n_values"], errors="coerce").sum()
        miss.append([lab, int(cells), int(cells - vals), _share(cells - vals, cells)])
    P += [_md(["Класс", "Ячеек", "Пустых", "Доля пустых"], miss), "\n",
          "## Дубликаты шаблонов\n"]
    sig = df.groupby("doc_ref")["header_text"].apply(lambda s: "|".join(s))
    sc = Counter(sig)
    P += [f"Различных наборов заголовков (шаблонов) среди документов: {len(sc)}; "
          f"самый частый используют {max(sc.values())} из {len(sig)} документов.\n"]
    dup = df.duplicated(["header_text", "values_sample", "label"]).sum()
    P += [f"Полностью одинаковых (заголовок, значения, класс) колонок: {dup} из {n} ({_share(dup, n)}). "
          "Из-за этого без мутаций модель просто запоминает заголовки.\n", "## Временная структура\n"]
    dd = docs[docs["doc_date"] != ""]
    if len(dd):
        P += [f"Период: {dd['doc_date'].min()} - {dd['doc_date'].max()}.\n",
              _md(["Месяц", "Документов"], sorted(Counter(dd["doc_date"].str[:7]).items())), "\n"]
    else:
        P += ["Даты документов в данных отсутствуют.\n"]
    P += ["## Разбиение (из ЛР1)\n",
          _md(["Часть", "Документов", "Колонок", "Кластеров"],
              [[s, int((docs["split"] == s).sum()), int((df["split"] == s).sum()),
                docs.loc[docs["split"] == s, "cluster_id"].nunique()] for s in ("train", "val", "test")]),
          "\n## Проверка утечек\n",
          _md(["Проверка", "Результат", "Ожидание"],
              [["контракты в нескольких частях", f"{leaks['contract_id']['in_several_splits']} из {leaks['contract_id']['values']}", "0"],
               ["продавцы в нескольких частях", f"{leaks['seller_id']['in_several_splits']} из {leaks['seller_id']['values']}", "0"],
               ["покупатели в нескольких частях", f"{leaks['buyer_id']['in_several_splits']} из {leaks['buyer_id']['values']}", "0"],
               ["кластеры в нескольких частях", f"{leaks['cluster_id']['in_several_splits']} из {leaks['cluster_id']['values']}", "0"],
               [f"почти одинаковые пары (Jaccard ≥ {leaks['near_duplicate_pairs']['threshold']}) между частями",
                f"{leaks['near_duplicate_pairs']['across_splits']} (всего пар: {leaks['near_duplicate_pairs']['pairs']})", "0"]]),
          "\nJaccard считается по заголовкам и до 3 значений из values_sample, поэтому это приближение проверки из ЛР1 "
          "(там использовались все значения колонки).\n"]
    return "\n".join(P)


def write_passport(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    leaks = leak_report(df)
    (out_dir / "data_passport.md").write_text(passport(df, leaks), encoding="utf-8")
    (out_dir / "split.json").write_text(json.dumps(split_json(df, leaks), ensure_ascii=False, indent=1), encoding="utf-8")
    return leaks
