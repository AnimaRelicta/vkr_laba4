"""Мутации заголовков M1-M6 (ТЗ, п. 3.2) и сценарии S0-S5 (п. 4.2).

Главное правило: M3, M5, M6 нельзя применять к train, M5 и M6 нельзя применять к val.
Нарушение приводит к ValueError (hold-out семейства проверяют обобщение, а не память).
Все случайные решения идут из переданного np.random.Generator, поэтому результат воспроизводим по seed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

RU_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
EN_LOWER = "abcdefghijklmnopqrstuvwxyz"
FAMILY_KEYS = ["M1", "M2", "M3", "M4", "M5", "M6"]


class HoldoutViolation(ValueError):
    pass


def check_visibility(spec: dict, split: str, mut_cfg: dict) -> None:
    for fam in spec:
        if split not in mut_cfg["visibility"][fam]:
            raise HoldoutViolation(f"Семейство {fam} не должно применяться к {split} (visibility в mutations.yaml)")


# ---------------------------------------------------------------- отдельные семейства
def _alphabet(ch: str) -> str | None:
    lo = ch.lower()
    if lo in RU_LOWER:
        return RU_LOWER
    if lo in EN_LOWER:
        return EN_LOWER
    return None


def m1_typos(text: str, p_char: float, ops: list[str], rng: np.random.Generator) -> str:
    """Опечатки: замена, пропуск, перестановка соседних символов; p_char - вероятность на букву."""
    chars = list(text)
    i = 0
    while i < len(chars):
        ch = chars[i]
        alpha = _alphabet(ch)
        if alpha is not None and rng.random() < p_char:
            op = ops[rng.integers(len(ops))]
            if op == "replace":
                new = alpha[rng.integers(len(alpha))]
                chars[i] = new.upper() if ch.isupper() else new
            elif op == "delete" and len(chars) > 1:
                del chars[i]
                continue
            elif op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                i += 1
        i += 1
    return "".join(chars)


def m2_abbreviate(text: str, pairs: list, rng: np.random.Generator) -> str:
    """Сокращения: первая подходящая замена обязательна, остальные с p = 0.5."""
    out, first = text, True
    for sub, rep in pairs:
        pat = re.compile(re.escape(sub), re.I)
        if pat.search(out) and (first or rng.random() < 0.5):
            out, first = pat.sub(rep, out, count=1), False
    return out


def m3_synonym(label: str, pool: dict, rng: np.random.Generator) -> str | None:
    cands = pool.get(label) or []
    return cands[rng.integers(len(cands))] if cands else None


def m4_format(text: str, ops: list[str], rng: np.random.Generator) -> str:
    """Регистр, пунктуация, пробелы, переносы: применяются 1-2 случайные операции."""
    out = text
    for op in rng.choice(ops, size=int(rng.integers(1, 3)), replace=False):
        if op == "upper":
            out = out.upper()
        elif op == "lower":
            out = out.lower()
        elif op == "punct":
            new = re.sub(r"[()\-,./]", " ", out)
            out = new if new != out else out + ":"
        elif op == "spaces":
            out = " ".join(["", out, ""]) if rng.random() < 0.3 else re.sub(r" ", lambda m: "  " if rng.random() < 0.5 else " ", out)
        elif op == "newline":
            idx = [m.start() for m in re.finditer(" ", out)]
            if idx:
                k = idx[rng.integers(len(idx))]
                out = out[:k] + "\n" + out[k + 1:]
        elif op == "trailing":
            out = out + str(rng.choice([":", ".", ";"]))
    return out


def m5_two_level(text: str, modes: list[str], rng: np.random.Generator) -> str:
    """Двухуровневая шапка: склейка верхнего и нижнего уровня либо потеря верхнего уровня."""
    if " / " in text:
        up, low = text.split(" / ", 1)
        glued = f"{up} {low}"          # исходный разделитель теряется
    else:
        words = text.split()
        if len(words) < 2:
            return text
        k = len(words) // 2
        up, low = " ".join(words[:k]), " ".join(words[k:])
        glued = f"{up} / {low}"
    mode = modes[rng.integers(len(modes))]
    return glued if mode == "glue" else low


# ---------------------------------------------------------------- применение к таблице
def synonym_pool(mut_cfg: dict, split: str) -> dict:
    """Какие синонимы видит часть: train - никаких, val - val_syn, test - val_syn + test_only."""
    pool = {}
    for lab, d in mut_cfg["synonyms"].items():
        if split == "train":
            pool[lab] = []
        elif split == "val":
            pool[lab] = list(d["val_syn"])
        else:
            pool[lab] = list(d["val_syn"]) + list(d["test_only"])
    return pool


def mutate_frame(df: pd.DataFrame, spec: dict, split: str, mut_cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Одна мутированная копия df. spec: {семейство: параметры}. Добавляет header_orig, mut (теги применённых семейств)."""
    check_visibility(spec, split, mut_cfg)
    out = df.copy()
    out["header_orig"] = out["header_text"]
    tags = [[] for _ in range(len(out))]
    headers = out["header_text"].tolist()
    labels = out["label"].tolist()
    pool = synonym_pool(mut_cfg, split)
    for fam in mut_cfg["order"]:
        if fam not in spec or fam == "M6":
            continue
        prm = spec[fam]
        for i in range(len(headers)):
            if fam == "M1":
                new = m1_typos(headers[i], prm["p_char"], mut_cfg["M1"]["ops"], rng)
            elif fam == "M2":
                if rng.random() >= prm["p"]:
                    continue
                new = m2_abbreviate(headers[i], mut_cfg["M2"]["abbreviations"], rng)
            elif fam == "M3":
                if rng.random() >= prm["p"]:
                    continue
                new = m3_synonym(labels[i], pool, rng) or headers[i]
            elif fam == "M4":
                if rng.random() >= prm["p"]:
                    continue
                new = m4_format(headers[i], mut_cfg["M4"]["ops"], rng)
            elif fam == "M5":
                if rng.random() >= prm["p"]:
                    continue
                new = m5_two_level(headers[i], mut_cfg["M5"]["modes"], rng)
            else:
                raise KeyError(fam)
            if new != headers[i]:
                headers[i] = new
                tags[i].append(fam)
    out["header_text"] = headers
    out["mut"] = ["+".join(t) for t in tags]
    if "M6" in spec:
        out = m6_doc_level(out, spec["M6"]["p_doc"], mut_cfg, rng)
    return out


def m6_doc_level(df: pd.DataFrame, p_doc: float, mut_cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Перестановка колонок и лишняя колонка: с вероятностью p_doc на документ.

    Классификатор не использует позицию, поэтому перестановка меняет только col_label (важна для каскада и правил
    «точное совпадение»). Лишняя колонка добавляет объект класса unmapped."""
    extras = mut_cfg["M6"]["extra_columns"]
    parts, new_rows = [], []
    for doc, g in df.groupby("doc_ref", sort=False):
        g = g.copy()
        if rng.random() < p_doc:
            g["col_label"] = rng.permutation(g["col_label"].to_numpy())
            g["mut"] = g["mut"].map(lambda t: (t + "+M6").lstrip("+"))
            h, v = extras[rng.integers(len(extras))]
            row = g.iloc[-1].copy()
            row.update({"header_text": h, "header_orig": h, "values_sample": v, "label": "unmapped",
                        "col_label": int(g["col_label"].max()) + 1, "mut": "M6+extra"})
            new_rows.append(row)
        parts.append(g)
    res = pd.concat(parts + ([pd.DataFrame(new_rows)] if new_rows else []), ignore_index=True)
    return res.sort_values(["doc_ref", "col_label"], kind="stable").reset_index(drop=True)


def build_set(df: pd.DataFrame, spec: dict, split: str, mut_cfg: dict, seed: int, copies: int,
              keep_original: bool) -> pd.DataFrame:
    """copies копий df; при keep_original первая копия - без мутаций, остальные мутированы независимыми розыгрышами."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(copies):
        if keep_original and c == 0:
            part = df.copy()
            part["header_orig"], part["mut"] = part["header_text"], ""
        else:
            part = mutate_frame(df, spec, split, mut_cfg, rng)
        part["copy"] = c
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def scenario_test_sets(scenarios_cfg: dict, mut_cfg: dict, copies_test: int, test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Мутированные test-таблицы сценариев S0-S4 и MAIN (всё, кроме марковского S5). Единственное место, где они строятся:
    её используют и Context (обучение), и этап 7 (bootstrap по документам), порядок строк в обоих случаях одинаков."""
    out: dict[str, pd.DataFrame] = {}
    for name, s in scenarios_cfg["scenarios"].items():
        if "mutations" in s:
            copies = 1 if name == "S0" else copies_test
            out[name] = build_set(test_df, s["mutations"], "test", mut_cfg, mut_cfg["seeds"]["test"], copies,
                                  keep_original=(name == "S0"))
    return out


# ---------------------------------------------------------------- S5: марковское переключение версий
def markov_chain(n_docs: int, stay: float, versions: list[str], start: str, rng: np.random.Generator) -> list[str]:
    """Цепь версий шаблона по документам: остаёмся с вероятностью stay, иначе одна из остальных поровну."""
    state, out = start, []
    for _ in range(n_docs):
        out.append(state)
        if rng.random() >= stay:
            others = [v for v in versions if v != state]
            state = others[rng.integers(len(others))]
    return out


def markov_frames(test_df: pd.DataFrame, scen: dict, mut_cfg: dict, seed: int):
    """Генератор мутированных test-таблиц: по одной на розыгрыш цепи. Документы упорядочены по doc_date, затем по doc_ref."""
    m = scen["markov"]
    docs = (test_df.drop_duplicates("doc_ref").sort_values(["doc_date", "doc_ref"])["doc_ref"].tolist())
    versions = list(m["versions"])
    rng = np.random.default_rng(seed)
    for _ in range(m["draws"]):
        chain = markov_chain(len(docs), m["stay"], versions, m["start"], rng)
        by_version = dict(zip(docs, chain))
        parts = []
        for ver in versions:
            sub = test_df[test_df["doc_ref"].map(by_version) == ver]
            if len(sub):
                spec = m["versions"][ver]
                parts.append(mutate_frame(sub, spec, "test", mut_cfg, rng) if spec else
                             sub.assign(header_orig=sub["header_text"], mut=""))
        yield pd.concat(parts, ignore_index=True), chain
