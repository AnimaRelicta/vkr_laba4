"""Признаки без torch: хеш символьных n-грамм заголовка и признаки значений (для компактной модели ngram)."""
from __future__ import annotations

import re
import zlib

_WS = re.compile(r"\s+")
_FRAC = re.compile(r"\d[.,]\d")


def normalize(text: str) -> str:
    """Для ngram-модели: регистр и пробелы нормализуются (как в ЛР1, где заголовки приводились к нижнему регистру)."""
    return _WS.sub(" ", text.casefold()).strip()


def ngram_ids(text: str, buckets: int, nmin: int = 2, nmax: int = 5) -> list[int]:
    """Хешированные символьные n-граммы (стабильный crc32, не зависит от PYTHONHASHSEED)."""
    t = f"<{normalize(text)}>"
    ids = []
    for n in range(nmin, nmax + 1):
        for i in range(len(t) - n + 1):
            ids.append(zlib.crc32(t[i:i + n].encode("utf-8")) % buckets)
    return ids or [0]


def value_features(values: str) -> list[float]:
    """Доля цифр, длина (нормирована, не больше 1), наличие дробной части."""
    s = values.replace(" | ", "").replace(" ", "")
    n = len(s)
    digits = sum(c.isdigit() for c in s)
    return [digits / n if n else 0.0, min(n / 100.0, 1.0), 1.0 if _FRAC.search(values) else 0.0]


def qwen_text(header: str, values: str) -> str:
    """Формат входа Qwen совпадает с промптом v0 из ЛР1."""
    return f"Заголовок: {header}\nЗначения: {values or '—'}"
