"""Синтетический аналог out/observations.csv из ЛР1: ТОЛЬКО для проверки конвейера.

Повторяет структуру из отчёта ЛР1 (116 документов, 17 колонок на документ, 7 кластеров, split 80/18/18,
февраль-октябрь 2026), но заголовки и значения придуманы. Результаты на этих данных в отчёт не идут.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

# 17 колонок документа: по одной на 12 классов + 2 страны + 3 unmapped
COLUMNS = [
    ("row_number", "№ п/п", "1 | 2"),
    ("item_code", "код товара/ работы, услуги", "01.11.11 | 10.20.14"),
    ("item_name", "наименование товара (описание выполненных работ, оказанных услуг), имущественного права", "Молоко | Крупа"),
    ("unmapped", "дополнительные данные", ""),
    ("unit_code", "единица измерения / код", "796 | 166"),
    ("unit_name", "единица измерения / условное обозначение (национальное)", "шт | кг"),
    ("quantity", "количество (объем)", "10 | 25"),
    ("price", "цена (тариф) за единицу измерения", "120.50 | 80.00"),
    ("amount_without_vat", "стоимость товаров (работ, услуг), имущественных прав без налога - всего", "1205.00 | 2000.00"),
    ("excise", "в том числе сумма акциза", "без акциза"),
    ("vat_rate", "налоговая ставка", "20% | 10%"),
    ("vat_amount", "сумма налога, предъявляемая покупателю", "241.00 | 200.00"),
    ("total_with_vat", "стоимость товаров (работ, услуг), имущественных прав с налогом - всего", "1446.00 | 2200.00"),
    ("country_of_origin", "страна происхождения товара / цифровой код", "643 | 156"),
    ("country_of_origin", "страна происхождения товара / краткое наименование", "россия | китай"),
    ("unmapped", "номер таможенной декларации", ""),
    ("unmapped", "примечание", "-"),
]
CLUSTER_SIZES = [44, 20, 16, 18, 10, 6, 2]   # 7 кластеров, 116 документов
CLUSTER_SPLIT = ["train", "train", "train", "val", "test", "test", "test"]   # 80 / 18 / 18


def make_demo(out_dir: str | Path, seed: int = 42) -> Path:
    rnd = random.Random(seed)
    rows = []
    doc_no = 0
    for cid, (size, split) in enumerate(zip(CLUSTER_SIZES, CLUSTER_SPLIT)):
        for _ in range(size):
            doc_no += 1
            month = rnd.randint(2, 10)
            date = f"2026-{month:02d}-{rnd.randint(1, 28):02d}"
            contract = f"c_{cid}_{rnd.randint(0, 3)}"   # контракты не пересекают кластеры, значит и части
            for k, (lab, header, vals) in enumerate(COLUMNS, start=1):
                vals_c = f"{vals} | k{cid}" if vals else ""   # значения зависят от кластера, как у реальных шаблонов
                rows.append({
                    "obs_id": len(rows), "split": split, "doc_ref": f"d_{doc_no:04d}", "cluster_id": cid,
                    "form_type": "УПД", "template_id": "demo", "col_label": k, "header_raw": header, "header_text": header,
                    "values_sample": vals_c, "n_rows": 2, "n_values": 2 if vals else 0, "n_numeric": 2 if vals and vals[0].isdigit() else 0,
                    "label": lab, "header_artifact": 0, "contract_id": contract, "seller_id": f"s_{cid}", "buyer_id": f"b_{cid}",
                    "doc_date": date, "year_month": date[:7]})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "observations.csv", index=False, encoding="utf-8-sig")
    return out
