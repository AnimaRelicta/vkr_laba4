"""Единый список классов.

В отчёте ЛР1 расхождение: в таблице классов 14 строк, в метриках zero-shot 16
(добавлены estimate_position и unit). Для ЛР4 берём 16 классов из LABELS в
run_llm_benchmark.py: это тот же набор, который видели LLM, и он нужен для каскада.
Классы без объектов в gold не входят в Macro-F1 (так же считает f1_macro в ЛР1).
"""

LABELS = [
    "item_code", "row_number", "estimate_position", "item_name", "unit", "unit_code", "unit_name",
    "quantity", "price", "amount_without_vat", "excise", "vat_rate", "vat_amount", "total_with_vat",
    "country_of_origin", "unmapped",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
N_CLASSES = len(LABELS)
