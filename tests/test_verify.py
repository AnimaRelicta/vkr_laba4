import json

from vkr.lab4.demo_data import make_demo
from vkr.lab4.verify import build_check, header_counts, load_dir


def test_header_counts_definitions():
    rows = [
        {"header_raw": "Цена ", "header_text": "Цена", "label": "price", "values_sample": "1"},
        {"header_raw": "цена", "header_text": "цена", "label": "price", "values_sample": "2"},
        {"header_raw": "Прим.", "header_text": "Прим.", "label": "unmapped", "values_sample": ""},
    ]
    c = header_counts(rows)
    assert c["строк (наблюдений)"] == 3
    assert c["уникальных header_text"] == 3
    assert c["уникальных header_text (регистр и пробелы не важны)"] == 2
    assert c["строк без unmapped"] == 2


def test_build_check_on_demo(tmp_path):
    d = make_demo(tmp_path / "out")
    n_test = len(load_dir(d)["splits"]["test"])
    llm = tmp_path / "llm"
    (llm / "preds").mkdir(parents=True)
    (llm / "preds" / "test__m__k3__v0__direct.jsonl").write_text(
        "\n".join(json.dumps({"gold": g, "pred": g}) for g in ("price", "unit", "price")), encoding="utf-8")
    md, summ = build_check([d], [llm], [n_test, 276], [3, 16])
    assert "Сверка чисел ЛР1" in md
    assert any("test" in w for w in summ["matches"][n_test])
    # лог LLM: 2 уникальных gold -> число 2 не заявлено, но N=3 запросов совпадает с заявленным 3
    assert any("n = 3" in w for w in summ["matches"][3])
