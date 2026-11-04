from vkr.lab4.data import leak_report, passport, split_json
from vkr.lab4.labels import LABELS


def test_demo_shape_matches_lab1_structure(demo_df):
    assert demo_df["doc_ref"].nunique() == 116
    assert demo_df["cluster_id"].nunique() == 7
    assert demo_df["header_text"].nunique() == 17
    docs = demo_df.drop_duplicates("doc_ref")["split"].value_counts().to_dict()
    assert docs == {"train": 80, "val": 18, "test": 18}


def test_labels_are_known(demo_df):
    assert set(demo_df["label"]) <= set(LABELS)


def test_no_leaks_in_demo(demo_df):
    rep = leak_report(demo_df)
    assert rep["clean"], rep


def test_leak_detected_when_cluster_split(demo_df):
    bad = demo_df.copy()
    first_doc = bad[bad["cluster_id"] == "0"]["doc_ref"].iloc[0]
    bad.loc[bad["doc_ref"] == first_doc, "split"] = "test"
    assert not leak_report(bad)["clean"]


def test_passport_and_split_json(demo_df):
    rep = leak_report(demo_df)
    text = passport(demo_df, rep)
    assert "Паспорт данных" in text and "unmapped" in text
    sj = split_json(demo_df, rep)
    assert sj["parts"]["train"]["documents"] == 80
