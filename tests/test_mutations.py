import numpy as np
import pytest

from vkr.lab4.mutations import (HoldoutViolation, build_set, m1_typos, m5_two_level, markov_chain, markov_frames,
                                mutate_frame, synonym_pool)


def tags(df):
    return set("+".join(df["mut"]).replace("+extra", "").split("+")) - {""}


def test_holdout_families_never_reach_train_or_val(cfg, splits):
    mc = cfg.mutations
    for fam, split in [("M3", "train"), ("M5", "train"), ("M6", "train"), ("M5", "val"), ("M6", "val")]:
        with pytest.raises(HoldoutViolation):
            mutate_frame(splits[split], {fam: {"p": 1.0, "p_doc": 1.0}}, split, mc, np.random.default_rng(0))


def test_train_and_val_sets_have_only_visible_families(cfg, splits):
    mc, sc = cfg.mutations, cfg.scenarios
    tr = build_set(splits["train"], sc["train_mutations"], "train", mc, 1000, 4, True)
    va = build_set(splits["val"], sc["val_mutations"], "val", mc, 2000, 2, True)
    assert tags(tr) <= {"M1", "M2", "M4"}
    assert tags(va) <= {"M1", "M2", "M3", "M4"}
    assert (tr["copy"] == 0).sum() == len(splits["train"]) and (tr.loc[tr["copy"] == 0, "mut"] == "").all()


def test_test_synonyms_are_not_in_train_pool_and_val_is_subset(cfg):
    mc = cfg.mutations
    assert all(not v for v in synonym_pool(mc, "train").values())
    val, test = synonym_pool(mc, "val"), synonym_pool(mc, "test")
    assert all(set(val[k]) < set(test[k]) for k in val)


def test_all_scenarios_run_and_keep_labels(cfg, splits):
    t = splits["test"]
    for name, s in cfg.scenarios["scenarios"].items():
        if "mutations" not in s:
            continue
        m = build_set(t, s["mutations"], "test", cfg.mutations, 3000, 1, False)
        assert set(m["label"]) <= set(t["label"]) | {"unmapped"}
        assert len(m) >= len(t)
        if name == "S0":
            assert (m["header_text"].to_numpy() == t["header_text"].to_numpy()).all()


def test_mutations_are_deterministic_per_seed(cfg, splits):
    spec = cfg.scenarios["scenarios"]["MAIN"]["mutations"]
    a = build_set(splits["test"], spec, "test", cfg.mutations, 3000, 2, False)
    b = build_set(splits["test"], spec, "test", cfg.mutations, 3000, 2, False)
    c = build_set(splits["test"], spec, "test", cfg.mutations, 3001, 2, False)
    assert a.equals(b) and not a["header_text"].equals(c["header_text"])


def test_m1_typo_rate_changes_with_p():
    rng = np.random.default_rng(0)
    base = "количество товаров " * 5
    ops = ["replace", "delete", "swap"]
    low = np.mean([m1_typos(base, 0.03, ops, rng) != base for _ in range(200)])
    assert low > 0.5
    assert m1_typos(base, 0.0, ops, rng) == base


def test_m5_two_level():
    rng = np.random.default_rng(1)
    outs = {m5_two_level("стоимость без налога - всего", ["glue", "break"], rng) for _ in range(30)}
    assert outs == {"стоимость без / налога - всего", "налога - всего"}      # склейка и потеря верхнего уровня
    assert m5_two_level("единица измерения / код", ["glue"], rng) == "единица измерения код"
    assert m5_two_level("акциз", ["glue", "break"], rng) == "акциз"


def test_markov_chain_stay_probability():
    rng = np.random.default_rng(0)
    ch = markov_chain(20000, 0.8, ["T0", "T1", "T2"], "T0", rng)
    stay = np.mean([a == b for a, b in zip(ch, ch[1:])])
    assert abs(stay - 0.8) < 0.02


def test_markov_frames_count(cfg, splits):
    frames = list(markov_frames(splits["test"], cfg.scenarios["scenarios"]["S5"], cfg.mutations, 7))
    assert len(frames) == 100 and all(len(f) == len(splits["test"]) for f, _ in frames)
