from __future__ import annotations

import generate


def test_select_target_epitopes_honors_top_n():
    epitope_scores = {"B": 0.5, "A": 0.9, "C": 0.2}

    assert generate._select_target_epitopes(epitope_scores, top_n_epitopes=2) == ["A", "B"]


def test_select_antigen_epitopes_limits_to_three():
    epitopes = ["A", "B", "C", "D"]

    assert generate._select_antigen_epitopes(epitopes, target_count=3) == ["A", "B", "C"]


def test_optimize_for_epitope_set_single_run(monkeypatch):
    calls: list[list[str]] = []

    def fake_latent_optimize(**kwargs):
        calls.append(kwargs["target_epitopes"])
        return {
            "sequence": "SEQ_ABC",
            "score": 1.23,
            "generation": 0,
            "score_breakdown": {"base_score": 1.23},
        }

    monkeypatch.setattr(generate, "latent_optimize", fake_latent_optimize)

    result = generate._optimize_for_epitope_set(
        model=object(),
        seed_latent=[0.0, 1.0],
        epitopes=["AAAA", "BBB", "CC"],
        sigma=0.5,
        popsize=2,
        generations=3,
        progress_reporter=None,
    )

    assert calls == [["AAAA", "BBB", "CC"]]
    assert result["target_epitopes"] == ["AAAA", "BBB", "CC"]
    assert result["sequence"] == "SEQ_ABC"
