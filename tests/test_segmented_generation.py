from __future__ import annotations

import generate


def test_select_target_epitopes_honors_top_n():
    epitope_scores = {"B": 0.5, "A": 0.9, "C": 0.2}

    assert generate._select_target_epitopes(epitope_scores, top_n_epitopes=2) == ["A", "B"]


def test_optimize_per_epitope_runs_once_per_epitope(monkeypatch):
    calls: list[str] = []

    def fake_latent_optimize(**kwargs):
        calls.append(kwargs["target_epitope"])
        ep = kwargs["target_epitope"]
        return {
            "sequence": f"SEQ_{ep}",
            "score": float(len(ep)),
            "generation": 0,
            "score_breakdown": {"base_score": float(len(ep))},
        }

    monkeypatch.setattr(generate, "latent_optimize", fake_latent_optimize)

    result = generate._optimize_per_epitope(
        model=object(),
        seed_latent=[0.0, 1.0],
        epitopes=["AAAA", "BBB"],
        sigma=0.5,
        popsize=2,
        generations=3,
        progress_reporter=None,
    )

    assert calls == ["AAAA", "BBB"]
    assert result["target_epitope"] == "AAAA"
    assert result["sequence"] == "SEQ_AAAA"
    assert len(result["per_epitope_results"]) == 2
