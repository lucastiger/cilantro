from __future__ import annotations

import scoring.antigen_scoring as antigen_scoring


def test_normalize_esm2_log_likelihood_directionality():
    assert antigen_scoring.normalize_esm2_log_likelihood(-4.0) == 0.0
    assert antigen_scoring.normalize_esm2_log_likelihood(-0.5) == 1.0

    mid = antigen_scoring.normalize_esm2_log_likelihood(-2.25)
    assert 0.49 < mid < 0.51


def test_score_antigen_candidate_uses_esm2_not_folding_or_plddt(monkeypatch):
    monkeypatch.setattr(antigen_scoring, "predict_immunogenicity", lambda seq: 0.8)
    monkeypatch.setattr(antigen_scoring, "predict_esm2_sequence_log_likelihood", lambda seq: -1.0)
    monkeypatch.setattr(antigen_scoring, "predict_toxicity", lambda seq: 0.05)

    score = antigen_scoring.score_antigen_candidate("MKTAYIAKQ")
    assert score > 0.0


def test_score_antigen_candidate_ignores_target_epitopes(monkeypatch):
    monkeypatch.setattr(antigen_scoring, "predict_immunogenicity", lambda seq: 1.0)
    monkeypatch.setattr(antigen_scoring, "predict_esm2_sequence_log_likelihood", lambda seq: -0.5)
    monkeypatch.setattr(antigen_scoring, "predict_toxicity", lambda seq: 0.0)

    score_without_targets = antigen_scoring.score_antigen_candidate("AAAAAAAAAA")
    score_with_targets = antigen_scoring.score_antigen_candidate("AAAAAAAAAA", target_epitopes={"SIINFEKL"})
    assert score_with_targets == score_without_targets


def test_soft_epitope_reward_prefers_better_window_match():
    weak_match = antigen_scoring.soft_epitope_reward("AAAAAAAAAA", {"SIINFEKL"})
    strong_match = antigen_scoring.soft_epitope_reward("SIINFEKLAA", {"SIINFEKL"})
    assert strong_match > weak_match
    assert 0.0 <= weak_match <= 1.0
    assert 0.0 <= strong_match <= 1.0


def test_toxicity_softmax_score_is_normalized():
    score = antigen_scoring.toxicity_softmax_score([0.0, 1.0], beta=10.0)
    assert 0.9 < score < 1.0


def test_predict_toxicity_uses_sliding_windows(monkeypatch):
    captured = {}

    def fake_batch(windows):
        captured["windows"] = windows
        return [0.2, 0.8]

    monkeypatch.setattr(antigen_scoring, "predict_toxicity_external_batch", fake_batch)

    score = antigen_scoring.predict_toxicity("ABCDEFGHIJKLMNOP", window_size=15, beta=10.0)

    assert captured["windows"] == ["ABCDEFGHIJKLMNO", "BCDEFGHIJKLMNOP"]
    assert 0.6 < score < 0.85


def test_predict_toxicity_reuses_cached_windows(monkeypatch):
    calls = {"count": 0}

    def fake_batch(windows):
        calls["count"] += 1
        return [0.4 for _ in windows]

    antigen_scoring._TOXICITY_WINDOW_CACHE.clear()
    monkeypatch.setattr(antigen_scoring, "predict_toxicity_external_batch", fake_batch)

    first = antigen_scoring.predict_toxicity("ABCDEFGHIJKLMNOP", window_size=15, beta=10.0)
    second = antigen_scoring.predict_toxicity("ABCDEFGHIJKLMNOP", window_size=15, beta=10.0)

    assert first == second
    assert calls["count"] == 1


def test_antigen_score_components_match_antigen_score():
    components = antigen_scoring.antigen_score_components(
        immunogenicity=0.8428,
        esm2_mean_log_likelihood=-0.3647,
        toxicity=0.0295,
    )

    base = antigen_scoring.antigen_score(0.8428, -0.3647, 0.0295)
    assert abs(base - components["base_score"]) < 1e-12


def test_score_breakdown_includes_weighted_base_and_total(monkeypatch):
    monkeypatch.setattr(antigen_scoring, "predict_immunogenicity", lambda seq: 0.8428)
    monkeypatch.setattr(antigen_scoring, "predict_esm2_sequence_log_likelihood", lambda seq: -0.3647)
    monkeypatch.setattr(antigen_scoring, "predict_toxicity", lambda seq: 0.0295)
    breakdown = antigen_scoring.score_antigen_candidate_with_breakdown(
        "MLSELVEGEELLLLKLLLLLLGAAGVLIVLAGGGKVPLKQLSELLLGKDLDLLLGLDYLYLLVDKKRLRGTLLLLLDAALLLLDDGEISSVATALLAAKTLVLLDGGGVIGVKVVA",
        target_epitopes={"SIINFEKL"},
    )

    expected = (
        antigen_scoring.BASE_SCORE_WEIGHT * breakdown["base_score"]
        - breakdown["weighted_hydrophobicity_penalty"]
    )
    assert abs(expected - breakdown["score"]) < 1e-12


def test_hydrophobicity_penalty_hits_poly_leucine_sequences():
    poly_leucine = "L" * 120
    mostly_polar = "STNQDEKRHG" * 12

    assert antigen_scoring.hydrophobic_fraction(poly_leucine) > 0.95
    assert antigen_scoring.hydrophobicity_penalty(poly_leucine, max_fraction=0.45) > 0.9

    assert antigen_scoring.hydrophobic_fraction(mostly_polar) < 0.1
    assert antigen_scoring.hydrophobicity_penalty(mostly_polar, max_fraction=0.45) == 0.0


def test_score_penalizes_hydrophobic_sequences(monkeypatch):
    monkeypatch.setattr(antigen_scoring, "predict_immunogenicity", lambda seq: 0.9)
    monkeypatch.setattr(antigen_scoring, "predict_esm2_sequence_log_likelihood", lambda seq: -0.8)
    monkeypatch.setattr(antigen_scoring, "predict_toxicity", lambda seq: 0.05)

    hydrophobic = antigen_scoring.score_antigen_candidate_with_breakdown("L" * 140)
    balanced = antigen_scoring.score_antigen_candidate_with_breakdown("STNQDEKRHG" * 14)

    assert hydrophobic["weighted_hydrophobicity_penalty"] > 0.0
    assert balanced["weighted_hydrophobicity_penalty"] == 0.0
    assert balanced["score"] > hydrophobic["score"]
