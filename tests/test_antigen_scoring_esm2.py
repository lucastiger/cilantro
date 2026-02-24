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


def test_score_antigen_candidate_hard_epitope_constraint(monkeypatch):
    monkeypatch.setattr(antigen_scoring, "predict_immunogenicity", lambda seq: 1.0)
    monkeypatch.setattr(antigen_scoring, "predict_esm2_sequence_log_likelihood", lambda seq: -0.5)
    monkeypatch.setattr(antigen_scoring, "predict_toxicity", lambda seq: 0.0)

    score = antigen_scoring.score_antigen_candidate("AAAAAAAAAA", target_epitopes={"SIINFEKL"})
    assert score == 0.0


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
