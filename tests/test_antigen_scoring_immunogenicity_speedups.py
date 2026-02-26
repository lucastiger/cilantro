from __future__ import annotations

import scoring.antigen_scoring as antigen_scoring


def test_predict_immunogenicity_deduplicates_peptides(monkeypatch):
    captured: dict[str, object] = {}

    def fake_predict(peptides, alleles):
        captured["peptides"] = list(peptides)
        captured["alleles"] = list(alleles)
        return {"HLA-A*02:01": {pep: 50.0 for pep in peptides}}

    monkeypatch.setattr(antigen_scoring, "mhcflurry_supported_alleles", lambda: ["HLA-A*02:01"])
    monkeypatch.setattr(antigen_scoring, "predict_mhcflurry_kd_matrix", fake_predict)
    monkeypatch.setattr(antigen_scoring, "parse_allele_frequencies_env", lambda: {})
    monkeypatch.setattr(
        antigen_scoring,
        "default_allele_frequencies",
        lambda alleles: {"HLA-A*02:01": 1.0},
    )

    antigen_scoring.predict_immunogenicity.cache_clear()
    score = antigen_scoring.predict_immunogenicity("AAAAAAAAAAAAAAAA")

    assert score > 0.0
    assert captured["peptides"] == ["AAAAAAAA", "AAAAAAAAA", "AAAAAAAAAA", "AAAAAAAAAAA"]


def test_predict_immunogenicity_only_predicts_for_frequency_backed_alleles(monkeypatch):
    captured: dict[str, object] = {}

    def fake_predict(peptides, alleles):
        captured["alleles"] = list(alleles)
        return {"HLA-A*02:01": {pep: 75.0 for pep in peptides}}

    monkeypatch.setattr(
        antigen_scoring,
        "mhcflurry_supported_alleles",
        lambda: ["HLA-A*02:01", "HLA-B*07:02", "HLA-C*07:02"],
    )
    monkeypatch.setattr(antigen_scoring, "predict_mhcflurry_kd_matrix", fake_predict)
    monkeypatch.setattr(
        antigen_scoring,
        "parse_allele_frequencies_env",
        lambda: {"HLA-A*02:01": 0.3, "HLA-Z*99:99": 0.7},
    )
    monkeypatch.setattr(
        antigen_scoring,
        "default_allele_frequencies",
        lambda alleles: {"HLA-A*02:01": 1.0, "HLA-B*07:02": 1.0},
    )

    antigen_scoring.predict_immunogenicity.cache_clear()
    antigen_scoring.predict_immunogenicity("ACDEFGHIKLMNPQRSTVWY")

    assert captured["alleles"] == ["HLA-A*02:01"]


def test_predict_immunogenicity_uses_cache(monkeypatch):
    call_count = {"predict": 0}

    def fake_predict(peptides, alleles):
        call_count["predict"] += 1
        return {"HLA-A*02:01": {pep: 100.0 for pep in peptides}}

    monkeypatch.setattr(antigen_scoring, "mhcflurry_supported_alleles", lambda: ["HLA-A*02:01"])
    monkeypatch.setattr(antigen_scoring, "predict_mhcflurry_kd_matrix", fake_predict)
    monkeypatch.setattr(antigen_scoring, "parse_allele_frequencies_env", lambda: {})
    monkeypatch.setattr(
        antigen_scoring,
        "default_allele_frequencies",
        lambda alleles: {"HLA-A*02:01": 1.0},
    )

    antigen_scoring.predict_immunogenicity.cache_clear()
    first = antigen_scoring.predict_immunogenicity("MKTAYIAKQ")
    second = antigen_scoring.predict_immunogenicity("MKTAYIAKQ")

    assert first == second
    assert call_count["predict"] == 1

