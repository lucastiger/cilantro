from pathlib import Path

from scoring.find_top_epitopes import find_top_epitopes


def test_find_top_epitopes_predicts_only_weighted_alleles(monkeypatch, tmp_path: Path):
    fasta_path = tmp_path / "single.fasta"
    fasta_path.write_text(">seq1\nACDEFGHIKLMNPQ\n", encoding="utf-8")

    calls = {}

    monkeypatch.setattr(
        "scoring.find_top_epitopes.mhcflurry_supported_alleles",
        lambda: ["HLA-A*01:01", "HLA-A*02:01", "HLA-B*07:02"],
    )
    monkeypatch.setattr(
        "scoring.find_top_epitopes.parse_allele_frequencies_env",
        lambda: {"HLA-A*02:01": 0.8},
    )

    def fake_predict(peptides, alleles):
        calls["alleles"] = list(alleles)
        peptide_list = list(peptides)
        return {alleles[0]: {pep: 50.0 for pep in peptide_list}}

    monkeypatch.setattr("scoring.find_top_epitopes.predict_mhcflurry_kd_matrix", fake_predict)

    result = find_top_epitopes(str(fasta_path), min_len=5, max_len=6, top_k=3)

    assert result
    assert calls["alleles"] == ["HLA-A*02:01"]


def test_find_top_epitopes_batches_prediction_once(monkeypatch, tmp_path: Path):
    fasta_path = tmp_path / "single.fasta"
    fasta_path.write_text(">seq1\nACDEFGHIKLMNPQ\n", encoding="utf-8")

    call_counter = {"count": 0}

    monkeypatch.setattr(
        "scoring.find_top_epitopes.mhcflurry_supported_alleles",
        lambda: ["HLA-A*01:01"],
    )
    monkeypatch.setattr(
        "scoring.find_top_epitopes.parse_allele_frequencies_env",
        lambda: {"HLA-A*01:01": 1.0},
    )

    def fake_predict(peptides, alleles):
        call_counter["count"] += 1
        peptide_list = list(peptides)
        return {"HLA-A*01:01": {pep: 50.0 for pep in peptide_list}}

    monkeypatch.setattr("scoring.find_top_epitopes.predict_mhcflurry_kd_matrix", fake_predict)

    find_top_epitopes(str(fasta_path), min_len=5, max_len=7, top_k=5)

    assert call_counter["count"] == 1


def test_find_top_epitopes_window_lengths_not_limited_by_short_outliers(monkeypatch, tmp_path: Path):
    fasta_path = tmp_path / "mixed_lengths.fasta"
    fasta_path.write_text(
        ">long1\nACDEFGHIKLMNPQRST\n>long2\nACDEFGHIKLMNPQRST\n>short\nACDEF\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scoring.find_top_epitopes.mhcflurry_supported_alleles",
        lambda: [],
    )
    monkeypatch.setattr(
        "scoring.find_top_epitopes.parse_allele_frequencies_env",
        lambda: {},
    )

    result = find_top_epitopes(str(fasta_path), min_len=5, max_len=15, top_k=300)

    returned_lengths = {entry["length"] for entry in result}
    assert 5 in returned_lengths
    assert 15 in returned_lengths
