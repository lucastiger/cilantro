from pathlib import Path

from scoring.find_top_epitopes import find_top_epitopes


def test_find_top_epitopes_ignores_short_sequences(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("scoring.find_top_epitopes.mhcflurry_supported_alleles", lambda: [])
    monkeypatch.setattr("scoring.find_top_epitopes.parse_allele_frequencies_env", lambda: {})
    fasta_path = tmp_path / "mixed.fasta"
    fasta_path.write_text(
        ">short1\nAA\n"
        ">short2\nM\n"
        ">long1\nACDEFGHIK\n"
        ">long2\nACDEFGHIK\n"
    )

    result = find_top_epitopes(str(fasta_path), min_len=5, max_len=5, top_k=3)

    assert result
    assert all(entry["length"] == 5 for entry in result)


def test_find_top_epitopes_returns_empty_when_all_too_short(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("scoring.find_top_epitopes.mhcflurry_supported_alleles", lambda: [])
    monkeypatch.setattr("scoring.find_top_epitopes.parse_allele_frequencies_env", lambda: {})
    fasta_path = tmp_path / "short_only.fasta"
    fasta_path.write_text(">a\nAA\n>b\nAAA\n")

    result = find_top_epitopes(str(fasta_path), min_len=5, max_len=7, top_k=5)

    assert result == []


def test_find_top_epitopes_not_limited_by_shortest_sequence(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("scoring.find_top_epitopes.mhcflurry_supported_alleles", lambda: [])
    monkeypatch.setattr("scoring.find_top_epitopes.parse_allele_frequencies_env", lambda: {})

    fasta_path = tmp_path / "mixed_lengths.fasta"
    fasta_path.write_text(
        ">short5\nACDEF\n"
        ">long9a\nACDEFGHIK\n"
        ">long9b\nACDEYGHIK\n"
    )

    result = find_top_epitopes(str(fasta_path), min_len=5, max_len=7, top_k=100)

    lengths = {entry["length"] for entry in result}
    assert {5, 6, 7}.issubset(lengths)
