from __future__ import annotations

import numpy as np

from optimize.cma_latent_search import latent_optimize
from scoring.find_top_epitopes import find_top_epitopes


def test_find_top_epitopes_reports_progress(monkeypatch, tmp_path):
    fasta_path = tmp_path / "seqs.fasta"
    fasta_path.write_text(">a\nACDEFG\n>b\nACDEYG\n", encoding="utf-8")

    monkeypatch.setattr(
        "scoring.find_top_epitopes.mhcflurry_supported_alleles",
        lambda: ["HLA-A*01:01"],
    )
    monkeypatch.setattr(
        "scoring.find_top_epitopes.parse_allele_frequencies_env",
        lambda: {"HLA-A*01:01": 1.0},
    )
    monkeypatch.setattr(
        "scoring.find_top_epitopes.predict_mhcflurry_kd_matrix",
        lambda peptides, alleles: {"HLA-A*01:01": {pep: 50.0 for pep in peptides}},
    )

    events: list[tuple[str, dict]] = []

    find_top_epitopes(
        str(fasta_path),
        min_len=2,
        max_len=3,
        top_k=3,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    labels = [event for event, _ in events]
    assert labels[0] == "start"
    assert labels.count("window_scan") == 2
    assert labels.count("window_score") == 2
    assert labels[-1] == "complete"


def test_latent_optimize_reports_generation_progress(monkeypatch):
    class FakeStrategy:
        def __init__(self, seed, sigma, options):
            self.popsize = options["popsize"]

        def ask(self):
            return [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

        def tell(self, solutions, losses):
            return None


    class FakeModel:
        def decode(self, z):
            return np.array([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])

    monkeypatch.setattr("optimize.cma_latent_search.cma.CMAEvolutionStrategy", FakeStrategy)
    monkeypatch.setattr("optimize.cma_latent_search.decode_sequence_from_ids", lambda _: "AAA")

    scores = iter([0.1, 0.4, 0.2, 0.3])
    next_base_scores = [0.1, 0.4, 0.2, 0.3]
    monkeypatch.setattr(
        "optimize.cma_latent_search.score_antigen_candidate_with_breakdown",
        lambda seq: {
            "score": next(scores),
            "immunogenicity": 0.8,
            "esm2_sequence_log_likelihood": -1.2,
            "toxicity": 0.1,
            "base_score": next_base_scores.pop(0),
        },
    )

    events: list[tuple[str, dict]] = []

    best = latent_optimize(
        model=FakeModel(),
        seed_latent=np.array([0.0, 0.0]),
        target_epitopes=["AAA", "BBB", "CCC"],
        popsize=2,
        generations=2,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    labels = [event for event, _ in events]
    assert labels[0] == "start"
    assert labels.count("candidate_evaluated") == 4
    assert labels.count("generation_complete") == 2
    generation_events = [payload for event, payload in events if event == "generation_complete"]
    new_best_events = [payload for event, payload in events if event == "new_best_candidate"]
    assert generation_events[0]["top_candidates"] == [
        {"sequence": "AAAAAAAAABBBAAACCCAAA", "score": 0.4},
        {"sequence": "AAAAAAAAABBBAAACCCAAA", "score": 0.1},
    ]
    assert generation_events[1]["top_candidates"] == [
        {"sequence": "AAAAAAAAABBBAAACCCAAA", "score": 0.3},
        {"sequence": "AAAAAAAAABBBAAACCCAAA", "score": 0.2},
    ]
    assert len(new_best_events) == 2
    assert new_best_events[1]["score"] == 0.4
    assert new_best_events[1]["sequence"] == "AAAAAAAAABBBAAACCCAAA"
    assert labels[-1] == "complete"
    assert best["score"] == 0.4


def test_latent_optimize_uses_total_score_not_base_score(monkeypatch):
    class FakeStrategy:
        def __init__(self, seed, sigma, options):
            self.popsize = options["popsize"]

        def ask(self):
            return [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

        def tell(self, solutions, losses):
            return None

    class FakeModel:
        def decode(self, z):
            return np.array([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])

    monkeypatch.setattr("optimize.cma_latent_search.cma.CMAEvolutionStrategy", FakeStrategy)
    monkeypatch.setattr("optimize.cma_latent_search.decode_sequence_from_ids", lambda _: "AAA")

    breakdowns = iter(
        [
            {
                "score": 0.30,
                "immunogenicity": 0.8,
                "esm2_sequence_log_likelihood": -1.2,
                "toxicity": 0.1,
                "base_score": 0.90,
                "hydrophobicity_penalty": 1.0,
                "weighted_hydrophobicity_penalty": 0.60,
            },
            {
                "score": 0.40,
                "immunogenicity": 0.8,
                "esm2_sequence_log_likelihood": -1.2,
                "toxicity": 0.1,
                "base_score": 0.50,
                "hydrophobicity_penalty": 0.0,
                "weighted_hydrophobicity_penalty": 0.0,
            },
        ]
    )
    monkeypatch.setattr(
        "optimize.cma_latent_search.score_antigen_candidate_with_breakdown",
        lambda seq: next(breakdowns),
    )

    best = latent_optimize(
        model=FakeModel(),
        seed_latent=np.array([0.0, 0.0]),
        target_epitopes=["AAA"],
        popsize=2,
        generations=1,
        progress_callback=None,
    )

    assert best["score"] == 0.40
