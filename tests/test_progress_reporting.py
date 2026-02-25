from __future__ import annotations

from types import SimpleNamespace

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

    class FakeArgmaxResult:
        def __init__(self, arr):
            self._arr = arr

        def numpy(self):
            return self._arr

    fake_tf = SimpleNamespace(
        constant=lambda x: x,
        argmax=lambda logits, axis=-1: FakeArgmaxResult(np.argmax(logits, axis=axis)),
    )

    class FakeModel:
        def decode(self, z):
            return np.array([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])

    monkeypatch.setattr("optimize.cma_latent_search.cma.CMAEvolutionStrategy", FakeStrategy)
    monkeypatch.setattr("optimize.cma_latent_search.tf", fake_tf)
    monkeypatch.setattr("optimize.cma_latent_search.decode_sequence_from_ids", lambda _: "AAA")

    scores = iter([0.1, 0.4, 0.2, 0.3])
    monkeypatch.setattr(
        "optimize.cma_latent_search.score_antigen_candidate",
        lambda seq, targets: next(scores),
    )

    events: list[tuple[str, dict]] = []

    best = latent_optimize(
        model=FakeModel(),
        seed_latent=np.array([0.0, 0.0]),
        target_epitopes={"AAA"},
        popsize=2,
        generations=2,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    labels = [event for event, _ in events]
    assert labels[0] == "start"
    assert labels.count("candidate_evaluated") == 4
    assert labels.count("generation_complete") == 2
    generation_events = [payload for event, payload in events if event == "generation_complete"]
    assert generation_events[0]["top_candidates"] == [
        {"sequence": "AAA", "score": 0.4},
        {"sequence": "AAA", "score": 0.1},
    ]
    assert generation_events[1]["top_candidates"] == [
        {"sequence": "AAA", "score": 0.3},
        {"sequence": "AAA", "score": 0.2},
    ]
    assert labels[-1] == "complete"
    assert best["score"] == 0.4
