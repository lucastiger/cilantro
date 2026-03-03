import pytest

from optimize.cma_latent_search import interleave_generated_sequences_and_epitopes


def test_interleave_generated_sequences_and_epitopes_twelve_epitopes():
    seq_a = "A" * 130
    seq_b = "B" * 130
    epitopes = [f"E{i}" for i in range(1, 13)]

    assembled = interleave_generated_sequences_and_epitopes([seq_a, seq_b], epitopes)

    assert assembled.count("AAY") == 6
    assert "E1AAYE2" in assembled
    assert "E11AAYE12" in assembled


def test_interleave_generated_sequences_and_epitopes_requires_twelve_epitopes():
    with pytest.raises(ValueError, match="exactly 12"):
        interleave_generated_sequences_and_epitopes(["A" * 130, "B" * 130], ["E1", "E2"])


def test_interleave_generated_sequences_and_epitopes_requires_two_generated_sequences():
    epitopes = [f"E{i}" for i in range(1, 13)]

    with pytest.raises(ValueError, match="exactly 2"):
        interleave_generated_sequences_and_epitopes(["AB"], epitopes)


def test_interleave_generated_sequences_and_epitopes_empty_epitopes_is_concat():
    assert (
        interleave_generated_sequences_and_epitopes(["SEQ1", "SEQ2"], [])
        == "SEQ1SEQ2"
    )


def test_latent_optimize_generates_two_sequences_per_candidate(monkeypatch):
    import numpy as np
    from optimize.cma_latent_search import latent_optimize

    class FakeStrategy:
        def __init__(self, seed, sigma, options):
            self.popsize = options["popsize"]

        def ask(self):
            return [np.array([0.1, 0.2, 0.3, 0.4])]

        def tell(self, solutions, losses):
            return None

    class FakeModel:
        def __init__(self):
            self.decode_calls = 0

        def decode(self, z):
            self.decode_calls += 1
            return np.array([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])

    model = FakeModel()
    monkeypatch.setattr("optimize.cma_latent_search.cma.CMAEvolutionStrategy", FakeStrategy)
    monkeypatch.setattr("optimize.cma_latent_search.decode_sequence_from_ids", lambda _: "AAA")
    monkeypatch.setattr(
        "optimize.cma_latent_search.score_antigen_candidate_with_breakdown",
        lambda _seq: {
            "score": 0.1,
            "immunogenicity": 0.8,
            "esm2_sequence_log_likelihood": -1.2,
            "toxicity": 0.1,
            "base_score": 0.1,
        },
    )

    best = latent_optimize(
        model=model,
        seed_latent=np.array([0.0, 0.0]),
        target_epitopes=[f"E{i}" for i in range(1, 13)],
        popsize=1,
        generations=1,
    )

    assert model.decode_calls == 2
    assert best["seqA"] == "AAA"
    assert best["seqB"] == "AAA"
