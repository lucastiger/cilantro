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
