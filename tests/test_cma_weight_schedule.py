import pytest

from optimize.cma_latent_search import interleave_generated_sequences_and_epitopes


def test_interleave_generated_sequences_and_epitopes_three_epitopes():
    generated_sequences = ["AAAA", "BBBB", "CCCC", "DDDD"]
    epitopes = ["E1", "E2", "E3"]

    assert (
        interleave_generated_sequences_and_epitopes(generated_sequences, epitopes)
        == "AAAAE1BBBBE2CCCCE3DDDD"
    )


def test_interleave_generated_sequences_and_epitopes_requires_segment_count_match():
    epitopes = ["X", "Y", "Z"]

    with pytest.raises(ValueError, match=r"len\(epitopes\) \+ 1"):
        interleave_generated_sequences_and_epitopes(["AB", "CD"], epitopes)


def test_interleave_generated_sequences_and_epitopes_empty_epitopes_is_concat():
    assert (
        interleave_generated_sequences_and_epitopes(["SEQ1", "SEQ2"], [])
        == "SEQ1SEQ2"
    )
