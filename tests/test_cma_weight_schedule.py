from optimize.cma_latent_search import insert_epitopes_with_generated_segments


def test_insert_epitopes_with_generated_segments_three_epitopes_even_split():
    sequence = "AAAABBBBCCCCDDDD"
    epitopes = ["E1", "E2", "E3"]

    assert (
        insert_epitopes_with_generated_segments(sequence, epitopes)
        == "AAAAE1BBBBE2CCCCE3DDDD"
    )


def test_insert_epitopes_with_generated_segments_handles_remainder():
    sequence = "ABCDEFG"
    epitopes = ["X", "Y", "Z"]

    assert insert_epitopes_with_generated_segments(sequence, epitopes) == "ABXCDYEFZG"


def test_insert_epitopes_with_generated_segments_empty_epitopes_is_noop():
    assert insert_epitopes_with_generated_segments("SEQUENCE", []) == "SEQUENCE"
