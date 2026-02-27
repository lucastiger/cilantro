from optimize.cma_latent_search import insert_epitope_at_midpoint


def test_insert_epitope_at_midpoint_even_length_sequence():
    assert insert_epitope_at_midpoint("AAAABBBB", "EPI") == "AAAAEPIBBBB"


def test_insert_epitope_at_midpoint_odd_length_sequence():
    assert insert_epitope_at_midpoint("AAAAABB", "EPI") == "AAAEPIAABB"


def test_insert_epitope_at_midpoint_empty_epitope_is_noop():
    assert insert_epitope_at_midpoint("SEQUENCE", "") == "SEQUENCE"
