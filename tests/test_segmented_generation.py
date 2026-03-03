from __future__ import annotations

import generate


def test_select_target_epitopes_honors_top_n():
    epitope_scores = {"B": 0.5, "A": 0.9, "C": 0.2}

    assert generate._select_target_epitopes(epitope_scores, top_n_epitopes=2) == ["A", "B"]


def test_select_antigen_epitopes_limits_to_twelve():
    epitopes = [f"E{i}" for i in range(1, 15)]

    assert generate._select_antigen_epitopes(epitopes, target_count=12) == [f"E{i}" for i in range(1, 13)]


def test_optimize_for_epitope_set_single_run(monkeypatch):
    calls: list[list[str]] = []

    def fake_latent_optimize(**kwargs):
        calls.append(kwargs["target_epitopes"])
        return {
            "sequence": "SEQ_ABC",
            "score": 1.23,
            "generation": 0,
            "score_breakdown": {"base_score": 1.23},
        }

    monkeypatch.setattr(generate, "latent_optimize", fake_latent_optimize)

    epitopes = [f"E{i}" for i in range(1, 13)]
    result = generate._optimize_for_epitope_set(
        model=object(),
        seed_latent=[0.0, 1.0],
        epitopes=epitopes,
        sigma=0.5,
        popsize=2,
        generations=3,
        progress_reporter=None,
    )

    assert calls == [epitopes]
    assert result["target_epitopes"] == epitopes
    assert result["sequence"] == "SEQ_ABC"


def test_parse_seed_epitopes_supports_spaces_and_commas():
    parsed = generate._parse_seed_epitopes(["AAA", "BBB,CCC", "DDD,,"])

    assert parsed == ["AAA", "BBB", "CCC", "DDD"]


def test_main_uses_user_provided_seed_epitopes(monkeypatch):
    calls = {"find_top_epitopes": 0, "latent_optimize": 0}

    class DummyModel:
        def __init__(self, weights_path):
            self.weights_path = weights_path

        def encode(self, tokenized):
            return [[0.1, 0.2]], None

    def fake_find_top_epitopes(*args, **kwargs):
        calls["find_top_epitopes"] += 1
        return []

    def fake_build_vocab_and_encode(seqs, max_len):
        return [[1, 2, 3]], None

    seed_panel = [f"E{i}" for i in range(1, 13)]

    def fake_latent_optimize(**kwargs):
        calls["latent_optimize"] += 1
        assert kwargs["target_epitopes"] == seed_panel
        return {"sequence": "SEQ", "score": 0.1, "generation": 0}

    monkeypatch.setattr(generate, "ProteinSeqVAE", DummyModel)
    monkeypatch.setattr(generate, "find_top_epitopes", fake_find_top_epitopes)
    monkeypatch.setattr(generate, "build_vocab_and_encode", fake_build_vocab_and_encode)
    monkeypatch.setattr(generate, "score_antigen_candidate_with_breakdown", lambda _seq: {"score": 0.1})
    monkeypatch.setattr(generate, "latent_optimize", fake_latent_optimize)

    args = type("Args", (), {
        "show_progress": False,
        "progress_per_candidate": False,
        "seed_sequence": "MKT",
        "seed_epitope_panel": seed_panel,
        "input_fasta": None,
        "max_len": 200,
        "ckpt": "checkpoint.pt",
        "min_ep_len": 5,
        "max_ep_len": 35,
        "top_k": 10,
        "top_n_epitopes": None,
        "sigma": 0.5,
        "popsize": 2,
        "generations": 1,
        "output_json": None,
    })

    generate.main(args)

    assert calls["find_top_epitopes"] == 0
    assert calls["latent_optimize"] == 1
