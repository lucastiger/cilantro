from __future__ import annotations

import numpy as np

import generate


def test_optimize_segmented_antigen_concatenates_segments(monkeypatch):
    calls: list[np.ndarray] = []

    def fake_latent_optimize(**kwargs):
        calls.append(np.asarray(kwargs["seed_latent"]))
        idx = len(calls)
        return {
            "sequence": f"SEG{idx}",
            "score": float(idx),
            "generation": 0,
            "score_breakdown": {"score": float(idx)},
        }

    monkeypatch.setattr(generate, "latent_optimize", fake_latent_optimize)
    monkeypatch.setattr(
        generate,
        "score_antigen_candidate_with_breakdown",
        lambda seq, targets: {"score": 9.0, "sequence_length": len(seq)},
    )

    result = generate._optimize_segmented_antigen(
        model=object(),
        seed_latent=np.array([0.0, 1.0], dtype=np.float32),
        target_epitopes={"AA"},
        sigma=0.5,
        popsize=2,
        generations=3,
        segments=3,
        segment_linker="GGGGS",
        segment_seed_jitter=0.0,
        progress_reporter=None,
    )

    assert len(calls) == 3
    assert result["sequence"] == "SEG1GGGGSSEG2GGGGSSEG3"
    assert result["score"] == 9.0
    assert result["segment_linker"] == "GGGGS"
    assert [segment["segment_index"] for segment in result["segments"]] == [1, 2, 3]


def test_assemble_segmented_sequence():
    seq = generate._assemble_segmented_sequence(
        [{"sequence": "AAAA"}, {"sequence": "BBBB"}],
        linker="GS",
    )
    assert seq == "AAAAGSBBBB"
