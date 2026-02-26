from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.vae import ProteinSeqVAE
from utils.seq_utils import build_vocab_and_encode, decode_sequence_from_ids


SMOKE_FLAG = "RUN_PROTEIN_VAE_SMOKE_TESTS"
CKPT_PATH = Path("protein-vae/produce_sequences/models/metal16_nostruc")


def _require_smoke_flag() -> None:
    if os.getenv(SMOKE_FLAG, "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip(
            f"Set {SMOKE_FLAG}=1 to run prot-vae encode/decode smoke tests."
        )


@pytest.mark.smoke
def test_protein_vae_encode_decode_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    _require_smoke_flag()

    if not CKPT_PATH.exists():
        pytest.skip(f"Pretrained prot-vae checkpoint not found at {CKPT_PATH}.")

    input_seq = "MKTIIALSYIFCLVFADYKDDDDK"
    tokenized, _ = build_vocab_and_encode([input_seq], max_len=140)

    model = ProteinSeqVAE(weights_path=str(CKPT_PATH))
    z_mean, z_log_var = model.encode(tokenized)
    decoded_token_ids = model.decode(z_mean)
    decoded_seq = decode_sequence_from_ids(decoded_token_ids[0])

    print(f"Input sequence:   {input_seq}")
    print(f"Latent mean shape: {z_mean.shape}")
    print(f"Latent log_var shape: {z_log_var.shape}")
    print(f"Decoded sequence: {decoded_seq}")

    captured = capsys.readouterr()
    assert "Input sequence:" in captured.out
    assert "Decoded sequence:" in captured.out

    assert z_mean.shape == (1, model.latent_dim)
    assert z_log_var.shape == (1, model.latent_dim)
    assert decoded_token_ids.shape == (1, 140)
    assert isinstance(decoded_seq, str)
    assert decoded_seq
