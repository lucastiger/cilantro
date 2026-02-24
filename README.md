# cilantro

Author: Lucas Wu
Date: 2025-10-29

Overview
--------
End-to-end computational pipeline for discovering conserved, immunogenic epitopes
and generating de novo antigen candidates using a Variational Autoencoder (VAE)
and CMA-ES latent space optimization. 

Quickstart
----------
1) Download protein sequences from NCBI (set an email for Entrez):

```
export NCBI_EMAIL="you@example.com"
python data/ncbi_downloader.py \
  --taxon "Influenza A virus" \
  --out_fasta data/influenza_a_proteins.fasta \
  --max_records 500
```

2) Train the VAE:

```
python train.py \
  --input_fasta data/influenza_a_proteins.fasta \
  --epochs 2 \
  --batch_size 4 \
  --max_len 200 \
  --ckpt_dir checkpoints
```

3) Generate and evaluate candidates:

```
python generate.py \
  --input_fasta data/influenza_a_proteins.fasta \
  --ckpt checkpoints/vae_epoch1.weights.h5 \
  --max_len 200 \
  --generations 50 \
  --popsize 16 \
  --output_json outputs/best.json

python evaluate.py \
  --seq_file outputs/best.json \
  --output_json final_scores.json
```

Using a saved `.weights.h5` VAE checkpoint in Google Colab
----------------------------------------------------------
If you already trained a VAE and have a checkpoint file (for example
`vae_epoch1.weights.h5`), you can load it in Colab and run the latent optimization
pipeline without retraining.

```python
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from models.vae import SeqVAE
from utils.seq_utils import build_vocab_and_encode, load_fasta_as_sequences, decode_sequence_from_ids

# 1) Load reference sequences to rebuild the amino-acid vocab and token shapes
max_len = 200
seqs = load_fasta_as_sequences('data/influenza_a_proteins.fasta')
tokenized, vocab = build_vocab_and_encode(seqs, max_len=max_len)

# 2) Recreate the model with the SAME architecture used in training
model = SeqVAE(
    vocab_size=len(vocab) + 3,
    max_len=max_len,
    emb_dim=128,
    enc_units=256,
    latent_dim=64,
    dec_units=256,
    dropout=0.2,
)
model.build((None, max_len))

# 3) Load weights (.weights.h5)
model.load_weights('/content/drive/MyDrive/checkpoints/vae_epoch1.weights.h5')
```

Simple encode/decode smoke test
-------------------------------
Use one protein sequence, encode to latent space, then decode back into a generated
protein-like sequence.

```python
import tensorflow as tf

test_seq = 'MKTIIALSYIFCLVFADYKDDDDA'  # replace with your sequence
test_ids, _ = build_vocab_and_encode([test_seq], max_len=max_len)
x = tf.constant(test_ids)

# Encode
z_mean, z_log_var = model.encode(x)
print('z_mean shape:', z_mean.shape)  # expected: (1, latent_dim)

# Decode (autoregressive because target_ids is omitted)
logits = model.decode(z_mean)
pred_ids = tf.argmax(logits, axis=-1).numpy()[0]
decoded_seq = decode_sequence_from_ids(pred_ids)

print('Input sequence:  ', test_seq)
print('Decoded sequence:', decoded_seq)
```

Notes:
- The model checkpoint is *weights only*, so you must recreate `SeqVAE` with matching
  hyperparameters before `load_weights`.
- Keep `max_len` consistent with training.
- If your training data differed, rebuild vocab from that same corpus before loading.
- For full antigen optimization after loading, run `generate.py` with your checkpoint.


Testing
-------
Unit tests (fast, mocked integrations):

```
pytest
```

Real scoring-tool smoke tests (runs actual MHCflurry / ToxinPred3 / ESM-2 path):

```
RUN_SCORING_TOOL_SMOKE_TESTS=1 pytest -m smoke tests/test_scoring_tool_smoke.py
```

Notes:
- Smoke tests are opt-in because they require heavyweight runtime dependencies and model/tool availability.
- `predict_esm2_mean_log_likelihood` smoke testing supports either `ESM2_LIKELIHOOD_CMD` or the local `fair-esm` + `torch` model path.

License
-------
MIT. See LICENSE file for details.
