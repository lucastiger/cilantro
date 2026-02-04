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

License
-------
MIT. See LICENSE file for details.
