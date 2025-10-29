# cilantro

Author: Lucas Wu (starter)
Date: 2025-10-29

Overview
--------
End-to-end computational pipeline for discovering conserved, immunogenic epitopes
and generating de novo antigen candidates using a Variational Autoencoder (VAE)
and CMA-ES latent space optimization. Includes hooks for AlphaFold/ColabFold,
MHC binding predictors, toxicity prediction, and immune simulation.

This repository provides:
- data ingestion (NCBI)
- epitope scoring (entropy + MHC)
- LSTM-VAE implementation in TensorFlow
- latent optimization via CMA-ES
- scoring wrapper for antigen candidates
- utilities, tests, and Dockerfile
- Colab notebook for running parts on T4

Important: this repo does not include NetMHCpan binary, AlphaFold model files,
or licensed third-party binaries. See Integration section below.

Quickstart (toy run)
--------------------
1. Install Python 3.10+ and create venv (recommended).
2. `pip install -r requirements.txt`
3. Fill `data/example_sequences.fasta` with toy sequences (example provided).
4. `python train.py --input_fasta data/example_sequences.fasta --epochs 2 --batch_size 4 --max_len 80`
5. `python generate.py --input_fasta data/example_sequences.fasta --ckpt checkpoints/vae_epoch1.ckpt`

Integration notes
-----------------
- AlphaFold/ColabFold: use ColabFold for structure prediction & pLDDT.
- MHC binding: use mhcflurry (pip) or NetMHCpan (license required).
- Folding energy: FoldX or Rosetta (binaries).
- Toxicity: ToxDL or heuristic filters.

License
-------
MIT. See LICENSE file for details.

Contact
-------
If you have questions or want help running this on a cluster, email: lucastiger33@gmail.com
