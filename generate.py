# generate.py
import argparse
import numpy as np
from models.vae import SeqVAE
import tensorflow as tf
from utils.seq_utils import build_vocab_and_encode, load_fasta_as_sequences, decode_sequence_from_ids
from optimize.cma_latent_search import latent_optimize

def main(args):
    seqs = load_fasta_as_sequences(args.input_fasta)
    tokenized, vocab = build_vocab_and_encode(seqs, max_len=args.max_len)
    model = SeqVAE(vocab_size=len(vocab), max_len=args.max_len)
    model.load_weights(args.ckpt)
    # encode first sequence as seed
    z_mean, z_log = model.encode(tf.constant(tokenized[:1]))
    z_seed = z_mean.numpy()[0]

    best = latent_optimize(
        model,
        z_seed,
        output_dir=args.output_dir,
        sigma=args.sigma,
        popsize=args.popsize,
        generations=args.generations,
    )
    
    print("Best candidate:", best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", default="data/example_sequences.fasta")
    parser.add_argument("--ckpt", default="checkpoints/vae_epoch1.ckpt")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--generations", type=int, default=10)
    args = parser.parse_args()
    main(args)
