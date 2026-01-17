import argparse
import tensorflow as tf
from models.vae import SeqVAE
from utils.seq_utils import build_vocab_and_encode, load_fasta_as_sequences
from scoring.find_top_epitopes import find_top_epitopes
from optimize.cma_latent_search import latent_optimize

def main(args):
    # Load sequences
    seqs = load_fasta_as_sequences(args.input_fasta)
    tokenized, vocab = build_vocab_and_encode(seqs, max_len=args.max_len)

    # Load model
    model = SeqVAE(
        vocab_size=len(vocab) + 3,
        max_len=args.max_len
    )
    model.build((None, args.max_len))
    model.load_weights(args.ckpt)

    # ---- Epitope discovery ----
    top_epitopes = find_top_epitopes(
        args.input_fasta,
        min_len=args.min_ep_len,
        max_len=args.max_ep_len,
        top_k=args.top_k
    )

    # Flatten epitope sequence set
    target_epitopes = set()
    for ep in top_epitopes:
        target_epitopes |= ep["sequences"]

    print(f"Identified {len(target_epitopes)} conserved epitope variants")

    # ---- Seed latent ----
    z_mean, _ = model.encode(tf.constant(tokenized[:1]))
    z_seed = z_mean.numpy()[0]

    # ---- Optimize ----
    best = latent_optimize(
        model=model,
        seed_latent=z_seed,
        target_epitopes=target_epitopes,
        sigma=args.sigma,
        popsize=args.popsize,
        generations=args.generations,
    )

    print("\nBest generated antigen:")
    print(best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max_len", type=int, default=200)

    parser.add_argument("--min_ep_len", type=int, default=5)
    parser.add_argument("--max_ep_len", type=int, default=35)
    parser.add_argument("--top_k", type=int, default=10)

    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--generations", type=int, default=50)

    args = parser.parse_args()
    main(args)
