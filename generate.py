import argparse
import json
import os

import tensorflow as tf

from models.vae import SeqVAE
from optimize.cma_latent_search import latent_optimize
from scoring.find_top_epitopes import find_top_epitopes
from utils.seq_utils import build_vocab_and_encode, load_fasta_as_sequences


def _build_progress_reporter(enabled: bool, per_candidate: bool):
    if not enabled:
        return None

    def _report(event: str, payload: dict):
        stage = payload.get("stage")

        if stage == "epitope_identification":
            if event == "start":
                print(
                    "[epitopes] Starting scan "
                    f"({payload['sequence_count']} sequences, "
                    f"{payload['total_windows']} windows)"
                )
            elif event in {"window_scan", "window_score"}:
                print(
                    f"[epitopes] window={payload['window']} "
                    f"{payload['processed_windows']}/{payload['total_windows']} "
                    f"({payload['percent_complete']:.1%})"
                )
            elif event == "complete":
                print(
                    "[epitopes] Done "
                    f"({payload['selected']} top regions, {payload['unique_epitopes']} unique epitopes)"
                )

        elif stage == "antigen_optimization":
            if event == "start":
                print(
                    "[optimize] Starting CMA-ES "
                    f"({payload['generations']} generations, popsize={payload['popsize']})"
                )
            elif event == "candidate_evaluated" and per_candidate:
                print(
                    f"[optimize] gen {payload['generation']}/{payload['generations']} "
                    f"candidate {payload['candidate_index']}/{payload['candidates_per_generation']} "
                    f"score={payload['candidate_score']:.4f} best={payload['best_score']:.4f}"
                )
            elif event == "generation_complete":
                print(
                    f"[optimize] generation {payload['generation']}/{payload['generations']} complete "
                    f"best={payload['best_score']:.4f}"
                )
            elif event == "complete":
                best = payload.get("best")
                if best:
                    print(
                        "[optimize] Done "
                        f"(best score={best['score']:.4f} at generation {best['generation']})"
                    )

    return _report


def main(args):
    progress_reporter = _build_progress_reporter(args.show_progress, args.progress_per_candidate)

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
        top_k=args.top_k,
        progress_callback=progress_reporter,
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
        progress_callback=progress_reporter,
    )

    print("\nBest generated antigen:")
    print(best)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump([best], f, indent=2)
        print(f"Wrote {args.output_json}")


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
    parser.add_argument("--output_json", default="outputs/best.json")
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Print progress updates for epitope identification and antigen optimization.",
    )
    parser.add_argument(
        "--progress_per_candidate",
        action="store_true",
        help="Print per-candidate optimization progress (verbose).",
    )

    args = parser.parse_args()
    main(args)
