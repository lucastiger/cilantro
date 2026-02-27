import argparse
import json
import os
from pathlib import Path

from models.vae import ProteinSeqVAE
from optimize.cma_latent_search import latent_optimize
from scoring.antigen_scoring import score_antigen_candidate_with_breakdown
from scoring.find_top_epitopes import find_top_epitopes
from utils.seq_utils import build_vocab_and_encode, load_fasta_as_sequences


def _resolve_output_path(output_json: str) -> Path:
    output_path = Path(output_json)
    if output_path.is_absolute():
        return output_path

    script_dir = Path(__file__).resolve().parent
    return script_dir / output_path


def _resolve_ckpt_path(ckpt: str) -> Path:
    ckpt_path = Path(ckpt)
    if ckpt_path.is_absolute():
        return ckpt_path

    script_dir = Path(__file__).resolve().parent
    return (script_dir / ckpt_path).resolve()


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
                for rank, candidate in enumerate(payload.get("top_candidates", []), start=1):
                    print(
                        f"  #{rank}: score={candidate['score']:.4f} "
                        f"sequence={candidate['sequence']}"
                    )
            elif event == "new_best_candidate":
                weighted_hydrophobicity_penalty = payload.get("weighted_hydrophobicity_penalty")
                print(
                    f"[optimize] new best at generation {payload['generation']}/{payload['generations']} "
                    f"score={payload['score']:.4f}"
                )
                print(f"  target_epitopes={payload['target_epitopes']}")
                print(f"  sequence={payload['sequence']}")
                print(f"  immunogenicity={payload['immunogenicity']:.4f}")
                print(
                    "  esm2_sequence_log_likelihood="
                    f"{payload['esm2_sequence_log_likelihood']:.4f}"
                )
                print(f"  toxicity={payload['toxicity']:.4f}")
                print(
                    "  normalized_terms="
                    f"immunogenicity:{payload['immunogenicity_term']:.4f}, "
                    f"esm2:{payload['esm2_term']:.4f}, "
                    f"toxicity:{payload['toxicity_term']:.4f}"
                )
                print(
                    "  weighted_base_terms="
                    f"immunogenicity:{payload['weighted_immunogenicity']:.4f}, "
                    f"esm2:{payload['weighted_esm2']:.4f}, "
                    f"toxicity:{payload['weighted_toxicity']:.4f}, "
                    f"base:{payload['base_score']:.4f}"
                )
                print(
                    "  total_score="
                    f"{payload['score']:.4f} "
                    f"(= {payload['base_score']:.4f}*{payload['base_score_weight']:.2f}"
                    + (
                        f" - {weighted_hydrophobicity_penalty:.4f}"
                        if weighted_hydrophobicity_penalty is not None
                        else ""
                    )
                    + ")"
                )
                if weighted_hydrophobicity_penalty is not None:
                    print(
                        "  hydrophobicity="
                        f"fraction:{payload.get('hydrophobic_fraction', 0.0):.4f}, "
                        f"threshold:{payload.get('hydrophobicity_threshold', 0.0):.4f}, "
                        f"penalty:{payload.get('hydrophobicity_penalty', 0.0):.4f}, "
                        f"weighted:{weighted_hydrophobicity_penalty:.4f}"
                    )
            elif event == "complete":
                best = payload.get("best")
                if best:
                    print(
                        "[optimize] Done "
                        f"(best score={best['score']:.4f} at generation {best['generation']})"
                    )

    return _report


def _select_target_epitopes(epitope_scores: dict[str, float], top_n_epitopes: int | None) -> list[str]:
    sorted_epitope_scores = sorted(epitope_scores.items(), key=lambda item: item[1], reverse=True)
    if not sorted_epitope_scores:
        return []
    if top_n_epitopes is None:
        return [ep for ep, _ in sorted_epitope_scores]
    selected_n = min(top_n_epitopes, len(sorted_epitope_scores))
    return [ep for ep, _ in sorted_epitope_scores[:selected_n]]


def _select_antigen_epitopes(epitopes: list[str], target_count: int = 3) -> list[str]:
    return epitopes[:target_count]


def _optimize_for_epitope_set(*, model, seed_latent, epitopes: list[str], sigma: float, popsize: int, generations: int, progress_reporter):
    if not epitopes:
        return {"sequence": "", "score": float("-inf"), "target_epitopes": []}

    best = latent_optimize(
        model=model,
        seed_latent=seed_latent,
        target_epitopes=epitopes,
        sigma=sigma,
        popsize=popsize,
        generations=generations,
        progress_callback=progress_reporter,
    )
    best["target_epitopes"] = epitopes
    return best


def main(args):
    progress_reporter = _build_progress_reporter(args.show_progress, args.progress_per_candidate)

    seqs = load_fasta_as_sequences(args.input_fasta)
    tokenized, _ = build_vocab_and_encode(seqs, max_len=args.max_len)
    model = ProteinSeqVAE(weights_path=str(_resolve_ckpt_path(args.ckpt)))

    top_epitopes = find_top_epitopes(
        args.input_fasta,
        min_len=args.min_ep_len,
        max_len=args.max_ep_len,
        top_k=args.top_k,
        progress_callback=progress_reporter,
    )

    epitope_scores = {}
    for ep in top_epitopes:
        for sequence, score in ep.get("epitope_scores", {}).items():
            epitope_scores[sequence] = max(epitope_scores.get(sequence, 0.0), score)

    target_epitopes = _select_target_epitopes(epitope_scores, args.top_n_epitopes)
    print(f"Identified {len(target_epitopes)} optimization-target epitopes")
    if target_epitopes:
        print("Selected epitopes:")
        for ep in target_epitopes:
            print(f"  {ep}: {epitope_scores[ep]:.4f}")

    z_mean, _ = model.encode(tokenized[:1])
    z_seed = z_mean[0]

    optimization_epitopes = _select_antigen_epitopes(target_epitopes, target_count=3)
    print(f"Using {len(optimization_epitopes)} epitopes in each generated antigen")

    best = _optimize_for_epitope_set(
        model=model,
        seed_latent=z_seed,
        epitopes=optimization_epitopes,
        sigma=args.sigma,
        popsize=args.popsize,
        generations=args.generations,
        progress_reporter=progress_reporter,
    )

    if best["sequence"]:
        best["score_breakdown"] = score_antigen_candidate_with_breakdown(best["sequence"])

    print("\nBest generated antigen:")
    print(best)

    if args.output_json:
        output_path = _resolve_output_path(args.output_json)
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([best], f, indent=2)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument(
        "--ckpt",
        default="../protein-vae/produce_sequences/models/metal16_nostruc",
        help="Path to pretrained protein-vae weights.",
    )
    parser.add_argument("--max_len", type=int, default=200)

    parser.add_argument("--min_ep_len", type=int, default=5)
    parser.add_argument("--max_ep_len", type=int, default=35)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--top_n_epitopes",
        type=int,
        default=None,
        help="Restrict optimization runs to the top N epitopes by score.",
    )

    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument(
        "--output_json",
        default="outputs/best.json",
        help="Output JSON path. Relative paths are resolved from this script's directory.",
    )
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
    if args.top_n_epitopes is not None and args.top_n_epitopes <= 0:
        parser.error("--top_n_epitopes must be a positive integer")
    main(args)
