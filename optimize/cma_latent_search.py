import cma
import numpy as np
from typing import Callable

from utils.seq_utils import decode_sequence_from_ids
from scoring.antigen_scoring import score_antigen_candidate_with_breakdown


def latent_optimize(
    model,
    seed_latent,
    target_epitopes: set,
    sigma=0.5,
    popsize=16,
    generations=100,
    progress_callback: Callable[[str, dict], None] | None = None,
):
    es = cma.CMAEvolutionStrategy(
        seed_latent.tolist(),
        sigma,
        {"popsize": popsize, "verb_log": 0}
    )

    best = None

    if progress_callback:
        progress_callback(
            "start",
            {
                "stage": "antigen_optimization",
                "generations": generations,
                "popsize": popsize,
            },
        )

    for gen in range(generations):
        solutions = es.ask()
        losses = []
        generation_candidates = []

        for idx, z_vec in enumerate(solutions, start=1):
            z = np.array(z_vec, dtype=np.float32)[None, :]
            token_ids = model.decode(z)[0]
            seq = decode_sequence_from_ids(token_ids)

            score_breakdown = score_antigen_candidate_with_breakdown(seq, target_epitopes)
            score = score_breakdown["score"]
            losses.append(-score)
            generation_candidates.append({"sequence": seq, "score": score})

            if best is None or score > best["score"]:
                best = {
                    "sequence": seq,
                    "score": score,
                    "generation": gen,
                    "score_breakdown": score_breakdown,
                }

                if progress_callback:
                    progress_callback(
                        "new_best_candidate",
                        {
                            "stage": "antigen_optimization",
                            "generation": gen + 1,
                            "generations": generations,
                            "sequence": seq,
                            "score": score,
                            "immunogenicity": score_breakdown["immunogenicity"],
                            "esm2_sequence_log_likelihood": score_breakdown["esm2_sequence_log_likelihood"],
                            "toxicity": score_breakdown["toxicity"],
                            "soft_epitope": score_breakdown["soft_epitope"],
                            "hard_epitope_constraint": score_breakdown["hard_epitope_constraint"],
                        },
                    )

            if progress_callback:
                progress_callback(
                    "candidate_evaluated",
                    {
                        "stage": "antigen_optimization",
                        "generation": gen + 1,
                        "generations": generations,
                        "candidate_index": idx,
                        "candidates_per_generation": len(solutions),
                        "candidate_score": score,
                        "best_score": best["score"],
                    },
                )

        es.tell(solutions, losses)

        top_candidates = sorted(
            generation_candidates,
            key=lambda candidate: candidate["score"],
            reverse=True,
        )[:10]

        if progress_callback:
            progress_callback(
                "generation_complete",
                {
                    "stage": "antigen_optimization",
                    "generation": gen + 1,
                    "generations": generations,
                    "best_score": best["score"] if best else None,
                    "top_candidates": top_candidates,
                },
            )

    if progress_callback:
        progress_callback(
            "complete",
            {
                "stage": "antigen_optimization",
                "best": best,
            },
        )

    return best
