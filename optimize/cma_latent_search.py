import cma
import numpy as np
from typing import Callable

from utils.seq_utils import decode_sequence_from_ids
from scoring.antigen_scoring import (
    BASE_SCORE_WEIGHT,
    HARD_CONSTRAINT_WEIGHT,
    SOFT_EPITOPE_WEIGHT,
    antigen_score_components,
    score_antigen_candidate_with_breakdown,
)


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
            base_components = {
                "immunogenicity_term": score_breakdown.get("immunogenicity_term"),
                "esm2_term": score_breakdown.get("esm2_term"),
                "toxicity_term": score_breakdown.get("toxicity_term"),
                "weighted_immunogenicity": score_breakdown.get("weighted_immunogenicity"),
                "weighted_esm2": score_breakdown.get("weighted_esm2"),
                "weighted_toxicity": score_breakdown.get("weighted_toxicity"),
                "base_score": score_breakdown.get("base_score"),
            }
            if any(value is None for value in base_components.values()):
                base_components = antigen_score_components(
                    score_breakdown["immunogenicity"],
                    score_breakdown["esm2_sequence_log_likelihood"],
                    score_breakdown["toxicity"],
                )
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
                            "immunogenicity_term": base_components["immunogenicity_term"],
                            "esm2_term": base_components["esm2_term"],
                            "toxicity_term": base_components["toxicity_term"],
                            "weighted_immunogenicity": base_components["weighted_immunogenicity"],
                            "weighted_esm2": base_components["weighted_esm2"],
                            "weighted_toxicity": base_components["weighted_toxicity"],
                            "base_score": base_components["base_score"],
                            "base_score_weight": BASE_SCORE_WEIGHT,
                            "soft_epitope_weight": SOFT_EPITOPE_WEIGHT,
                            "hard_constraint_weight": HARD_CONSTRAINT_WEIGHT,
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
