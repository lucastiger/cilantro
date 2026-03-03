import cma
import numpy as np
from typing import Callable

from utils.seq_utils import decode_sequence_from_ids
from scoring.antigen_scoring import (
    BASE_SCORE_WEIGHT,
    antigen_score_components,
    score_antigen_candidate_with_breakdown,
)


def interleave_generated_sequences_and_epitopes(generated_sequences: list[str], epitopes: list[str]) -> str:
    clean_epitopes = [ep for ep in epitopes if ep]
    if not clean_epitopes:
        return "".join(generated_sequences)

    if len(clean_epitopes) != 12:
        raise ValueError("epitopes must contain exactly 12 non-empty entries")
    if len(generated_sequences) != 2:
        raise ValueError("generated_sequences must contain exactly 2 entries (seqA and seqB)")

    seq_a, seq_b = generated_sequences

    scaffolds = [
        seq_a[10:65],
        seq_a[55:100],
        seq_a[90:],
        seq_b[10:45],
        seq_b[35:75],
        seq_b[60:105],
    ]
    scaffold_tail = seq_b[100:]

    clusters = [
        f"{clean_epitopes[idx]}AAY{clean_epitopes[idx + 1]}"
        for idx in range(0, len(clean_epitopes), 2)
    ]

    antigen_parts: list[str] = []
    for scaffold, cluster in zip(scaffolds, clusters):
        antigen_parts.append(scaffold)
        antigen_parts.append(cluster)
    antigen_parts.append(scaffold_tail)

    return "".join(antigen_parts)


def latent_optimize(
    model,
    seed_latent,
    target_epitopes: list[str],
    sigma=0.5,
    popsize=16,
    generations=100,
    progress_callback: Callable[[str, dict], None] | None = None,
):
    clean_target_epitopes = [ep for ep in target_epitopes if ep]
    if len(clean_target_epitopes) != 12:
        raise ValueError("target_epitopes must contain exactly 12 non-empty entries")

    seed_latent_array = np.asarray(seed_latent, dtype=np.float32)
    segment_count = 2
    expanded_seed = np.tile(seed_latent_array, segment_count)
    latent_dim = seed_latent_array.shape[0]

    es = cma.CMAEvolutionStrategy(
        expanded_seed.tolist(),
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
        base_score_weight = BASE_SCORE_WEIGHT

        solutions = es.ask()
        losses = []
        generation_candidates = []

        for idx, z_vec in enumerate(solutions, start=1):
            flattened = np.array(z_vec, dtype=np.float32)
            generated_sequences = []
            for segment_idx in range(segment_count):
                start = segment_idx * latent_dim
                end = start + latent_dim
                z_segment = flattened[start:end][None, :]
                token_ids = model.decode(z_segment)[0]
                generated_sequences.append(decode_sequence_from_ids(token_ids))

            seq = interleave_generated_sequences_and_epitopes(
                generated_sequences,
                clean_target_epitopes,
            )

            score_breakdown = score_antigen_candidate_with_breakdown(seq)
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
                    "seqA": generated_sequences[0],
                    "seqB": generated_sequences[1],
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
                            "seqA": generated_sequences[0],
                            "seqB": generated_sequences[1],
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
                            "base_score_weight": base_score_weight,
                            "hydrophobic_fraction": score_breakdown.get("hydrophobic_fraction"),
                            "hydrophobicity_threshold": score_breakdown.get("hydrophobicity_threshold"),
                            "hydrophobicity_penalty": score_breakdown.get("hydrophobicity_penalty"),
                            "weighted_hydrophobicity_penalty": score_breakdown.get("weighted_hydrophobicity_penalty"),
                            "target_epitopes": clean_target_epitopes,
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
