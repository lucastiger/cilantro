import cma
import numpy as np
import tensorflow as tf
from typing import Callable

from utils.seq_utils import decode_sequence_from_ids
from scoring.antigen_scoring import score_antigen_candidate


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

        for idx, z_vec in enumerate(solutions, start=1):
            z = np.array(z_vec, dtype=np.float32)[None, :]
            logits = model.decode(tf.constant(z))
            token_ids = tf.argmax(logits, axis=-1).numpy()[0]
            seq = decode_sequence_from_ids(token_ids)

            score = score_antigen_candidate(seq, target_epitopes)
            losses.append(-score)

            if best is None or score > best["score"]:
                best = {
                    "sequence": seq,
                    "score": score,
                    "generation": gen
                }

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

        if progress_callback:
            progress_callback(
                "generation_complete",
                {
                    "stage": "antigen_optimization",
                    "generation": gen + 1,
                    "generations": generations,
                    "best_score": best["score"] if best else None,
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
