import cma
import numpy as np
import tensorflow as tf
from utils.seq_utils import decode_sequence_from_ids
from scoring.antigen_scoring import score_antigen_candidate


def latent_optimize(
    model,
    seed_latent,
    target_epitopes: set,
    sigma=0.5,
    popsize=16,
    generations=100,
):
    es = cma.CMAEvolutionStrategy(
        seed_latent.tolist(),
        sigma,
        {"popsize": popsize, "verb_log": 0}
    )

    best = None

    for gen in range(generations):
        solutions = es.ask()
        losses = []

        for z_vec in solutions:
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

        es.tell(solutions, losses)

    return best
