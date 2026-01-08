import cma
import numpy as np
import tensorflow as tf
from utils.seq_utils import decode_sequence_from_ids
from scoring.antigen_scoring import score_antigen_candidate

def latent_optimize(
    model,
    seed_latent,
    output_dir="outputs",
    sigma=0.5,
    popsize=16,
    generations=100,
):
    dim = seed_latent.shape[-1]
    es = cma.CMAEvolutionStrategy(seed_latent.tolist(), sigma, {'popsize': popsize})
    best = None

    for gen in range(generations):
        solutions = es.ask()
        scores = []

        for z_vec in solutions:
            z = np.array(z_vec, dtype=np.float32)[None, :]
            logits = model.decode(tf.constant(z), seq_len=model.max_len)
            token_ids = tf.argmax(logits, axis=-1).numpy()[0]
            seq = decode_sequence_from_ids(token_ids)
            score = score_antigen_candidate(seq)
            scores.append(-score)

            if best is None or score > best["score"]:
                best = {"seq": seq, "score": score, "gen": gen}

        es.tell(solutions, scores)

    return best
