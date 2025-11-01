# optimize/cma_latent_search.py
import cma
import numpy as np
import os
import json
import tensorflow as tf
from utils.seq_utils import decode_sequence_from_ids

def latent_optimize(model, seed_latent, score_fn, output_dir="outputs", sigma=0.5, popsize=16, generations=100):
    os.makedirs(output_dir, exist_ok=True)
    dim = seed_latent.shape[-1]
    es = cma.CMAEvolutionStrategy(seed_latent.tolist(), sigma, {'popsize': popsize, 'verb_log':0})
    best = None
    for gen in range(generations):
        solutions = es.ask()
        decoded = []
        scores = []
        for s in solutions:
            z = np.array(s).astype(np.float32)[None, :]
            logits = model.decode(tf.constant(z), seq_len=model.max_len)
            token_ids = tf.argmax(logits, axis=-1).numpy()[0]
            seq = decode_sequence_from_ids(token_ids)
            sc = score_fn(seq)
            decoded.append((seq, sc))
            scores.append(-sc)
        es.tell(solutions, scores)
        es.disp()
        best_idx = int(np.argmin(scores))
        best_seq, best_score = decoded[best_idx]
        with open(os.path.join(output_dir, f"gen_{gen:03d}.json"), "w") as f:
            json.dump({"gen": gen, "best_seq": best_seq, "score": float(best_score)}, f)
        if best is None or best_score > best["score"]:
            best = {"seq": best_seq, "score": best_score, "gen": gen}
    return best
