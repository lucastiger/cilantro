#evaluation of candidate antigens via the antigen scoring function


# orchestration script to run downstream checks (AlphaFold/MHC/Tox/Simulation)
import argparse
import json
import os
from scoring.antigen_scoring import score_antigen_candidate

def _load_sequences(seq_file):
    if seq_file.endswith(".json"):
        with open(seq_file) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        seqs = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    seqs.append(item)
                elif isinstance(item, dict):
                    seq = item.get("sequence") or item.get("seq")
                    if seq:
                        seqs.append(seq)
        return seqs

    seqs = []
    with open(seq_file) as f:
        for line in f:
            seq = line.strip()
            if seq:
                seqs.append(seq)
    return seqs


def evaluate_sequences(seq_file, output_json="evaluation_results.json"):
    # seq_file: newline-delimited sequences or JSON outputs from optimizer
    results = []
    for seq in _load_sequences(seq_file):
        sc = score_antigen_candidate(seq)
        results.append({"seq": seq, "score": sc})
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_file", required=True)
    parser.add_argument("--output_json", default="evaluation_results.json")
    args = parser.parse_args()
    evaluate_sequences(args.seq_file, args.output_json)
