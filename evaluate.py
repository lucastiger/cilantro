#evaluation of candidate antigens via the antigen scoring function


# orchestration script to run downstream checks (AlphaFold/MHC/Tox/Simulation)
import argparse
import os
import json
from scoring.antigen_scoring import score_antigen_candidate

def evaluate_sequences(seq_file, output_json="evaluation_results.json"):
    # seq_file: newline-delimited sequences or JSON outputs from optimizer
    results = []
    with open(seq_file) as f:
        for line in f:
            seq = line.strip()
            if not seq:
                continue
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
