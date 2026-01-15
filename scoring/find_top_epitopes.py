# scoring/find_top_epitopes.py
from typing import List, Tuple, Dict
from collections import Counter
import math
from Bio import SeqIO

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def shannon_entropy(column: List[str]) -> float:
    counts = Counter(column)
    total = sum(counts.values())
    if total == 0:
        return 1.0

    entropy = 0.0
    for aa in AMINO_ACIDS:
        p = counts.get(aa, 0) / total
        if p > 0:
            entropy -= p * math.log2(p)

    # normalize to [0,1]
    return entropy / math.log2(len(AMINO_ACIDS))


def load_sequences(fasta_path: str) -> List[str]:
    return [str(rec.seq).upper() for rec in SeqIO.parse(fasta_path, "fasta")]


def find_top_epitopes(
    fasta_path: str,
    min_len: int = 5,
    max_len: int = 35,
    top_k: int = 20,
) -> List[Dict]:
    """
    Returns a list of top conserved epitopes across window sizes.
    Each entry:
    {
        'start': int,
        'length': int,
        'entropy': float,
        'sequences': set(str)
    }
    """
    seqs = load_sequences(fasta_path)
    if not seqs:
        raise ValueError("No sequences found.")

    L = min(len(s) for s in seqs)
    results = []

    for window in range(min_len, max_len + 1):
        for start in range(0, L - window + 1):
            entropies = []
            for i in range(start, start + window):
                column = [
                    s[i] for s in seqs
                    if i < len(s) and s[i] != "-"
                ]
                entropies.append(shannon_entropy(column))

            avg_entropy = sum(entropies) / len(entropies)

            epitope_set = {
                s[start:start + window]
                for s in seqs
                if len(s) >= start + window
            }

            results.append({
                "start": start,
                "length": window,
                "entropy": avg_entropy,
                "sequences": epitope_set
            })

    # lower entropy = more conserved
    results.sort(key=lambda x: x["entropy"])
    return results[:top_k]
