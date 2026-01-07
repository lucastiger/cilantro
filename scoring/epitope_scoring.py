import math
from collections import Counter
from typing import List

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def shannon_entropy(column):
    counts = Counter(column)
    total = sum(counts.values())
    entropy = 0.0
    for aa in AMINO_ACIDS:
        p = counts.get(aa, 0) / total if total > 0 else 0
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def normalized_entropy(column):
    return shannon_entropy(column) / math.log2(len(AMINO_ACIDS))

def epitope_conservation_score(msa: List[str], start: int, length: int):
    entropies = []
    for i in range(start, start + length):
        column = [seq[i] for seq in msa if i < len(seq) and seq[i] != "-"]
        if not column:
            continue
        entropies.append(normalized_entropy(column))
    if not entropies:
        return 1.0
    return sum(entropies) / len(entropies)
