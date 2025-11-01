# scoring/epitope_scoring.py
import math
from collections import Counter
from typing import List
from Bio import SeqIO

AA = list("ACDEFGHIKLMNPQRSTVWY")

def shannon_entropy(column_aa_counts: Counter) -> float:
    total = sum(column_aa_counts.values())
    if total == 0:
        return 1.0
    ent = 0.0
    for aa, c in column_aa_counts.items():
        p = c / total
        ent -= p * math.log2(p)
    # normalize by log2(20) to give 0..1
    return ent / math.log2(20)

def msa_entropy(msa_sequences: List[str], window_start: int, window_length: int) -> float:
    L = window_length
    ent_sum = 0.0
    counted = 0
    for i in range(window_start, window_start + L):
        column = [seq[i] for seq in msa_sequences if i < len(seq) and seq[i] != '-']
        if not column:
            continue
        counts = Counter(column)
        ent_sum += shannon_entropy(counts)
        counted += 1
    if counted == 0:
        return 1.0
    return ent_sum / counted

def sliding_epitope_scores(msa_sequences: List[str], min_len=5, max_len=35):
    L = max(len(s) for s in msa_sequences)
    results = []
    for l in range(min_len, min(max_len, L) + 1):
        for start in range(0, L - l + 1):
            ent = msa_entropy(msa_sequences, start, l)
            results.append((start, l, ent))
    return results

def load_fasta(path):
    seqs = []
    for rec in SeqIO.parse(path, "fasta"):
        seqs.append(str(rec.seq).upper())
    return seqs
