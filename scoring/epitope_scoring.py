# scoring/epitope_scoring.py
import math
from collections import Counter
from typing import List
from Bio import SeqIO

AA = list("ACDEFGHIKLMNPQRSTVWY")

def shannon_entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 1.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
        
    return ent / math.log2(20)

def conservation_score(msa, seq):
    score = 0.0
    L = min(len(seq), len(msa[0]))

    for i in range(L):
        col = [s[i] for s in msa if i < len(s)]
        score += 1.0 - shannon_entropy(Counter(col))

    return score / L

def load_fasta(path):
    seqs = []
    for rec in SeqIO.parse(path, "fasta"):
        seqs.append(str(rec.seq).upper())
    return seqs
