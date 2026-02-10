# scoring/find_top_epitopes.py
from typing import Dict, List
from collections import Counter
import math
from Bio import SeqIO

from scoring.prediction_tools import predict_mhci_ic50s

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
IC50_BEST_NM = 1.0
IC50_WORST_NM = 50000.0


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


def normalize_ic50(ic50_nm: float) -> float:
    """
    Normalize IC50 into [0,1] where 1 is best (low IC50).
    Uses log scaling to cover nM to uM range.
    """
    ic50_clamped = max(IC50_BEST_NM, min(ic50_nm, IC50_WORST_NM))
    log_best = math.log10(IC50_BEST_NM)
    log_worst = math.log10(IC50_WORST_NM)
    log_val = math.log10(ic50_clamped)
    return max(0.0, min(1.0, 1.0 - (log_val - log_best) / (log_worst - log_best)))


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
        'entropy_score': float,
        'ic50': float,
        'ic50_score': float,
        'overall_score': float,
        'sequences': set(str),
    }
    """
    seqs = load_sequences(fasta_path)
    if not seqs:
        raise ValueError("No sequences found.")

    L = min(len(s) for s in seqs)
    results = []

    ic50_cache: Dict[str, float] = {}

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

            missing_epitopes = [ep for ep in epitope_set if ep not in ic50_cache]
            if missing_epitopes:
                ic50_cache.update(predict_mhci_ic50s(missing_epitopes))

            ic50_values = [
                ic50_cache.get(ep, IC50_WORST_NM)
                for ep in epitope_set
            ]
            avg_ic50 = sum(ic50_values) / len(ic50_values) if ic50_values else IC50_WORST_NM
            ic50_score = normalize_ic50(avg_ic50)
            entropy_score = 1.0 - avg_entropy
            overall_score = 0.6 * entropy_score + 0.4 * ic50_score

            results.append({
                "start": start,
                "length": window,
                "entropy": avg_entropy,
                "entropy_score": entropy_score,
                "ic50": avg_ic50,
                "ic50_score": ic50_score,
                "overall_score": overall_score,
                "sequences": epitope_set,
            })

    # higher overall_score = more conserved + better binding affinity
    results.sort(key=lambda x: x["overall_score"], reverse=True)
    return results[:top_k]
