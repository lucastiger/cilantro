# scoring/find_top_epitopes.py
from typing import Callable, Dict, List
from collections import Counter
import math
from Bio import SeqIO

from scoring.prediction_tools import (
    default_allele_frequencies,
    mhcflurry_supported_alleles,
    parse_allele_frequencies_env,
    predict_mhcflurry_kd_matrix,
)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
IC50_BEST_NM = 1.0
IC50_WORST_NM = 50000.0
KD_BINDING_THRESHOLD_NM = 500.0


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


def _kd_contribution(kd_nm: float) -> float:
    kd_clamped = max(float(kd_nm), 1e-12)
    return max(0.0, math.log10(KD_BINDING_THRESHOLD_NM) - math.log10(kd_clamped))


def _epitope_immunogenicity_score(
    epitope: str,
    kd_by_allele: Dict[str, Dict[str, float]],
    allele_frequencies: Dict[str, float],
) -> float:
    total_freq = sum(freq for freq in allele_frequencies.values() if freq > 0)
    if total_freq <= 0:
        return 0.0

    score = 0.0
    for allele, freq in allele_frequencies.items():
        if freq <= 0:
            continue
        kd = kd_by_allele.get(allele, {}).get(epitope)
        if kd is None:
            continue
        pa = freq / total_freq
        score += pa * _kd_contribution(kd)
    return score


def find_top_epitopes(
    fasta_path: str,
    min_len: int = 9,
    max_len: int = 9,
    top_k: int = 20,
    progress_callback: Callable[[str, Dict], None] | None = None,
) -> List[Dict]:
    """
    Returns a list of top conserved epitopes across window sizes.
    Each entry:
    {
        'start': int,
        'length': int,
        'entropy': float,
        'entropy_score': float,
        'epitope_immunogenicity': float,
        'overall_score': float,
        'sequences': set(str),
        'epitope_scores': dict(str, float),
    }
    """
    seqs = load_sequences(fasta_path)
    if not seqs:
        raise ValueError("No sequences found.")

    # Use the longest available sequence when scanning windows. Shorter
    # sequences are already excluded per-window below, so tying the scan range
    # to the shortest sequence can collapse candidate lengths as dataset size
    # increases (e.g., when a short outlier is introduced).
    L = max(len(s) for s in seqs)
    results = []

    if min_len <= 0 or max_len <= 0:
        raise ValueError("Epitope lengths must be positive.")
    if min_len > max_len:
        raise ValueError("min_len must be <= max_len.")
    if min_len > L:
        return []

    max_len = min(max_len, L)
    total_windows = sum((L - window + 1) for window in range(min_len, max_len + 1))

    if progress_callback:
        progress_callback(
            "start",
            {
                "stage": "epitope_identification",
                "total_windows": total_windows,
                "window_range": [min_len, max_len],
                "sequence_count": len(seqs),
            },
        )

    supported_alleles = mhcflurry_supported_alleles()
    supported_allele_set = set(supported_alleles)
    freq_map = parse_allele_frequencies_env()
    freq_map = {
        allele: freq
        for allele, freq in freq_map.items()
        if allele in supported_allele_set and freq > 0
    }
    if not freq_map:
        freq_map = default_allele_frequencies(supported_alleles)
    if not freq_map and supported_alleles:
        uniform = 1.0 / len(supported_alleles)
        freq_map = {allele: uniform for allele in supported_alleles}

    prediction_alleles = sorted(freq_map.keys())

    # Entropy is independent of window boundaries, so compute once per position.
    position_entropy = []
    for i in range(L):
        column = [s[i] for s in seqs if i < len(s) and s[i] != "-"]
        position_entropy.append(shannon_entropy(column))

    entropy_prefix = [0.0]
    for value in position_entropy:
        entropy_prefix.append(entropy_prefix[-1] + value)

    unique_epitopes = set()
    processed_windows = 0
    scored_windows = 0
    for window in range(min_len, max_len + 1):
        for start in range(0, L - window + 1):
            unique_epitopes.update(
                s[start:start + window]
                for s in seqs
                if len(s) >= start + window
            )
            processed_windows += 1
        if progress_callback:
            progress_callback(
                "window_scan",
                {
                    "stage": "epitope_identification",
                    "window": window,
                    "processed_windows": processed_windows,
                    "total_windows": total_windows,
                    "percent_complete": processed_windows / total_windows if total_windows else 1.0,
                },
            )

    epitope_score_cache: Dict[str, float] = {}
    if unique_epitopes and prediction_alleles:
        kd_by_allele = predict_mhcflurry_kd_matrix(unique_epitopes, alleles=prediction_alleles)
        for ep in unique_epitopes:
            epitope_score_cache[ep] = _epitope_immunogenicity_score(
                ep,
                kd_by_allele,
                freq_map,
            )
    else:
        for ep in unique_epitopes:
            epitope_score_cache[ep] = 0.0

    for window in range(min_len, max_len + 1):
        for start in range(0, L - window + 1):
            avg_entropy = (entropy_prefix[start + window] - entropy_prefix[start]) / window

            epitope_set = {
                s[start:start + window]
                for s in seqs
                if len(s) >= start + window
            }

            epitope_scores = [
                epitope_score_cache.get(ep, 0.0)
                for ep in epitope_set
            ]
            avg_epitope_score = sum(epitope_scores) / len(epitope_scores) if epitope_scores else 0.0
            entropy_score = 1.0 - avg_entropy
            overall_score = 0.6 * entropy_score + 0.4 * avg_epitope_score

            results.append({
                "start": start,
                "length": window,
                "entropy": avg_entropy,
                "entropy_score": entropy_score,
                "epitope_immunogenicity": avg_epitope_score,
                "overall_score": overall_score,
                "sequences": epitope_set,
                "epitope_scores": {
                    ep: epitope_score_cache.get(ep, 0.0)
                    for ep in epitope_set
                },
            })

        scored_windows += (L - window + 1)
        if progress_callback:
            progress_callback(
                "window_score",
                {
                    "stage": "epitope_identification",
                    "window": window,
                    "processed_windows": scored_windows,
                    "total_windows": total_windows,
                    "percent_complete": scored_windows / total_windows if total_windows else 1.0,
                },
            )

    # higher overall_score = more conserved + better binding affinity
    results.sort(key=lambda x: x["overall_score"], reverse=True)
    top_results = results[:top_k]

    if progress_callback:
        progress_callback(
            "complete",
            {
                "stage": "epitope_identification",
                "selected": len(top_results),
                "total_candidates": len(results),
                "unique_epitopes": len(unique_epitopes),
            },
        )

    return top_results
