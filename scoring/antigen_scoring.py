# scoring/antigen_scoring.py
import math
from typing import Iterable, List

from scoring.prediction_tools import (
    default_allele_frequencies,
    mhcflurry_supported_alleles,
    parse_allele_frequencies_env,
    predict_esm2_mean_log_likelihood,
    predict_mhcflurry_kd_matrix,
    predict_toxicity_external,
)
ESM2_LOG_LIKELIHOOD_BEST = -0.5
ESM2_LOG_LIKELIHOOD_WORST = -4.0
TOXICITY_WORST = 0.3
KD_BINDING_THRESHOLD_NM = 500.0


def kd_contribution(kd_nm: float) -> float:
    """
    f(Kd) = max(0, log10(500) - log10(Kd)).
    """
    kd_clamped = max(float(kd_nm), 1e-12)
    return max(0.0, math.log10(KD_BINDING_THRESHOLD_NM) - math.log10(kd_clamped))


def immunogenicity_from_kd_matrix(
    kd_by_allele: dict[str, dict[str, float]],
    allele_frequencies: dict[str, float] | None = None,
) -> float:
    """
    I(X) = sum_a p_a * (1 - exp(-S_a(X)))
    S_a(X) = sum_i f(Kd_{a,i})
    """
    if not kd_by_allele:
        return 0.0

    freqs = allele_frequencies or {}
    if not freqs:
        uniform = 1.0 / len(kd_by_allele)
        freqs = {allele: uniform for allele in kd_by_allele}

    total_freq = sum(v for v in freqs.values() if v > 0)
    if total_freq <= 0:
        return 0.0

    score = 0.0
    for allele, peptide_kds in kd_by_allele.items():
        pa = max(0.0, freqs.get(allele, 0.0)) / total_freq
        if pa == 0.0:
            continue
        sa = sum(kd_contribution(kd) for kd in peptide_kds.values())
        score += pa * (1.0 - math.exp(-sa))
    return score


def normalize_esm2_log_likelihood(mean_log_likelihood: float) -> float:
    if mean_log_likelihood <= ESM2_LOG_LIKELIHOOD_WORST:
        return 0.0
    if mean_log_likelihood >= ESM2_LOG_LIKELIHOOD_BEST:
        return 1.0
    return (
        (mean_log_likelihood - ESM2_LOG_LIKELIHOOD_WORST)
        / (ESM2_LOG_LIKELIHOOD_BEST - ESM2_LOG_LIKELIHOOD_WORST)
    )


def normalize_toxicity(toxicity: float) -> float:
    toxicity_clamped = max(0.0, min(toxicity, 1.0))
    if toxicity_clamped >= TOXICITY_WORST:
        return 0.0
    return 1.0 - toxicity_clamped / TOXICITY_WORST


def antigen_score(immunogenicity, esm2_mean_log_likelihood, toxicity):
    """
    Weighted antigen score:
    0.50 immunogenicity
    0.35 ESM-2 sequence mean log-likelihood
    0.15 toxicity
    """
    
    # Normalize each term to [0,1]
    immunogenicity_term = max(0.0, min(float(immunogenicity), 1.0))
    esm2_term = normalize_esm2_log_likelihood(esm2_mean_log_likelihood)
    toxicity_term = normalize_toxicity(toxicity)

    return (
        0.50 * immunogenicity_term +
        0.35 * esm2_term +
        0.15 * toxicity_term
    )

def epitope_presence_score(seq: str, target_epitopes: set) -> float:
    """
    Fraction of target epitopes present in the sequence
    """
    hits = sum(1 for e in target_epitopes if e in seq)
    return hits / len(target_epitopes) if target_epitopes else 0.0


def _generate_peptides(seq: str, lengths: Iterable[int]) -> List[str]:
    peptides: List[str] = []
    for length in lengths:
        if len(seq) < length:
            continue
        for start in range(0, len(seq) - length + 1):
            peptides.append(seq[start:start + length])
    return peptides


def predict_immunogenicity(seq: str) -> float:
    peptides = _generate_peptides(seq, lengths=range(8, 12))
    if not peptides:
        return 0.0

    alleles = mhcflurry_supported_alleles()
    kd_by_allele = predict_mhcflurry_kd_matrix(peptides, alleles=alleles)
    if not kd_by_allele:
        return 0.0

    freq_map = parse_allele_frequencies_env()
    if not freq_map:
        freq_map = default_allele_frequencies(alleles)
    return immunogenicity_from_kd_matrix(kd_by_allele, allele_frequencies=freq_map)

def predict_esm2_sequence_log_likelihood(seq: str) -> float:
    return predict_esm2_mean_log_likelihood(seq)

def predict_toxicity(seq: str) -> float:
    return predict_toxicity_external(seq)
    
def score_antigen_candidate(seq: str, target_epitopes: set | None = None) -> float:
    """
    HARD CONSTRAINT:
    If no target epitope is present → score = 0
    """
    if target_epitopes:
        ep_score = epitope_presence_score(seq, target_epitopes)
        if ep_score == 0.0:
            return 0.0
    else:
        ep_score = 1.0

    base = antigen_score(
        predict_immunogenicity(seq),
        predict_esm2_sequence_log_likelihood(seq),
        predict_toxicity(seq),
    )

    return 0.7 * base + 0.3 * ep_score
