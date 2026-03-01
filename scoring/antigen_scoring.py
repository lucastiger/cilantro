# scoring/antigen_scoring.py
import math
import os
from collections import OrderedDict
from functools import lru_cache
from typing import Iterable, List

from scoring.prediction_tools import (
    default_allele_frequencies,
    mhcflurry_supported_alleles,
    parse_allele_frequencies_env,
    predict_esm2_mean_log_likelihood,
    predict_mhcflurry_kd_matrix,
    predict_toxicity_external_batch,
)
ESM2_LOG_LIKELIHOOD_BEST = -0.05
ESM2_LOG_LIKELIHOOD_WORST = -1
TOXICITY_WORST = 0.3
KD_BINDING_THRESHOLD_NM = 150.0
TOXICITY_WINDOW_SIZE = 15
TOXICITY_BETA = 10.0
SOFT_EPITOPE_ALPHA = 8.0
BASE_SCORE_WEIGHT = 1.0
HYDROPHOBICITY_MAX_FRACTION = 0.45
HYDROPHOBICITY_PENALTY_WEIGHT = 0.25
HYDROPHOBIC_RESIDUES = frozenset({"A", "V", "I", "L", "M", "F", "W", "Y", "C"})
AMINO_ACID_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
IMMUNOGENICITY_ALLELE_COVERAGE = 0.9
IMMUNOGENICITY_MAX_ALLELES = 12
TOXICITY_WINDOW_CACHE_MAXSIZE = 50000
_TOXICITY_WINDOW_CACHE: OrderedDict[str, float] = OrderedDict()


def _toxicity_windows_cached_scores(windows: list[str]) -> list[float]:
    misses: list[str] = []
    seen_misses: set[str] = set()
    for window in windows:
        if window in _TOXICITY_WINDOW_CACHE:
            _TOXICITY_WINDOW_CACHE.move_to_end(window)
            continue
        if window not in seen_misses:
            seen_misses.add(window)
            misses.append(window)

    if misses:
        predicted = predict_toxicity_external_batch(misses)
        for window, score in zip(misses, predicted):
            _TOXICITY_WINDOW_CACHE[window] = score
            _TOXICITY_WINDOW_CACHE.move_to_end(window)
            while len(_TOXICITY_WINDOW_CACHE) > TOXICITY_WINDOW_CACHE_MAXSIZE:
                _TOXICITY_WINDOW_CACHE.popitem(last=False)

    return [_TOXICITY_WINDOW_CACHE.get(window, 0.0) for window in windows]


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
    """
    Convert toxicity risk to a goodness term in [0, 1].

    The toxicity predictor emits risk-like values where larger numbers are worse.
    We map that to a score where larger is better, and we intentionally saturate
    everything at or above TOXICITY_WORST to 0 so very toxic candidates are all
    treated as equally unacceptable.
    """
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


def antigen_score_components(
    immunogenicity: float,
    esm2_mean_log_likelihood: float,
    toxicity: float,
) -> dict[str, float]:
    """Return raw, normalized, weighted, and aggregate base score components."""
    immunogenicity_term = max(0.0, min(float(immunogenicity), 1.0))
    esm2_term = normalize_esm2_log_likelihood(esm2_mean_log_likelihood)
    toxicity_term = normalize_toxicity(toxicity)

    weighted_immunogenicity = 0.50 * immunogenicity_term
    weighted_esm2 = 0.35 * esm2_term
    weighted_toxicity = 0.15 * toxicity_term
    base_score = weighted_immunogenicity + weighted_esm2 + weighted_toxicity

    return {
        "immunogenicity_term": immunogenicity_term,
        "esm2_term": esm2_term,
        "toxicity_term": toxicity_term,
        "weighted_immunogenicity": weighted_immunogenicity,
        "weighted_esm2": weighted_esm2,
        "weighted_toxicity": weighted_toxicity,
        "base_score": base_score,
    }


def hydrophobic_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    hydrophobic_count = sum(1 for residue in seq if residue in HYDROPHOBIC_RESIDUES)
    return hydrophobic_count / len(seq)


def hydrophobicity_penalty(
    seq: str,
    max_fraction: float = HYDROPHOBICITY_MAX_FRACTION,
) -> float:
    """
    Penalty in [0, 1] that activates when hydrophobic fraction exceeds max_fraction.
    """
    threshold = max(0.0, min(float(max_fraction), 1.0))
    observed = hydrophobic_fraction(seq)
    if observed <= threshold:
        return 0.0
    if threshold >= 1.0:
        return 0.0
    overflow = (observed - threshold) / (1.0 - threshold)
    return max(0.0, min(overflow, 1.0))

def has_any_target_epitope(seq: str, target_epitopes: set[str]) -> bool:
    return any(epitope in seq for epitope in target_epitopes)


def _peptide_one_hot_embedding(peptide: str) -> list[float]:
    vectors: list[float] = []
    for residue in peptide:
        vectors.extend(1.0 if residue == aa else 0.0 for aa in AMINO_ACID_ALPHABET)
    return vectors


def _cosine_similarity(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or not values_a:
        return 0.0

    dot = sum(a * b for a, b in zip(values_a, values_b))
    norm_a = math.sqrt(sum(a * a for a in values_a))
    norm_b = math.sqrt(sum(b * b for b in values_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _max_window_cosine_similarity(seq: str, epitope: str) -> float:
    if not epitope or len(seq) < len(epitope):
        return 0.0

    epitope_embed = _peptide_one_hot_embedding(epitope)
    max_similarity = 0.0
    epitope_len = len(epitope)

    for start in range(0, len(seq) - epitope_len + 1):
        window = seq[start:start + epitope_len]
        similarity = _cosine_similarity(_peptide_one_hot_embedding(window), epitope_embed)
        if similarity > max_similarity:
            max_similarity = similarity
    return max_similarity


def soft_epitope_reward(
    seq: str,
    target_epitopes: set[str],
    alpha: float = SOFT_EPITOPE_ALPHA,
) -> float:
    if not target_epitopes:
        return 1.0
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    max_similarity = max(_max_window_cosine_similarity(seq, epitope) for epitope in target_epitopes)
    numerator = math.exp(alpha * max_similarity) - 1.0
    denominator = math.exp(alpha) - 1.0
    return numerator / denominator if denominator > 0 else 0.0


def _generate_peptides(seq: str, lengths: Iterable[int]) -> List[str]:
    peptides: List[str] = []
    for length in lengths:
        if len(seq) < length:
            continue
        for start in range(0, len(seq) - length + 1):
            peptides.append(seq[start:start + length])
    return peptides


def _immunogenicity_prediction_alleles(supported_alleles: list[str]) -> list[str]:
    """Prefer the most frequent supported alleles to avoid unnecessary MHC predictions."""
    freq_map = parse_allele_frequencies_env()
    if not freq_map:
        freq_map = default_allele_frequencies(supported_alleles)

    if not freq_map:
        return supported_alleles

    supported_set = set(supported_alleles)
    ranked = sorted(
        (
            (allele, max(0.0, float(freq)))
            for allele, freq in freq_map.items()
            if allele in supported_set
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if not ranked:
        return supported_alleles

    coverage_target = float(os.getenv("IMMUNOGENICITY_ALLELE_COVERAGE", IMMUNOGENICITY_ALLELE_COVERAGE))
    coverage_target = max(0.0, min(coverage_target, 1.0))
    max_alleles = int(os.getenv("IMMUNOGENICITY_MAX_ALLELES", str(IMMUNOGENICITY_MAX_ALLELES)))
    max_alleles = max(1, max_alleles)

    total = sum(freq for _, freq in ranked)
    if total <= 0:
        return supported_alleles

    selected: list[str] = []
    cumulative = 0.0
    for allele, freq in ranked:
        selected.append(allele)
        cumulative += freq
        if len(selected) >= max_alleles:
            break
        if cumulative / total >= coverage_target:
            break
    return selected or supported_alleles


@lru_cache(maxsize=4096)
def predict_immunogenicity(seq: str) -> float:
    peptides = sorted(set(_generate_peptides(seq, lengths=range(8, 12))))
    if not peptides:
        return 0.0

    supported_alleles = mhcflurry_supported_alleles()
    alleles = _immunogenicity_prediction_alleles(supported_alleles)
    kd_by_allele = predict_mhcflurry_kd_matrix(peptides, alleles=alleles)
    if not kd_by_allele:
        return 0.0

    freq_map = parse_allele_frequencies_env()
    if not freq_map:
        freq_map = default_allele_frequencies(alleles)
    return immunogenicity_from_kd_matrix(kd_by_allele, allele_frequencies=freq_map)

def predict_esm2_sequence_log_likelihood(seq: str) -> float:
    return predict_esm2_mean_log_likelihood(seq)


def _toxicity_windows(seq: str, window_size: int = TOXICITY_WINDOW_SIZE) -> List[str]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if len(seq) <= window_size:
        return [seq]
    return [seq[i:i + window_size] for i in range(0, len(seq) - window_size + 1)]


def toxicity_softmax_score(window_toxicities: List[float], beta: float = TOXICITY_BETA) -> float:
    if not window_toxicities:
        return 0.0
    if beta <= 0:
        raise ValueError("beta must be > 0")

    tox = [max(0.0, min(float(t), 1.0)) for t in window_toxicities]
    scaled = [beta * t for t in tox]
    m = max(scaled)
    logsumexp = m + math.log(sum(math.exp(x - m) for x in scaled))
    raw = logsumexp / beta

    # Normalize away dependence on number of windows so the score remains [0, 1].
    normalization_offset = math.log(len(tox)) / beta
    normalized = raw - normalization_offset
    return max(0.0, min(normalized, 1.0))


def predict_toxicity(
    seq: str,
    window_size: int = TOXICITY_WINDOW_SIZE,
    beta: float = TOXICITY_BETA,
) -> float:
    windows = _toxicity_windows(seq, window_size=window_size)
    window_toxicities = _toxicity_windows_cached_scores(windows)
    return toxicity_softmax_score(window_toxicities, beta=beta)


def score_antigen_candidate_with_breakdown(
    seq: str,
    target_epitopes: set | None = None,
) -> dict[str, float]:
    immunogenicity = predict_immunogenicity(seq)
    esm2_sequence_log_likelihood = predict_esm2_sequence_log_likelihood(seq)
    toxicity = predict_toxicity(seq)
    components = antigen_score_components(
        immunogenicity,
        esm2_sequence_log_likelihood,
        toxicity,
    )
    base = components["base_score"]
    hydrophobicity_threshold = float(
        os.getenv("HYDROPHOBICITY_MAX_FRACTION", str(HYDROPHOBICITY_MAX_FRACTION))
    )
    hydrophobicity_weight = max(
        0.0,
        float(os.getenv("HYDROPHOBICITY_PENALTY_WEIGHT", str(HYDROPHOBICITY_PENALTY_WEIGHT))),
    )
    observed_hydrophobicity = hydrophobic_fraction(seq)
    hydrophobic_penalty = hydrophobicity_penalty(seq, max_fraction=hydrophobicity_threshold)
    weighted_hydrophobic_penalty = hydrophobicity_weight * hydrophobic_penalty
    total_score = BASE_SCORE_WEIGHT * base - weighted_hydrophobic_penalty

    return {
        "score": total_score,
        "immunogenicity": immunogenicity,
        "esm2_sequence_log_likelihood": esm2_sequence_log_likelihood,
        "toxicity": toxicity,
        "hydrophobic_fraction": observed_hydrophobicity,
        "hydrophobicity_threshold": hydrophobicity_threshold,
        "hydrophobicity_penalty": hydrophobic_penalty,
        "weighted_hydrophobicity_penalty": weighted_hydrophobic_penalty,
        **components,
    }


def score_antigen_candidate(seq: str, target_epitopes: set | None = None) -> float:
    return score_antigen_candidate_with_breakdown(seq, target_epitopes)["score"]
