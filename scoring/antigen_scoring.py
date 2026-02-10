# scoring/antigen_scoring.py
import math
from typing import Iterable, List

from scoring.prediction_tools import (
    predict_folding_energy_foldx,
    predict_mhci_ic50s,
    predict_plddt_from_esmfold,
    predict_toxicity_external,
)
IC50_BEST_NM = 1.0
IC50_WORST_NM = 50000.0
PLDDT_BEST = 90.0
PLDDT_WORST = 50.0
FOLDING_BEST = -15.0
FOLDING_WORST = 0.0
TOXICITY_WORST = 0.3


def normalize_ic50(ic50_nm: float) -> float:
    ic50_clamped = max(IC50_BEST_NM, min(ic50_nm, IC50_WORST_NM))
    log_best = math.log10(IC50_BEST_NM)
    log_worst = math.log10(IC50_WORST_NM)
    log_val = math.log10(ic50_clamped)
    return max(0.0, min(1.0, 1.0 - (log_val - log_best) / (log_worst - log_best)))


def normalize_plddt(plddt: float) -> float:
    if plddt <= PLDDT_WORST:
        return 0.0
    if plddt >= PLDDT_BEST:
        return 1.0
    return (plddt - PLDDT_WORST) / (PLDDT_BEST - PLDDT_WORST)


def normalize_folding_energy(folding_energy: float) -> float:
    if folding_energy <= FOLDING_BEST:
        return 1.0
    if folding_energy >= FOLDING_WORST:
        return 0.0
    return (FOLDING_WORST - folding_energy) / (FOLDING_WORST - FOLDING_BEST)


def normalize_toxicity(toxicity: float) -> float:
    toxicity_clamped = max(0.0, min(toxicity, 1.0))
    if toxicity_clamped >= TOXICITY_WORST:
        return 0.0
    return 1.0 - toxicity_clamped / TOXICITY_WORST


def antigen_score(ic50, plddt, folding_energy, toxicity):
    """
    Weighted antigen score:
    0.50 IC50
    0.20 pLDDT
    0.15 folding energy
    0.15 toxicity
    """
    
    # Normalize each term to [0,1]
    ic50_term = normalize_ic50(ic50)
    plddt_term = normalize_plddt(plddt)
    folding_term = normalize_folding_energy(folding_energy)
    toxicity_term = normalize_toxicity(toxicity)

    return (
        0.50 * ic50_term +
        0.20 * plddt_term +
        0.15 * folding_term +
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


def predict_ic50(seq: str) -> float:
    peptides = _generate_peptides(seq, lengths=range(8, 12))
    if not peptides:
        return IC50_WORST_NM
    predictions = predict_mhci_ic50s(peptides)
    if not predictions:
        return IC50_WORST_NM
    return min(predictions.values())

def predict_plddt(seq: str) -> float:
    return predict_plddt_from_esmfold(seq)

def predict_folding_energy(seq: str) -> float:
    return predict_folding_energy_foldx(seq)

def predict_toxicity(seq: str) -> float:
    return predict_toxicity_external(seq)
    
def score_antigen_candidate(seq: str, target_epitopes: set) -> float:
    """
    HARD CONSTRAINT:
    If no target epitope is present → score = 0
    """
    ep_score = epitope_presence_score(seq, target_epitopes)
    if ep_score == 0.0:
        return 0.0

    base = antigen_score(
        predict_ic50(seq),
        predict_plddt(seq),
        predict_folding_energy(seq),
        predict_toxicity(seq),
    )

    return 0.7 * base + 0.3 * ep_score
