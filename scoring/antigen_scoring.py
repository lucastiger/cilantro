# scoring/antigen_scoring.py
import math

def antigen_score(ic50, plddt, folding_energy, toxicity):
    """
    Weighted antigen score:
    0.50 IC50
    0.20 pLDDT
    0.15 folding energy
    0.15 toxicity
    """
    
    # Normalize each term to [0,1]
    ic50_term = max(0.0, 1.0 - ic50 / 50.0)        # strong binders <50 nM
    plddt_term = min(plddt / 100.0, 1.0)
    folding_term = max(0.0, min(1.0, -folding_energy / 10.0))
    toxicity_term = max(0.0, 1.0 - toxicity)

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


def predict_ic50(seq):
    # integrate mhcflurry or NetMHCpan here
    return 50.0

def predict_plddt(seq):
    # call AlphaFold/ColabFold; for now safe default
    return 80.0

def predict_folding_energy(seq):
    return -8.0

def predict_toxicity(seq):
    return 0.05
    
def score_antigen_candidate(seq: str, target_epitopes: set) -> float:
    """
    HARD CONSTRAINT:
    If no target epitope is present → score = 0
    """
    ep_score = epitope_presence_score(seq, target_epitopes)
    if ep_score == 0.0:
        return 0.0

    base = antigen_base_score(
        predict_ic50(seq),
        predict_plddt(seq),
        predict_folding_energy(seq),
        predict_toxicity(seq),
    )

    return 0.7 * base + 0.3 * ep_score


