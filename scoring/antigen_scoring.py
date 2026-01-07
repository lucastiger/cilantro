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
    
def score_antigen_candidate(seq):
    return antigen_score(
        predict_ic50(seq),
        predict_plddt(seq),
        predict_folding_energy(seq),
        predict_toxicity(seq),
    )

