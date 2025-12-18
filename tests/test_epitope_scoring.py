# tests/test_epitope_scoring.py
from scoring.epitope_scoring import shannon_entropy
from collections import Counter
def test_entropy_zero():
    c = Counter({"A":10})
    assert abs(shannon_entropy(c) - 0.0) < 1e-6
def test_entropy_uniform():
    c = Counter({a:1 for a in "ACDEFGHIKLMNPQRSTVWY"})
    assert 0.9 < shannon_entropy(c) < 1.01
