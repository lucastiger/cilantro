import inspect

import pytest

from optimize.cma_latent_search import linear_weight_schedule, latent_optimize


def test_latent_optimize_uses_annealing_by_default():
    signature = inspect.signature(latent_optimize)
    assert signature.parameters["weight_schedule"].default is linear_weight_schedule


def test_linear_weight_schedule_normalizes_to_one():
    weights = linear_weight_schedule(0, 100)
    assert pytest.approx(sum(weights), rel=1e-6) == 1.0


def test_linear_weight_schedule_moves_toward_epitope_focus():
    start_base, start_soft, start_hard = linear_weight_schedule(0, 100)
    end_base, end_soft, end_hard = linear_weight_schedule(99, 100)

    assert start_base > end_base
    assert start_soft < end_soft
    assert start_hard == end_hard


def test_linear_weight_schedule_single_generation_uses_end_weights():
    base, soft, hard = linear_weight_schedule(0, 1, base_end=0.2, soft_end=0.8, hard_end=0.0)

    assert pytest.approx(base, rel=1e-6) == 0.2
    assert pytest.approx(soft, rel=1e-6) == 0.8
    assert pytest.approx(hard, rel=1e-6) == 0.0
