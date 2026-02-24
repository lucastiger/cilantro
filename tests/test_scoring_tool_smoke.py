from __future__ import annotations

import math
import os
import shlex
import shutil
from urllib.error import URLError

import pytest

from scoring.prediction_tools import (
    ESM2_LIKELIHOOD_CMD,
    TOXICITY_PREDICTOR_CMD,
    mhcflurry_supported_alleles,
    predict_esm2_mean_log_likelihood,
    predict_mhcflurry_kd_matrix,
    predict_toxicity_external_batch,
)


SMOKE_FLAG = "RUN_SCORING_TOOL_SMOKE_TESTS"


def _require_smoke_flag() -> None:
    if os.getenv(SMOKE_FLAG, "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip(
            f"Set {SMOKE_FLAG}=1 to run real external/model smoke tests for scoring tools."
        )


def _require_command_available(command_str: str) -> None:
    command = shlex.split(command_str)
    if not command:
        pytest.skip("Command is empty.")
    executable = command[0]
    if shutil.which(executable) is None:
        pytest.skip(f"Command not available on PATH: {executable}")


@pytest.mark.smoke
@pytest.mark.filterwarnings(
    "ignore:Transparent hugepages are not enabled:UserWarning:jax._src.cloud_tpu_init"
)
@pytest.mark.filterwarnings(
    "ignore:Downcasting behavior in `replace` is deprecated:FutureWarning:mhcflurry.amino_acid"
)
def test_mhcflurry_real_predictor_smoke():
    _require_smoke_flag()

    try:
        alleles = mhcflurry_supported_alleles()
    except RuntimeError as exc:
        pytest.skip(f"MHCflurry model assets not available: {exc}")

    if not alleles:
        pytest.skip("No supported MHCflurry alleles found.")

    selected = "HLA-A*02:01" if "HLA-A*02:01" in alleles else alleles[0]
    kd_by_allele = predict_mhcflurry_kd_matrix(["SIINFEKL"], alleles=[selected])

    assert selected in kd_by_allele
    score = kd_by_allele[selected]["SIINFEKL"]
    assert isinstance(score, float)
    assert math.isfinite(score)
    assert score > 0


@pytest.mark.smoke
def test_toxinpred3_real_command_smoke():
    _require_smoke_flag()
    _require_command_available(TOXICITY_PREDICTOR_CMD)

    scores = predict_toxicity_external_batch(["MKTAYIAKQ", "GILGFVFTL"], timeout=180)

    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)
    assert all(0.0 <= score <= 1.0 for score in scores)


@pytest.mark.smoke
def test_esm2_log_likelihood_smoke():
    _require_smoke_flag()

    if ESM2_LIKELIHOOD_CMD:
        _require_command_available(ESM2_LIKELIHOOD_CMD)

    try:
        value = predict_esm2_mean_log_likelihood("MKTAYIAKQ")
    except (RuntimeError, URLError, OSError) as exc:
        pytest.skip(f"ESM-2 model/command unavailable in this environment: {exc}")

    assert isinstance(value, float)
    assert math.isfinite(value)
