from __future__ import annotations

import scoring.prediction_tools as prediction_tools


def test_mhci_command_path(monkeypatch):
    monkeypatch.setattr(prediction_tools, "MHCI_PREDICTOR_CMD", "mhci_tool")

    def fake_run(command, input, check, stdout, stderr, text, timeout):
        assert command == ["mhci_tool"]
        assert "SIINFEKL" in input
        return type("Result", (), {"stdout": "peptide\tic50\nSIINFEKL\t31.5\n", "stderr": ""})

    monkeypatch.setattr(prediction_tools.subprocess, "run", fake_run)
    out = prediction_tools.predict_mhci_ic50s(["SIINFEKL"])
    assert out["SIINFEKL"] == 31.5


def test_plddt_and_folding_command_paths(monkeypatch):
    monkeypatch.setattr(prediction_tools, "PLDDT_PREDICTOR_CMD", "plddt_tool")
    monkeypatch.setattr(prediction_tools, "FOLDING_PREDICTOR_CMD", "fold_tool")

    def fake_run(command, input, check, stdout, stderr, text, timeout):
        if command == ["plddt_tool"]:
            return type("Result", (), {"stdout": "78.3\n", "stderr": ""})
        if command == ["fold_tool"]:
            return type("Result", (), {"stdout": "-8.1\n", "stderr": ""})
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(prediction_tools.subprocess, "run", fake_run)

    assert prediction_tools.predict_plddt_from_esmfold("MKTAYIAKQ") == 78.3
    assert prediction_tools.predict_folding_energy_foldx("MKTAYIAKQ") == -8.1


def test_esm2_likelihood_command_path(monkeypatch):
    monkeypatch.setattr(prediction_tools, "ESM2_LIKELIHOOD_CMD", "esm2_ll_tool")

    def fake_run(command, input, check, stdout, stderr, text, timeout):
        assert command == ["esm2_ll_tool"]
        assert input == "MKTAYIAKQ"
        return type("Result", (), {"stdout": "-1.23\n", "stderr": ""})

    monkeypatch.setattr(prediction_tools.subprocess, "run", fake_run)

    assert prediction_tools.predict_esm2_mean_log_likelihood("MKTAYIAKQ") == -1.23


def test_toxicity_command_not_found(monkeypatch):
    monkeypatch.setattr(prediction_tools, "TOXICITY_PREDICTOR_CMD", "toxdl")

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("toxdl")

    monkeypatch.setattr(prediction_tools.subprocess, "run", fake_run)

    try:
        prediction_tools.predict_toxicity_external("MKTAYIAKQ")
    except RuntimeError as exc:
        assert "Command not found" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
