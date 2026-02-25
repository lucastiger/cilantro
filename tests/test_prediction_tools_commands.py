from __future__ import annotations

import scoring.prediction_tools as prediction_tools


class _FakeMHCFlurryPredictor:
    supported_alleles = ["HLA-A*02:01", "HLA-B*07:02"]

    def predict(self, peptides, allele, verbose=None):
        assert allele in self.supported_alleles
        assert verbose == 0
        return [10.0 + i for i, _ in enumerate(peptides)]


def test_mhcflurry_kd_matrix(monkeypatch):
    monkeypatch.setattr(prediction_tools, "_mhcflurry_predictor", lambda: _FakeMHCFlurryPredictor())

    out = prediction_tools.predict_mhcflurry_kd_matrix(["SIINFEKL", "LLFGYPVYV"], alleles=["A*02:01"])

    assert list(out.keys()) == ["HLA-A*02:01"]
    assert out["HLA-A*02:01"]["SIINFEKL"] == 10.0
    assert out["HLA-A*02:01"]["LLFGYPVYV"] == 11.0




def test_mhcflurry_kd_matrix_skips_unsupported_lengths(monkeypatch):
    monkeypatch.setattr(prediction_tools, "_mhcflurry_predictor", lambda: _FakeMHCFlurryPredictor())

    out = prediction_tools.predict_mhcflurry_kd_matrix(
        ["SIINFEKL", "A" * 20],
        alleles=["A*02:01"],
    )

    assert list(out.keys()) == ["HLA-A*02:01"]
    assert list(out["HLA-A*02:01"].keys()) == ["SIINFEKL"]


def test_mhcflurry_kd_matrix_returns_empty_when_all_unsupported(monkeypatch):
    monkeypatch.setattr(prediction_tools, "_mhcflurry_predictor", lambda: _FakeMHCFlurryPredictor())

    out = prediction_tools.predict_mhcflurry_kd_matrix(["A" * 25], alleles=["A*02:01"])

    assert out == {}


def test_mhcflurry_kd_matrix_falls_back_when_predictor_lacks_verbose(monkeypatch):
    class _LegacyPredictor:
        supported_alleles = ["HLA-A*02:01"]

        def predict(self, peptides, allele):
            assert allele == "HLA-A*02:01"
            return [42.0 for _ in peptides]

    monkeypatch.setattr(prediction_tools, "_mhcflurry_predictor", lambda: _LegacyPredictor())

    out = prediction_tools.predict_mhcflurry_kd_matrix(["SIINFEKL"], alleles=["A*02:01"])
    assert out == {"HLA-A*02:01": {"SIINFEKL": 42.0}}


def test_mhcflurry_kd_matrix_suppresses_predictor_progress_output(monkeypatch, capsys):
    class _NoisyLegacyPredictor:
        supported_alleles = ["HLA-A*02:01"]

        def predict(self, peptides, allele):
            print("2/2 [==============================] - 0s 17ms/step")
            return [7.0 for _ in peptides]

    monkeypatch.setattr(prediction_tools, "_mhcflurry_predictor", lambda: _NoisyLegacyPredictor())

    out = prediction_tools.predict_mhcflurry_kd_matrix(["SIINFEKL"], alleles=["A*02:01"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert out == {"HLA-A*02:01": {"SIINFEKL": 7.0}}

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


def test_toxicity_batch_toxinpred3_command(monkeypatch):
    monkeypatch.setattr(prediction_tools, "TOXICITY_PREDICTOR_CMD", "toxinpred3")

    def fake_run(command, check, stdout, stderr, text, timeout):
        assert command[0] == "toxinpred3"
        out_path = command[command.index("-o") + 1]
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write("Hybrid Score\n0.43\n0.0\n")
        return type("Result", (), {"stdout": "", "stderr": ""})

    monkeypatch.setattr(prediction_tools.subprocess, "run", fake_run)

    out = prediction_tools.predict_toxicity_external_batch(["ACDEFGHIKLMNPQ", "RRFFLLRRFF"])
    assert out == [0.43, 0.0]
