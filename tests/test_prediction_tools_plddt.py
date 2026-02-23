from __future__ import annotations

import scoring.prediction_tools as prediction_tools


def _atom_line(atom_id: int, atom_name: str, residue: str, residue_id: int, b_factor: float) -> str:
    return (
        f"ATOM  {atom_id:5d} {atom_name:<4}{residue:>3} A{residue_id:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.00:6.2f}{b_factor:6.2f}           C"
    )


def test_residue_plddt_from_pdb_groups_atoms_per_residue():
    pdb_text = "\n".join(
        [
            _atom_line(1, "N", "ALA", 1, 70.0),
            _atom_line(2, "CA", "ALA", 1, 70.0),
            _atom_line(3, "N", "GLY", 2, 80.0),
            _atom_line(4, "CA", "GLY", 2, 80.0),
        ]
    )

    residue_scores = prediction_tools.residue_plddt_from_pdb(pdb_text)

    assert residue_scores == [70.0, 80.0]


def test_plddt_measurements_from_esmfold(monkeypatch):
    pdb_text = "\n".join(
        [
            _atom_line(1, "N", "ALA", 1, 65.0),
            _atom_line(2, "CA", "ALA", 1, 75.0),
            _atom_line(3, "N", "GLY", 2, 85.0),
            _atom_line(4, "CA", "GLY", 2, 95.0),
        ]
    )

    monkeypatch.setattr(prediction_tools, "fetch_esmfold_pdb", lambda sequence, timeout=120: pdb_text)

    out = prediction_tools.plddt_measurements_from_esmfold("MKTAYIAKQ")

    assert out["per_residue_plddt"] == [70.0, 90.0]
    assert out["mean_plddt"] == 80.0


def test_fetch_esmfold_pdb_local_backend(monkeypatch):
    monkeypatch.setattr(prediction_tools, "PLDDT_PREDICTOR_CMD", "")
    monkeypatch.setattr(prediction_tools, "ESMFOLD_BACKEND", "local")
    monkeypatch.setattr(prediction_tools, "_infer_local_esmfold_pdb", lambda sequence: "ATOM")

    assert prediction_tools.fetch_esmfold_pdb("MKTAYIAKQ") == "ATOM"
