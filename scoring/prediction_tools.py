# scoring/prediction_tools.py
from __future__ import annotations

import csv
import io
import os
import shlex
import subprocess
import tempfile
from typing import Dict, Iterable, List

import requests

MHC_ALLELE_DEFAULT = os.getenv("MHC_ALLELE", "HLA-A*02:01")
IEDB_MHCI_URL = os.getenv(
    "IEDB_MHCI_URL",
    "http://tools-cluster-interface.iedb.org/tools_api/mhci/",
)
IEDB_MHCI_METHOD = os.getenv("IEDB_MHCI_METHOD", "netmhcpan")
ESMFOLD_URL = os.getenv(
    "ESMFOLD_URL",
    "https://api.esmatlas.com/foldSequence/v1/pdb/",
)

# Optional local command integrations (preferred when available)
# Expected output format for MHCI_PREDICTOR_CMD: TSV with columns including `peptide` and `ic50`.
MHCI_PREDICTOR_CMD = os.getenv("MHCI_PREDICTOR_CMD", "")
# Expected output format for PLDDT_PREDICTOR_CMD: either a numeric value or PDB text with B-factors.
PLDDT_PREDICTOR_CMD = os.getenv("PLDDT_PREDICTOR_CMD", "")
# Expected output format for FOLDING_PREDICTOR_CMD: numeric folding energy value.
FOLDING_PREDICTOR_CMD = os.getenv("FOLDING_PREDICTOR_CMD", "")
FOLDX_BIN = os.getenv("FOLDX_BIN", "foldx")
TOXICITY_PREDICTOR_CMD = os.getenv(
    "TOXICITY_PREDICTOR_CMD",
    "toxdl predict --stdin",
)


def _parse_iedb_mhci_response(text: str) -> Dict[str, float]:
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    rows = [row for row in reader if row]
    if not rows:
        raise RuntimeError("MHC-I predictor response was empty.")

    header = [h.strip().lower() for h in rows[0]]
    if "peptide" not in header or "ic50" not in header:
        raise RuntimeError(f"Unexpected MHC-I response header: {rows[0]}")

    peptide_idx = header.index("peptide")
    ic50_idx = header.index("ic50")
    predictions = {}
    for row in rows[1:]:
        if len(row) <= max(peptide_idx, ic50_idx):
            continue
        peptide = row[peptide_idx].strip()
        try:
            ic50_val = float(row[ic50_idx])
        except ValueError:
            continue
        if peptide:
            predictions[peptide] = ic50_val
    return predictions


def _run_command(command_str: str, stdin_text: str, timeout: int) -> str:
    command = shlex.split(command_str)
    if not command:
        raise RuntimeError("Configured command is empty after parsing.")
    try:
        result = subprocess.run(
            command,
            input=stdin_text,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise RuntimeError(f"Command failed ({' '.join(command)}): {stderr}") from exc
    return result.stdout


def _post_no_proxy(url: str, data, timeout: int) -> requests.Response:
    session = requests.Session()
    # proxy in this environment can block external tool hosts; try direct network first.
    session.trust_env = False
    try:
        response = session.post(url, data=data, timeout=timeout)
        response.raise_for_status()
        return response
    except Exception:
        # Fallback to environment proxy settings.
        response = requests.post(url, data=data, timeout=timeout)
        response.raise_for_status()
        return response


def predict_mhci_ic50s(
    peptides: Iterable[str],
    allele: str | None = None,
    method: str | None = None,
    url: str | None = None,
    timeout: int = 60,
) -> Dict[str, float]:
    peptides_list = [pep for pep in peptides if pep]
    if not peptides_list:
        return {}

    if MHCI_PREDICTOR_CMD:
        input_payload = "\n".join(peptides_list)
        output = _run_command(MHCI_PREDICTOR_CMD, input_payload, timeout=timeout)
        return _parse_iedb_mhci_response(output)

    payload = {
        "method": method or IEDB_MHCI_METHOD,
        "sequence_text": "\n".join(peptides_list),
        "allele": allele or MHC_ALLELE_DEFAULT,
    }
    response = _post_no_proxy(url or IEDB_MHCI_URL, data=payload, timeout=timeout)
    return _parse_iedb_mhci_response(response.text)


def fetch_esmfold_pdb(sequence: str, timeout: int = 120) -> str:
    if not sequence:
        raise ValueError("Sequence must not be empty.")

    if PLDDT_PREDICTOR_CMD:
        out = _run_command(PLDDT_PREDICTOR_CMD, sequence, timeout=timeout)
        # if command returns PDB text, use directly; if numeric, we cannot build structure.
        if "ATOM" in out:
            return out
        raise RuntimeError(
            "PLDDT_PREDICTOR_CMD returned numeric-only output; "
            "fetch_esmfold_pdb requires PDB text."
        )

    response = _post_no_proxy(ESMFOLD_URL, data=sequence, timeout=timeout)
    return response.text


def average_plddt_from_pdb(pdb_text: str) -> float:
    b_factors: List[float] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            b_factor = float(line[60:66].strip())
        except ValueError:
            continue
        b_factors.append(b_factor)
    if not b_factors:
        raise RuntimeError("No pLDDT values found in PDB output.")
    return sum(b_factors) / len(b_factors)


def predict_plddt_from_esmfold(sequence: str, timeout: int = 120) -> float:
    if PLDDT_PREDICTOR_CMD:
        out = _run_command(PLDDT_PREDICTOR_CMD, sequence, timeout=timeout).strip()
        try:
            return float(out)
        except ValueError:
            return average_plddt_from_pdb(out)

    pdb_text = fetch_esmfold_pdb(sequence, timeout=timeout)
    return average_plddt_from_pdb(pdb_text)


def predict_folding_energy_foldx(sequence: str, timeout: int = 300) -> float:
    if FOLDING_PREDICTOR_CMD:
        out = _run_command(FOLDING_PREDICTOR_CMD, sequence, timeout=timeout).strip()
        try:
            return float(out)
        except ValueError as exc:
            raise RuntimeError("FOLDING_PREDICTOR_CMD did not return a numeric value.") from exc

    pdb_text = fetch_esmfold_pdb(sequence, timeout=timeout)
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "model.pdb")
        with open(pdb_path, "w", encoding="utf-8") as handle:
            handle.write(pdb_text)

        command = [
            FOLDX_BIN,
            "--command=Stability",
            f"--pdb={os.path.basename(pdb_path)}",
            f"--output-dir={tmpdir}",
        ]
        try:
            subprocess.run(
                command,
                cwd=tmpdir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"FoldX binary '{FOLDX_BIN}' not found. Set FOLDX_BIN to a valid executable."
            ) from exc

        summary_path = os.path.join(tmpdir, "Summary_model.fxout")
        if not os.path.exists(summary_path):
            raise RuntimeError("FoldX did not produce Summary_model.fxout.")

        with open(summary_path, encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        if len(lines) < 2:
            raise RuntimeError("FoldX output missing summary values.")

        header = [col.strip().lower() for col in lines[0].split("\t")]
        values = lines[1].split("\t")
        if "total energy" not in header:
            raise RuntimeError("FoldX summary missing Total Energy column.")
        idx = header.index("total energy")
        return float(values[idx])


def predict_toxicity_external(sequence: str, timeout: int = 120) -> float:
    if not sequence:
        raise ValueError("Sequence must not be empty.")

    out = _run_command(TOXICITY_PREDICTOR_CMD, sequence, timeout=timeout).strip()
    try:
        return float(out)
    except ValueError as exc:
        raise RuntimeError("Toxicity predictor did not return a numeric value.") from exc
