# scoring/prediction_tools.py
from __future__ import annotations

import csv
import io
import json
import os
import shlex
import subprocess
import tempfile
from functools import lru_cache
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

# Approximate global class-I HLA allele frequencies (used as defaults when
# MHC_ALLELE_FREQUENCIES is not provided). Values are treated as relative
# weights and are renormalized during scoring.
DEFAULT_MHC_ALLELE_FREQUENCIES = {
    "HLA-A*01:01": 0.165,
    "HLA-A*02:01": 0.259,
    "HLA-A*02:06": 0.020,
    "HLA-A*03:01": 0.122,
    "HLA-A*11:01": 0.151,
    "HLA-A*23:01": 0.040,
    "HLA-A*24:02": 0.171,
    "HLA-A*26:01": 0.021,
    "HLA-A*29:02": 0.031,
    "HLA-A*30:01": 0.033,
    "HLA-A*30:02": 0.029,
    "HLA-A*31:01": 0.039,
    "HLA-A*32:01": 0.022,
    "HLA-A*33:03": 0.035,
    "HLA-A*68:01": 0.036,
    "HLA-A*68:02": 0.028,
    "HLA-B*07:02": 0.112,
    "HLA-B*08:01": 0.098,
    "HLA-B*15:01": 0.049,
    "HLA-B*35:01": 0.067,
    "HLA-B*40:01": 0.041,
    "HLA-B*44:02": 0.061,
    "HLA-B*44:03": 0.050,
    "HLA-B*51:01": 0.046,
    "HLA-B*53:01": 0.025,
    "HLA-B*57:01": 0.022,
    "HLA-B*58:01": 0.026,
    "HLA-C*03:04": 0.080,
    "HLA-C*04:01": 0.121,
    "HLA-C*05:01": 0.049,
    "HLA-C*06:02": 0.080,
    "HLA-C*07:01": 0.110,
    "HLA-C*07:02": 0.150,
    "HLA-C*08:02": 0.020,
    "HLA-C*12:03": 0.070,
    "HLA-C*14:02": 0.021,
    "HLA-C*15:02": 0.020,
}


def _normalize_allele_name(allele: str) -> str:
    normalized = allele.strip()
    if not normalized:
        return normalized
    if normalized.upper().startswith("HLA-"):
        return normalized
    return f"HLA-{normalized}"


@lru_cache(maxsize=1)
def _mhcflurry_predictor():
    from mhcflurry import Class1AffinityPredictor

    return Class1AffinityPredictor.load()



def default_allele_frequencies(alleles: Iterable[str] | None = None) -> Dict[str, float]:
    known = {
        _normalize_allele_name(allele): float(freq)
        for allele, freq in DEFAULT_MHC_ALLELE_FREQUENCIES.items()
        if float(freq) > 0
    }
    if alleles is None:
        return known

    requested = {_normalize_allele_name(a) for a in alleles}
    return {allele: freq for allele, freq in known.items() if allele in requested}


def parse_allele_frequencies_env() -> Dict[str, float]:
    """
    Optional allele frequency map from JSON string in MHC_ALLELE_FREQUENCIES.
    Example:
    {
      "HLA-A*02:01": 0.15,
      "HLA-B*07:02": 0.10
    }
    """
    raw = os.getenv("MHC_ALLELE_FREQUENCIES", "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MHC_ALLELE_FREQUENCIES is not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("MHC_ALLELE_FREQUENCIES must be a JSON object mapping allele->frequency.")

    out: Dict[str, float] = {}
    for allele, freq in parsed.items():
        try:
            freq_val = float(freq)
        except (TypeError, ValueError):
            continue
        if freq_val > 0:
            out[_normalize_allele_name(str(allele))] = freq_val
    return out


def mhcflurry_supported_alleles() -> List[str]:
    predictor = _mhcflurry_predictor()
    alleles = sorted({_normalize_allele_name(a) for a in predictor.supported_alleles})
    return [a for a in alleles if a]


def predict_mhcflurry_kd_matrix(
    peptides: Iterable[str],
    alleles: Iterable[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    peptides_list = [pep for pep in peptides if pep]
    if not peptides_list:
        return {}

    predictor = _mhcflurry_predictor()
    supported = {_normalize_allele_name(a) for a in predictor.supported_alleles}

    requested_alleles = (
        [_normalize_allele_name(a) for a in alleles]
        if alleles is not None
        else sorted(supported)
    )
    filtered_alleles = [a for a in requested_alleles if a in supported]
    if not filtered_alleles:
        return {}

    result: Dict[str, Dict[str, float]] = {}
    for allele in filtered_alleles:
        affinities = predictor.predict(peptides=peptides_list, allele=allele)
        result[allele] = {
            peptide: float(kd) for peptide, kd in zip(peptides_list, affinities)
        }
    return result


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
