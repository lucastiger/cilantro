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
ESMFOLD_BACKEND = os.getenv("ESMFOLD_BACKEND", "local").strip().lower()

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
    "toxinpred3",
)
# Expected output format for ESM2_LIKELIHOOD_CMD: numeric mean log-likelihood per residue.
ESM2_LIKELIHOOD_CMD = os.getenv("ESM2_LIKELIHOOD_CMD", "")

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

    if ESMFOLD_BACKEND == "local":
        return _infer_local_esmfold_pdb(sequence)

    response = _post_no_proxy(ESMFOLD_URL, data=sequence, timeout=timeout)
    return response.text


@lru_cache(maxsize=1)
def _load_local_esmfold_model():
    try:
        import torch
        import esm
    except ImportError as exc:
        raise RuntimeError(
            "Local ESMFold backend requires optional dependencies: torch and fair-esm."
        ) from exc

    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, torch


def _infer_local_esmfold_pdb(sequence: str) -> str:
    model, torch = _load_local_esmfold_model()
    with torch.no_grad():
        return model.infer_pdb(sequence)



def residue_plddt_from_pdb(pdb_text: str) -> List[float]:
    residue_scores: List[float] = []
    current_residue = None
    current_scores: List[float] = []

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue

        try:
            residue_id = int(line[22:26].strip())
            b_factor = float(line[60:66].strip())
        except ValueError:
            continue

        if current_residue is None:
            current_residue = residue_id

        if residue_id != current_residue:
            residue_scores.append(sum(current_scores) / len(current_scores))
            current_scores = []
            current_residue = residue_id

        current_scores.append(b_factor)

    if current_scores:
        residue_scores.append(sum(current_scores) / len(current_scores))

    if not residue_scores:
        raise RuntimeError("No per-residue pLDDT values found in PDB output.")

    return residue_scores


def plddt_measurements_from_esmfold(sequence: str, timeout: int = 120) -> Dict[str, float | List[float]]:
    pdb_text = fetch_esmfold_pdb(sequence, timeout=timeout)
    per_residue = residue_plddt_from_pdb(pdb_text)
    return {
        "mean_plddt": sum(per_residue) / len(per_residue),
        "per_residue_plddt": per_residue,
    }


def predict_plddt_from_esmfold(sequence: str, timeout: int = 120) -> float:
    if PLDDT_PREDICTOR_CMD:
        out = _run_command(PLDDT_PREDICTOR_CMD, sequence, timeout=timeout).strip()
        try:
            return float(out)
        except ValueError:
            residue_scores = residue_plddt_from_pdb(out)
            return sum(residue_scores) / len(residue_scores)

    measurements = plddt_measurements_from_esmfold(sequence, timeout=timeout)
    return float(measurements["mean_plddt"])


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


def predict_toxicity_external_batch(
    sequences: Iterable[str],
    threshold: float = 0.38,
    model: int = 2,
    display: int = 2,
    timeout: int = 120,
) -> List[float]:
    seqs = [seq.strip() for seq in sequences if seq and seq.strip()]
    if not seqs:
        return []

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as in_handle:
        for seq in seqs:
            in_handle.write(f"{seq}\n")
        input_path = in_handle.name

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv", encoding="utf-8") as out_handle:
        output_path = out_handle.name

    command = shlex.split(TOXICITY_PREDICTOR_CMD)
    command.extend(
        [
            "-i",
            input_path,
            "-o",
            output_path,
            "-t",
            str(threshold),
            "-m",
            str(model),
            "-d",
            str(display),
        ]
    )

    try:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Command not found: {command[0]}") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"Command failed ({' '.join(command)}): {stderr}") from exc

        with open(output_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        if len(rows) != len(seqs):
            raise RuntimeError(
                "Toxicity predictor returned an unexpected number of rows. "
                f"Expected {len(seqs)}, got {len(rows)}."
            )

        scores: List[float] = []
        for row in rows:
            if "Hybrid Score" not in row:
                raise RuntimeError("Toxicity predictor output missing 'Hybrid Score' column.")
            try:
                score = float(row["Hybrid Score"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Toxicity predictor returned a non-numeric hybrid score.") from exc
            scores.append(max(0.0, min(score, 1.0)))
        return scores
    finally:
        for path in (input_path, output_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def predict_toxicity_external(sequence: str, timeout: int = 120) -> float:
    if not sequence:
        raise ValueError("Sequence must not be empty.")
    return predict_toxicity_external_batch([sequence], timeout=timeout)[0]


@lru_cache(maxsize=1)
def _load_local_esm2_model():
    try:
        import torch
        import esm
    except ImportError as exc:
        raise RuntimeError(
            "ESM-2 likelihood scoring requires optional dependencies: torch and fair-esm."
        ) from exc

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, device, torch


def predict_esm2_mean_log_likelihood(sequence: str) -> float:
    """
    Returns the mean per-residue log-likelihood under ESM-2.

    Larger (less negative) values indicate a sequence that is more probable
    under the ESM-2 protein language model.
    """
    if not sequence:
        raise ValueError("Sequence must not be empty.")

    if ESM2_LIKELIHOOD_CMD:
        out = _run_command(ESM2_LIKELIHOOD_CMD, sequence, timeout=120).strip()
        try:
            return float(out)
        except ValueError as exc:
            raise RuntimeError("ESM2_LIKELIHOOD_CMD did not return a numeric value.") from exc

    model, _, batch_converter, device, torch = _load_local_esm2_model()
    _, _, batch_tokens = batch_converter([("sequence", sequence)])
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        logits = model(batch_tokens, repr_layers=[], return_contacts=False)["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)

    # Exclude BOS/EOS tokens and average across sequence length.
    token_log_probs = log_probs[:, 1:-1, :]
    sequence_tokens = batch_tokens[:, 1:-1]
    if sequence_tokens.numel() == 0:
        raise RuntimeError("No amino-acid residues found after tokenization.")

    per_residue_logp = token_log_probs.gather(-1, sequence_tokens.unsqueeze(-1)).squeeze(-1)
    return float(per_residue_logp.mean().item())
