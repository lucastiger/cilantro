# scoring/prediction_tools.py
from __future__ import annotations

import contextlib
import csv
import logging
import os
import shlex
import subprocess
import tempfile
from functools import lru_cache
from typing import Dict, Iterable, List

# Optional local command integration for toxicity prediction.
TOXICITY_PREDICTOR_CMD = os.getenv("TOXICITY_PREDICTOR_CMD", "toxinpred3")
# Optional local command integration for ESM-2 likelihood.
# Expected output format: numeric mean log-likelihood per residue.
ESM2_LIKELIHOOD_CMD = os.getenv("ESM2_LIKELIHOOD_CMD", "")

# Approximate global class-I HLA allele frequencies (used as defaults when
# MHC_ALLELE_FREQUENCIES is not provided). Values are treated as relative
# weights and are renormalized during scoring.
MHCFLURRY_MIN_PEPTIDE_LENGTH = 5
MHCFLURRY_MAX_PEPTIDE_LENGTH = 15

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

    import json

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

    supported_peptides = [
        pep
        for pep in peptides_list
        if MHCFLURRY_MIN_PEPTIDE_LENGTH <= len(pep) <= MHCFLURRY_MAX_PEPTIDE_LENGTH
    ]
    unsupported_count = len(peptides_list) - len(supported_peptides)
    if unsupported_count:
        logging.warning(
            "%d peptides have lengths outside of supported range [%d, %d] and will be skipped.",
            unsupported_count,
            MHCFLURRY_MIN_PEPTIDE_LENGTH,
            MHCFLURRY_MAX_PEPTIDE_LENGTH,
        )
    if not supported_peptides:
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
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            try:
                affinities = predictor.predict(peptides=supported_peptides, allele=allele, verbose=0)
            except TypeError as exc:
                if "verbose" not in str(exc):
                    raise
                affinities = predictor.predict(peptides=supported_peptides, allele=allele)
        result[allele] = {
            peptide: float(kd)
            for peptide, kd in zip(supported_peptides, affinities)
        }
    return result


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
