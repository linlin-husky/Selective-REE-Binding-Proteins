"""Tests for the ProteinData and ProteinDataset Pydantic schemas."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from agentic_ai.schemas import ProteinData, ProteinDataset

# A realistic Lanmodulin-style fragment using only valid amino acid letters
_VALID_SEQUENCE = "MKKLLFAIPLVVPFYSHSAAQNNDGDGKVGV"


def _build_valid_payload() -> dict:
    """
    Builds a known-good payload dict that satisfies every schema rule.
    Used as a baseline so each test can mutate one field at a time.
    return : A dict suitable for `ProteinData(**payload)`.
    """
    return {
        "protein_sequence": _VALID_SEQUENCE,
        "microbial_origin": "Methylobacterium extorquens AM1",
        "target_REE": "Neodymium",
        "binding_affinity": 1.2e-12,
    }


def test_protein_data_accepts_valid_record():
    """
    Verifies that a fully-conforming payload constructs a ProteinData
    instance without raising.
    """
    record = ProteinData(**_build_valid_payload())

    assert record.protein_sequence == _VALID_SEQUENCE
    assert record.target_REE == "Neodymium"
    assert record.binding_affinity > 0


def test_protein_data_normalizes_sequence_case_and_whitespace():
    """
    Verifies that the sequence validator strips surrounding whitespace
    and uppercases lowercase inputs.
    """
    payload = _build_valid_payload()
    payload["protein_sequence"] = f"  {_VALID_SEQUENCE.lower()}  "

    record = ProteinData(**payload)

    assert record.protein_sequence == _VALID_SEQUENCE


def test_protein_data_rejects_invalid_amino_acid_characters():
    """
    Verifies that any non-standard character in the sequence triggers a
    ValidationError with an informative message.
    """
    payload = _build_valid_payload()
    payload["protein_sequence"] = "MKKLLFA1PLVVPFYSHSAAQ"

    with pytest.raises(ValidationError, match="invalid characters"):
        ProteinData(**payload)


def test_protein_data_rejects_sequence_below_min_length():
    """
    Verifies that sequences shorter than the 10-character minimum are
    rejected, since real Lanmodulin variants are >100 residues.
    """
    payload = _build_valid_payload()
    payload["protein_sequence"] = "MKK"

    with pytest.raises(ValidationError):
        ProteinData(**payload)


def test_protein_data_rejects_zero_binding_affinity():
    """
    Verifies that a binding affinity of exactly zero is rejected, since
    Kd must be strictly positive.
    """
    payload = _build_valid_payload()
    payload["binding_affinity"] = 0.0

    with pytest.raises(ValidationError):
        ProteinData(**payload)


def test_protein_data_rejects_negative_binding_affinity():
    """
    Verifies that a negative binding affinity is rejected.
    """
    payload = _build_valid_payload()
    payload["binding_affinity"] = -1.0e-10

    with pytest.raises(ValidationError):
        ProteinData(**payload)


def test_protein_data_rejects_missing_required_field():
    """
    Verifies that omitting any of the four required fields raises a
    ValidationError. Tests the `target_REE` field as a representative.
    """
    payload = _build_valid_payload()
    del payload["target_REE"]

    with pytest.raises(ValidationError):
        ProteinData(**payload)


def test_protein_dataset_defaults_to_empty():
    """
    Verifies that ProteinDataset can be instantiated with no records and
    reports length zero.
    """
    dataset = ProteinDataset()

    assert len(dataset) == 0


def test_protein_dataset_holds_multiple_records():
    """
    Verifies that ProteinDataset correctly stores and counts multiple
    ProteinData records.
    """
    record = ProteinData(**_build_valid_payload())
    dataset = ProteinDataset(records=[record, record, record])

    assert len(dataset) == 3
    assert dataset.records[0].target_REE == "Neodymium"

def test_protein_data_rejects_biologically_implausible_affinity():
    """
    Verifies that the schema rejects Kd values above the biological
    plausibility ceiling (1e-2 M). This catches the most common LLM
    extraction failure: dropping the scientific-notation exponent
    (e.g. reading '2.4e-12' as '2.4').
    """
    payload = _build_valid_payload()
    payload["binding_affinity"] = 2.4  # what happens when LLM drops "e-12"

    with pytest.raises(ValidationError):
        ProteinData(**payload)
