"""Pydantic schemas enforcing structured LLM output (Week 1, Block 2).

These schemas serve two purposes:
1. They define the exact shape of the data the CrewAI researcher agent
   must return when extracting protein records from scientific literature.
2. The `description` strings on each field act as inline instructions to
   the LLM, telling it exactly what to extract and in what format.

Block 4 will wire `ProteinData` into the CrewAI agent's `output_pydantic`
contract, at which point any malformed extraction will be rejected
automatically before reaching the DataFrame.
"""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field, field_validator

# The 20 canonical amino acid one-letter codes
_VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


class ProteinData(BaseModel):
    """
    Represents a single Lanmodulin variant and its measured rare earth
    element (REE) binding metric, as extracted from scientific literature.
    @param protein_sequence: Full amino acid sequence as a single uppercase
                             string using the 20 standard one-letter codes.
    @param microbial_origin: Source organism of the protein.
    @param target_REE: Full element name of the rare earth target.
    @param binding_affinity: Measured dissociation constant in molar units.
    """

    protein_sequence: str = Field(
        ...,
        description=(
            "Full amino acid sequence of the protein, expressed as a single "
            "uppercase string using only the 20 standard one-letter codes "
            "(ACDEFGHIKLMNPQRSTVWY). Do not include spaces, gaps, numbers, "
            "or non-standard residues."
        ),
        min_length=10,
    )
    microbial_origin: str = Field(
        ...,
        description=(
            "Source organism of the protein, given as the full scientific "
            "name including strain when available, e.g. "
            "'Methylobacterium extorquens AM1' or 'Hansschlegelia quercus'."
        ),
    )
    target_REE: str = Field(
        ...,
        description=(
            "Target rare earth element as the full element name (e.g. "
            "'Neodymium', 'Dysprosium', 'Lanthanum'). Do not use the "
            "chemical symbol."
        ),
    )
    binding_affinity: float = Field(
        ...,
        description=(
            "Measured dissociation constant (Kd) in molar units, in "
            "scientific notation. Lower values indicate tighter binding. "
            "For protein-metal binding this is typically between 1e-15 "
            "and 1e-3 (femtomolar to millimolar). Example: a picomolar "
            "Kd should be expressed as 1.2e-12, not 1.2."
        ),
        gt=0,
        lt=1e-2,
    )

    @field_validator("protein_sequence")
    @classmethod
    def _sequence_is_valid_amino_acids(cls, value: str) -> str:
        """
        Validates that the sequence contains only the 20 standard amino
        acid one-letter codes after normalization.
        @param value: The raw sequence string supplied to the model.
        return : The cleaned (stripped, uppercased) sequence string.
        raises : ValueError if any non-standard characters are found.
        """
        cleaned = value.strip().upper()

        invalid = set(cleaned) - _VALID_AMINO_ACIDS
        if invalid:
            raise ValueError(
                f"Sequence contains invalid characters: {sorted(invalid)}"
            )

        return cleaned


class ProteinDataset(BaseModel):
    """
    A collection of ProteinData records. This is the top-level return
    type the CrewAI agent will produce when iterating over multiple
    papers in Block 4.
    @param records: The list of extracted protein records.
    """

    records: List[ProteinData] = Field(default_factory=list)

    def __len__(self) -> int:
        """
        return : The number of protein records currently held.
        """
        return len(self.records)
