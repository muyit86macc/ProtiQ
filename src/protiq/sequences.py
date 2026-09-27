"""Sequence input, validation, and FASTA handling."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
# Ambiguous or rare residues that appear in real databases. Accepted on input,
# ignored when computing composition features.
EXTENDED_AA = STANDARD_AA + "XBZUO"

# Official UniProt accession pattern (covers IDs that do not start with P or Q).
UNIPROT_ACCESSION_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)


class InvalidSequenceError(ValueError):
    """Raised when a sequence contains characters that are not amino acids."""


@dataclass
class ProteinRecord:
    id: str
    sequence: str
    description: str = ""


def clean_sequence(sequence: str) -> str:
    """Uppercase, strip whitespace, digits, and a trailing stop codon ('*')."""
    seq = re.sub(r"[\s\d]", "", sequence or "").upper()
    return seq.rstrip("*")


def validate_sequence(sequence: str, min_length: int = 10) -> str:
    """Return the cleaned sequence or raise InvalidSequenceError with a helpful message."""
    seq = clean_sequence(sequence)
    if len(seq) < min_length:
        raise InvalidSequenceError(
            f"Sequence is too short ({len(seq)} residues, minimum is {min_length})."
        )
    bad = sorted(set(seq) - set(EXTENDED_AA))
    if bad:
        raise InvalidSequenceError(
            f"Sequence contains non amino acid characters: {''.join(bad)}. "
            "Did you paste a DNA sequence or a FASTA header?"
        )
    return seq


def is_valid_protein(sequence: str) -> bool:
    try:
        validate_sequence(sequence, min_length=1)
        return True
    except InvalidSequenceError:
        return False


def extract_uniprot_id(text: str) -> str | None:
    """Pull a UniProt accession out of a FASTA header such as 'sp|P69905|HBA_HUMAN'."""
    if not text:
        return None
    m = UNIPROT_ACCESSION_RE.search(text)
    return m.group(1) if m else None


def read_fasta(path: str) -> List[ProteinRecord]:
    """Read every record from a FASTA file (no Biopython dependency needed)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"FASTA file not found: {path}")
    records: List[ProteinRecord] = []
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(_make_record(header, chunks))
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        records.append(_make_record(header, chunks))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def _make_record(header: str, chunks: List[str]) -> ProteinRecord:
    parts = header.split(None, 1)
    rid = parts[0]
    uid = extract_uniprot_id(rid)
    return ProteinRecord(
        id=uid or rid,
        sequence=clean_sequence("".join(chunks)),
        description=parts[1] if len(parts) > 1 else "",
    )


def write_fasta(records: Iterable[ProteinRecord], path: str, width: int = 60) -> str:
    with open(path, "w") as fh:
        for r in records:
            fh.write(f">{r.id}\n")
            for i in range(0, len(r.sequence), width):
                fh.write(r.sequence[i : i + width] + "\n")
    return path


def iter_chunks(items: List, size: int) -> Iterator[List]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
