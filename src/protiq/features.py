"""Turn a protein into a feature vector.

Feature groups (each can be switched off for ablation, or weighted, see model.py):

* ``sequence``   top sequence hit identity and E-value, plus score weighted votes of the
                 top sequence neighbors for each function class (homology transfer)
* ``structure``  top Foldseek TM-score plus structural neighbor votes for each class
* ``alphafold``  pLDDT confidence summary and compactness of the AlphaFold model
* ``physchem``   length, charge, hydrophobicity, predicted secondary structure fractions,
                 and amino acid composition
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from .homology import Hits, SearchBackend
from .sequences import STANDARD_AA
from .structure import fetch_alphafold_pdb, load_feature_cache, structure_features

GROUPS = ("sequence", "structure", "alphafold", "physchem")


def _safe(fn, default=math.nan):
    try:
        v = fn()
        return float(v) if v is not None and np.isfinite(v) else default
    except Exception:
        return default


def physchem_features(sequence: str) -> Dict[str, float]:
    seq = "".join(c for c in sequence if c in STANDARD_AA)
    feats: Dict[str, float] = {}
    if not seq:
        return {**{f"pc_{k}": math.nan for k in ("log_length", "mol_weight_kda", "isoelectric_point", "gravy",
                                                  "aromaticity", "instability", "charge_ph7", "helix_frac",
                                                  "turn_frac", "sheet_frac")},
                **{f"aa_{a}": math.nan for a in STANDARD_AA}}
    pa = ProteinAnalysis(seq)
    helix, turn, sheet = pa.secondary_structure_fraction()
    feats.update({
        "pc_log_length": math.log10(len(seq)),
        "pc_mol_weight_kda": _safe(lambda: pa.molecular_weight() / 1000),
        "pc_isoelectric_point": _safe(pa.isoelectric_point),
        "pc_gravy": _safe(pa.gravy),
        "pc_aromaticity": _safe(pa.aromaticity),
        "pc_instability": _safe(pa.instability_index),
        "pc_charge_ph7": _safe(lambda: pa.charge_at_pH(7.0) / len(seq) * 100),
        "pc_helix_frac": helix,
        "pc_turn_frac": turn,
        "pc_sheet_frac": sheet,
    })
    counts = {a: seq.count(a) / len(seq) for a in STANDARD_AA}
    feats.update({f"aa_{a}": v for a, v in counts.items()})
    return feats


def neighbor_votes(hits, labels: Dict[str, str], classes: Sequence[str], prefix: str) -> Dict[str, float]:
    """Score weighted share of each class among the neighbors. This is the 'most common
    annotation among the top similar proteins' idea, made smooth so kNN can use it."""
    votes = {c: 0.0 for c in classes}
    total = 0.0
    for h in hits:
        lab = labels.get(h.target)
        if lab in votes:
            w = max(h.score, 0.0)
            votes[lab] += w
            total += w
    return {f"{prefix}_vote_{c}": (v / total if total > 0 else 0.0) for c, v in votes.items()}


def group_of(column: str) -> str:
    if column.startswith("seq_"):
        return "sequence"
    if column.startswith("struct_"):
        return "structure"
    if column.startswith("af_"):
        return "alphafold"
    return "physchem"


class FeatureBuilder:
    """Holds the annotated reference set and computes features for any protein."""

    def __init__(self, seq_backend: SearchBackend, struct_backend: Optional[SearchBackend] = None,
                 use_alphafold: bool = False, use_physchem: bool = True, top_n: int = 10,
                 structure_cache: str = "data/structures"):
        self.seq_backend = seq_backend
        self.struct_backend = struct_backend
        self.use_alphafold = use_alphafold
        self.use_physchem = use_physchem
        self.top_n = top_n
        self.structure_cache = structure_cache
        self.labels: Dict[str, str] = {}
        self.classes: List[str] = []
        self.ref_names: Dict[str, str] = {}

    def fit(self, ref: pd.DataFrame) -> "FeatureBuilder":
        """ref needs columns uniprot_id, sequence, label (protein_name optional)."""
        self.labels = dict(zip(ref["uniprot_id"], ref["label"]))
        self.classes = sorted(ref["label"].unique())
        if "protein_name" in ref:
            self.ref_names = dict(zip(ref["uniprot_id"], ref["protein_name"].fillna("")))
        self.seq_backend.fit(ref["uniprot_id"].tolist(), ref["sequence"].tolist())
        if self.struct_backend is not None:
            self.struct_backend.fit(ref["uniprot_id"].tolist(), ref["sequence"].tolist())
        return self

    def transform(self, ids: Sequence[str], sequences: Sequence[str], exclude_self: bool = False,
                  pdb_paths: Optional[Dict[str, str]] = None):
        """Return (features DataFrame indexed by id, {'sequence': hits, 'structure': hits})."""
        ids, sequences = list(ids), list(sequences)
        seq_hits: Hits = self.seq_backend.search(ids, sequences, top_n=self.top_n, exclude_self=exclude_self)
        struct_hits: Hits = {}
        if self.struct_backend is not None:
            struct_hits = self.struct_backend.search(ids, sequences, top_n=self.top_n,
                                                     exclude_self=exclude_self, pdb_paths=pdb_paths)
        af_cache = load_feature_cache(self.structure_cache) if self.use_alphafold else {}
        rows = []
        for pid, seq in zip(ids, sequences):
            row: Dict[str, float] = {}
            sh = seq_hits.get(pid, [])
            row["seq_top_identity"] = sh[0].identity if sh else 0.0
            row["seq_top_log_evalue"] = (
                min(-math.log10(max(sh[0].evalue, 1e-180)), 180.0) if sh and not math.isnan(sh[0].evalue) else
                (0.0 if not sh or self.seq_backend.name == "diamond" else math.nan)
            )
            row["seq_n_hits"] = float(len(sh))
            row.update(neighbor_votes(sh, self.labels, self.classes, "seq"))
            if self.struct_backend is not None:
                th = struct_hits.get(pid, [])
                row["struct_top_tmscore"] = th[0].score if th else 0.0
                row["struct_n_hits"] = float(len(th))
                row.update(neighbor_votes(th, self.labels, self.classes, "struct"))
            if self.use_alphafold:
                user_pdb = (pdb_paths or {}).get(pid)
                if user_pdb:
                    row.update(structure_features(user_pdb))
                elif pid in af_cache:
                    row.update(af_cache[pid])
                else:
                    row.update(structure_features(fetch_alphafold_pdb(pid, self.structure_cache)))
            if self.use_physchem:
                row.update(physchem_features(seq))
            rows.append(row)
        X = pd.DataFrame(rows, index=ids)
        # A backend without E-values (k-mer) gives an all NaN column: drop it.
        if X["seq_top_log_evalue"].isna().all():
            X = X.drop(columns="seq_top_log_evalue")
        return X, {"sequence": seq_hits, "structure": struct_hits}

    @property
    def groups_enabled(self) -> List[str]:
        g = ["sequence"]
        if self.struct_backend is not None:
            g.append("structure")
        if self.use_alphafold:
            g.append("alphafold")
        if self.use_physchem:
            g.append("physchem")
        return g
