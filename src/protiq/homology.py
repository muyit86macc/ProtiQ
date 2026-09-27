"""Sequence similarity search against an annotated reference set.

Two interchangeable backends return the same ``Hit`` objects:

* ``DiamondBackend``  DIAMOND blastp (a BLAST compatible aligner, about 100x faster than
  BLAST). Used when the ``diamond`` binary is installed. This is the backend that matches
  the pBLAST step of the original project.
* ``KmerBackend``     pure Python 3-mer cosine similarity. No installation needed, so the
  pipeline runs anywhere (Colab, laptop, CI). Less sensitive for distant homologs.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from .sequences import ProteinRecord, write_fasta


@dataclass
class Hit:
    query: str
    target: str
    identity: float  # 0 to 100
    evalue: float  # nan when the backend has no E-value
    bitscore: float
    score: float  # similarity in [0, 1], used for neighbor vote weights


Hits = Dict[str, List[Hit]]


class SearchBackend:
    name = "base"

    def fit(self, ids: Sequence[str], sequences: Sequence[str]) -> "SearchBackend":
        raise NotImplementedError

    def search(self, ids: Sequence[str], sequences: Sequence[str], top_n: int = 10,
               exclude_self: bool = False) -> Hits:
        raise NotImplementedError


# ------------------------------------------------------------------ k-mer backend
class KmerBackend(SearchBackend):
    name = "kmer"

    def __init__(self, k: int = 3, chunk: int = 512):
        self.k = k
        self.chunk = chunk
        self.vectorizer: Optional[CountVectorizer] = None
        self.ref_ids: List[str] = []
        self.ref_matrix = None

    def fit(self, ids, sequences):
        self.vectorizer = CountVectorizer(analyzer="char", ngram_range=(self.k, self.k), lowercase=False)
        X = self.vectorizer.fit_transform(list(sequences))
        self.ref_matrix = normalize(X.astype(np.float32)).T.tocsr()
        self.ref_ids = list(ids)
        return self

    def search(self, ids, sequences, top_n=10, exclude_self=False):
        if self.vectorizer is None:
            raise RuntimeError("Call fit() before search().")
        ids, sequences = list(ids), list(sequences)
        out: Hits = {}
        ref_index = {r: i for i, r in enumerate(self.ref_ids)}
        for start in range(0, len(ids), self.chunk):
            q_ids = ids[start : start + self.chunk]
            Q = normalize(self.vectorizer.transform(sequences[start : start + self.chunk]).astype(np.float32))
            S = (Q @ self.ref_matrix).toarray()
            for row, qid in enumerate(q_ids):
                sims = S[row]
                if exclude_self and qid in ref_index:
                    sims[ref_index[qid]] = -1.0
                n = min(top_n, len(sims))
                idx = np.argpartition(-sims, n - 1)[:n]
                idx = idx[np.argsort(-sims[idx])]
                out[qid] = [
                    Hit(qid, self.ref_ids[j], float(sims[j] * 100), math.nan, float(sims[j]), float(sims[j]))
                    for j in idx
                    if sims[j] > 0
                ]
        return out


# ------------------------------------------------------------------ DIAMOND backend
class DiamondBackend(SearchBackend):
    name = "diamond"

    def __init__(self, workdir: Optional[str] = None, sensitivity: str = "--sensitive",
                 threads: int = 2, max_evalue: float = 1e-3):
        if not diamond_available():
            raise RuntimeError(
                "DIAMOND is not installed. Install it (conda install -c bioconda diamond, "
                "or apt install diamond-aligner) or use --backend kmer."
            )
        self.workdir = workdir or tempfile.mkdtemp(prefix="protiq_diamond_")
        self.sensitivity = sensitivity
        self.threads = threads
        self.max_evalue = max_evalue
        self.db = os.path.join(self.workdir, "reference")
        self.ref_ids: List[str] = []
        self.ref_sequences: List[str] = []

    def __getstate__(self):  # keep saved models portable: rebuild the DB on load
        state = self.__dict__.copy()
        state["_built"] = False
        return state

    def fit(self, ids, sequences):
        self.ref_ids, self.ref_sequences = list(ids), list(sequences)
        self._build()
        return self

    def _build(self):
        os.makedirs(self.workdir, exist_ok=True)
        fasta = os.path.join(self.workdir, "reference.fasta")
        write_fasta([ProteinRecord(i, s) for i, s in zip(self.ref_ids, self.ref_sequences)], fasta)
        _run(["diamond", "makedb", "--in", fasta, "-d", self.db, "--quiet"])
        self._built = True

    def search(self, ids, sequences, top_n=10, exclude_self=False):
        if not getattr(self, "_built", False) or not os.path.exists(self.db + ".dmnd"):
            self._build()
        q = os.path.join(self.workdir, "query.fasta")
        out = os.path.join(self.workdir, "hits.tsv")
        write_fasta([ProteinRecord(i, s) for i, s in zip(ids, sequences)], q)
        _run([
            "diamond", "blastp", "-q", q, "-d", self.db, "-o", out, self.sensitivity,
            "--outfmt", "6", "qseqid", "sseqid", "pident", "evalue", "bitscore",
            "-k", str(top_n + 1), "--evalue", str(self.max_evalue),
            "--threads", str(self.threads), "--quiet",
        ])
        hits = parse_blast_tab(out)
        result: Hits = {i: [] for i in ids}
        for qid, lst in hits.items():
            if exclude_self:
                lst = [h for h in lst if h.target != qid]
            result[qid] = lst[:top_n]
        return result


def diamond_available() -> bool:
    return shutil.which("diamond") is not None


def parse_blast_tab(path: str) -> Hits:
    """Parse tabular output with columns qseqid sseqid pident evalue bitscore."""
    hits: Hits = {}
    if not os.path.exists(path):
        return hits
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            q, t, pid, ev, bits = line.rstrip("\n").split("\t")[:5]
            h = Hit(q, t, float(pid), float(ev), float(bits), float(pid) / 100.0)
            lst = hits.setdefault(q, [])
            if all(x.target != t for x in lst):  # keep best HSP per target
                lst.append(h)
    for lst in hits.values():
        lst.sort(key=lambda h: -h.bitscore)
    return hits


def _run(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout


def make_backend(name: str = "auto", **kwargs) -> SearchBackend:
    if name == "auto":
        name = "diamond" if diamond_available() else "kmer"
    if name == "diamond":
        return DiamondBackend(**kwargs)
    if name == "kmer":
        return KmerBackend(**kwargs)
    raise ValueError(f"Unknown backend '{name}'. Use auto, diamond, or kmer.")
