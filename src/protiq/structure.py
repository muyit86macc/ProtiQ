"""AlphaFold structures and Foldseek structural similarity search.

* AlphaFold DB models are downloaded by UniProt accession and cached on disk.
* Structure quality features come straight from the model: AlphaFold stores per residue
  confidence (pLDDT) in the B-factor column, so no extra software is needed.
* Foldseek searches a query structure against the reference proteins' AlphaFold models and
  returns the same ``Hit`` objects as the sequence search, so structural neighbors can vote
  on function exactly like sequence neighbors do.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import requests

from .homology import Hit, Hits, SearchBackend

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{uid}"
ALPHAFOLD_FILE = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v{ver}.pdb"
DEFAULT_CACHE = os.path.join("data", "structures")

STRUCTURE_FEATURES = ["af_plddt_mean", "af_plddt_high_frac", "af_plddt_low_frac", "af_radius_gyration"]


# ------------------------------------------------------------------ download
def fetch_alphafold_pdb(uniprot_id: str, cache_dir: str = DEFAULT_CACHE,
                        session: Optional[requests.Session] = None) -> Optional[str]:
    """Return a local path to the AlphaFold model for ``uniprot_id`` or None if there is none.

    Uses the AlphaFold DB API to find the current file (model versions change over time),
    and falls back to guessing the file name for older or newer versions.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"AF-{uniprot_id}.pdb")
    missing_marker = path + ".missing"
    if os.path.exists(path):
        return path
    if os.path.exists(missing_marker):
        return None
    session = session or requests.Session()
    urls: List[str] = []
    try:
        r = session.get(ALPHAFOLD_API.format(uid=uniprot_id), timeout=30)
        if r.ok:
            entries = r.json()
            if isinstance(entries, dict):
                entries = [entries]
            urls += [e["pdbUrl"] for e in entries if e.get("pdbUrl")]
    except (requests.RequestException, ValueError):
        pass
    urls += [ALPHAFOLD_FILE.format(uid=uniprot_id, ver=v) for v in (6, 5, 4)]
    for url in urls:
        try:
            r = session.get(url, timeout=60)
            if r.ok and r.text.lstrip().startswith(("HEADER", "ATOM", "MODEL", "REMARK", "TITLE")):
                with open(path, "w") as fh:
                    fh.write(r.text)
                return path
        except requests.RequestException:
            continue
    open(missing_marker, "w").close()  # remember misses so batch runs do not retry forever
    return None


# ------------------------------------------------------------------ features
def parse_ca_atoms(pdb_path: str):
    """Return (coords Nx3, bfactors N) for C-alpha atoms of the first model."""
    coords, bfac = [], []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                bfac.append(float(line[60:66]))
    return np.asarray(coords, dtype=float), np.asarray(bfac, dtype=float)


def structure_features(pdb_path: Optional[str]) -> Dict[str, float]:
    """pLDDT confidence summary and compactness. NaN when no structure is available."""
    if not pdb_path or not os.path.exists(pdb_path):
        return {k: math.nan for k in STRUCTURE_FEATURES}
    coords, plddt = parse_ca_atoms(pdb_path)
    if len(coords) == 0:
        return {k: math.nan for k in STRUCTURE_FEATURES}
    if plddt.max() <= 1.0:  # some pipelines store pLDDT as 0 to 1
        plddt = plddt * 100
    centered = coords - coords.mean(axis=0)
    rg = float(np.sqrt((centered ** 2).sum(axis=1).mean()))
    # Normalize by the expected radius of gyration of a folded chain (about 2.2 * N^0.38).
    rg_norm = rg / (2.2 * len(coords) ** 0.38)
    return {
        "af_plddt_mean": float(plddt.mean()),
        "af_plddt_high_frac": float((plddt > 90).mean()),
        "af_plddt_low_frac": float((plddt < 50).mean()),
        "af_radius_gyration": rg_norm,
    }


FEATURE_CACHE_FILE = "structure_features.csv"


def load_feature_cache(cache_dir: str = DEFAULT_CACHE) -> Dict[str, Dict[str, float]]:
    """Precomputed AlphaFold features (uniprot_id + STRUCTURE_FEATURES columns), if present.
    Lets large batches reuse features without keeping thousands of PDB files around."""
    import pandas as pd

    path = os.path.join(cache_dir, FEATURE_CACHE_FILE)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path).drop_duplicates("uniprot_id", keep="last").set_index("uniprot_id")
    return {i: {k: float(r[k]) for k in STRUCTURE_FEATURES} for i, r in df.iterrows()}


def append_feature_cache(rows: Dict[str, Dict[str, float]], cache_dir: str = DEFAULT_CACHE) -> str:
    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, FEATURE_CACHE_FILE)
    new = pd.DataFrame([{"uniprot_id": i, **f} for i, f in rows.items()])
    if os.path.exists(path):
        new = pd.concat([pd.read_csv(path), new]).drop_duplicates("uniprot_id", keep="last")
    new.to_csv(path, index=False)
    return path


def alphafold_features_batch(ids: Iterable[str], cache_dir: str = DEFAULT_CACHE, keep_pdb: bool = False,
                             verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """Compute and cache AlphaFold features for many proteins (skips ones already cached)."""
    cached = load_feature_cache(cache_dir)
    todo = [i for i in dict.fromkeys(ids) if i not in cached]
    session = requests.Session()
    new: Dict[str, Dict[str, float]] = {}
    for n, uid in enumerate(todo, 1):
        path = fetch_alphafold_pdb(uid, cache_dir, session)
        new[uid] = structure_features(path)
        if path and not keep_pdb:
            os.remove(path)
        if verbose and (n % 50 == 0 or n == len(todo)):
            print(f"  {n}/{len(todo)} structures processed")
        if n % 200 == 0:
            append_feature_cache(new, cache_dir)
    if new:
        append_feature_cache(new, cache_dir)
    cached.update(new)
    return cached


# ------------------------------------------------------------------ Foldseek
def foldseek_available() -> bool:
    return shutil.which("foldseek") is not None


FOLDSEEK_COLUMNS = "query,target,fident,alntmscore,evalue,bits"


def parse_foldseek_m8(path: str, evalue_threshold: float = 1e-3,
                      columns: str = FOLDSEEK_COLUMNS) -> Hits:
    """Parse Foldseek tabular output.

    With ProtiQ's ``--format-output`` the columns are query, target, fident, alntmscore,
    evalue, bits. The Foldseek default (BLAST m8 style, 12 columns) is also supported:
    there, E-value is column 11 and bitscore column 12. The original notebook read the
    bitscore (column 12) as the E-value, which filtered out every good hit.
    """
    hits: Hits = {}
    if not os.path.exists(path):
        return hits
    cols = columns.split(",")
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            try:
                if len(parts) >= 12 and len(cols) != len(parts):  # default m8 format
                    q, t, fident = parts[0], parts[1], float(parts[2])
                    evalue, bits, tm = float(parts[10]), float(parts[11]), math.nan
                else:
                    rec = dict(zip(cols, parts))
                    q, t = rec["query"], rec["target"]
                    fident = float(rec.get("fident", "nan"))
                    tm = float(rec.get("alntmscore", "nan"))
                    evalue, bits = float(rec["evalue"]), float(rec["bits"])
            except (ValueError, KeyError):
                continue
            if evalue >= evalue_threshold:
                continue
            q, t = _strip_af_name(q), _strip_af_name(t)
            fident = fident * 100 if fident <= 1.0 else fident
            score = tm if not math.isnan(tm) else min(bits / 500.0, 1.0)
            lst = hits.setdefault(q, [])
            if all(h.target != t for h in lst):
                lst.append(Hit(q, t, fident, evalue, bits, float(score)))
    for lst in hits.values():
        lst.sort(key=lambda h: (-h.score, h.evalue))
    return hits


def _strip_af_name(name: str) -> str:
    """'AF-P12345.pdb' or 'AF-P12345-F1-model_v4' -> 'P12345'."""
    base = os.path.basename(name)
    for suffix in (".pdb", ".cif", ".gz"):
        base = base.replace(suffix, "")
    if base.startswith("AF-"):
        base = base[3:].split("-F1")[0]
    return base


class FoldseekBackend(SearchBackend):
    """Structural neighbor search over AlphaFold models of the reference proteins."""

    name = "foldseek"

    def __init__(self, cache_dir: str = DEFAULT_CACHE, workdir: Optional[str] = None,
                 threads: int = 2, evalue: float = 1e-3, exhaustive: Optional[bool] = None):
        if not foldseek_available():
            raise RuntimeError(
                "Foldseek is not installed. Install it (conda install -c bioconda foldseek, "
                "or download from https://github.com/steineggerlab/foldseek) or skip structure search."
            )
        self.cache_dir = cache_dir
        self.workdir = workdir or tempfile.mkdtemp(prefix="protiq_foldseek_")
        self.threads = threads
        self.evalue = evalue
        # Exhaustive search skips the k-mer prefilter: slower per query, but more sensitive and
        # cheap for reference sets of a few thousand structures. None = decide by set size.
        self.exhaustive = exhaustive
        self.ref_ids: List[str] = []

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_built"] = False
        return state

    def fit(self, ids: Sequence[str], sequences: Sequence[str] = ()):
        self.ref_ids = [i for i in ids if fetch_alphafold_pdb(i, self.cache_dir)]
        self._build()
        return self

    def _build(self):
        refdir = os.path.join(self.workdir, "ref_pdbs")
        os.makedirs(refdir, exist_ok=True)
        for i in self.ref_ids:
            src = os.path.join(self.cache_dir, f"AF-{i}.pdb")
            dst = os.path.join(refdir, f"AF-{i}.pdb")
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)
        self.db = os.path.join(self.workdir, "refdb")
        subprocess.run(["foldseek", "createdb", refdir, self.db], check=True, capture_output=True)
        self._built = True

    def search(self, ids, sequences=(), top_n=10, exclude_self=False, pdb_paths: Optional[Dict[str, str]] = None):
        if not getattr(self, "_built", False):
            self._build()
        qdir = tempfile.mkdtemp(prefix="q_", dir=self.workdir)
        result: Hits = {i: [] for i in ids}
        n = 0
        for i in ids:
            src = (pdb_paths or {}).get(i) or fetch_alphafold_pdb(i, self.cache_dir)
            if src:
                shutil.copy(src, os.path.join(qdir, f"AF-{i}.pdb"))
                n += 1
        if n == 0:
            return result
        out = os.path.join(self.workdir, "fs_hits.m8")
        exhaustive = self.exhaustive if self.exhaustive is not None else len(self.ref_ids) <= 5000
        cmd = ["foldseek", "easy-search", qdir, self.db, out, os.path.join(self.workdir, "tmp"),
               "--format-output", FOLDSEEK_COLUMNS, "--max-seqs", str(top_n + 1),
               "-e", str(self.evalue), "--threads", str(self.threads)]
        if exhaustive:
            cmd += ["--exhaustive-search", "1"]
        subprocess.run(cmd, check=True, capture_output=True)
        for q, lst in parse_foldseek_m8(out, self.evalue).items():
            if exclude_self:
                lst = [h for h in lst if h.target != q]
            result[q] = lst[:top_n]
        return result
