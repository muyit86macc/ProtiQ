"""Download labeled training data and target proteins from the UniProt REST API.

Two label schemes match the goals of the project:

* ``ec``         enzyme class (EC 1 to 7) or Non-enzyme
* ``virulence``  Virulence vs Non-virulence, from UniProt keyword KW-0843

Labels are always assigned client side from the returned annotation columns, so a
query that returns something unexpected can never produce a wrong label.
"""

from __future__ import annotations

import time
from io import StringIO
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
BACTERIA_TAXON = 2
PA14_TAXON = 208963  # Pseudomonas aeruginosa UCBPP-PA14

# Pathogen genera/species used for organism matched negatives in the virulence scheme, so the
# model learns virulence rather than "which organism is this from".
PATHOGEN_TAXA = [287, 1280, 562, 590, 1773, 1301, 662, 1637, 629, 445, 517, 209, 1279, 1386, 570]

EC_CLASS_NAMES = {
    "1": "Oxidoreductase",
    "2": "Transferase",
    "3": "Hydrolase",
    "4": "Lyase",
    "5": "Isomerase",
    "6": "Ligase",
    "7": "Translocase",
}
NON_ENZYME = "Non-enzyme"
VIRULENCE = "Virulence"
NON_VIRULENCE = "Non-virulence"

FIELDS = ["accession", "sequence", "ec", "keyword", "protein_name", "organism_name"]
COLUMNS = ["uniprot_id", "sequence", "ec", "keywords", "protein_name", "organism"]


# ---------------------------------------------------------------- labeling rules
def ec_label(ec_field) -> Optional[str]:
    """'2.7.11.1; 2.7.10.2' -> 'Transferase'. Mixed top level classes -> None (dropped)."""
    if ec_field is None or (isinstance(ec_field, float) and pd.isna(ec_field)) or not str(ec_field).strip():
        return NON_ENZYME
    tops = {e.strip().split(".")[0] for e in str(ec_field).split(";") if e.strip()}
    tops = {t for t in tops if t in EC_CLASS_NAMES}
    if len(tops) != 1:
        return None
    return EC_CLASS_NAMES[tops.pop()]


def virulence_label(keywords) -> str:
    kw = "" if keywords is None or (isinstance(keywords, float) and pd.isna(keywords)) else str(keywords)
    return VIRULENCE if "virulence" in kw.lower() else NON_VIRULENCE


LABELERS: Dict[str, Callable] = {
    "ec": lambda row: ec_label(row["ec"]),
    "virulence": lambda row: virulence_label(row["keywords"]),
}


# ---------------------------------------------------------------- HTTP helpers
def _get(session: requests.Session, url: str, params=None, retries: int = 4) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 400:
                r.raise_for_status()  # bad query, retrying will not help
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                raise
            last = e
        except requests.RequestException as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"UniProt request failed after {retries} attempts: {last}")


def search_uniprot(
    query: str,
    limit: int,
    fields: List[str] = FIELDS,
    columns: List[str] = COLUMNS,
    session: Optional[requests.Session] = None,
    page_size: int = 500,
) -> pd.DataFrame:
    """Run a UniProtKB query and follow pagination until ``limit`` rows are collected."""
    session = session or requests.Session()
    params = {"query": query, "fields": ",".join(fields), "format": "tsv", "size": min(page_size, limit)}
    frames, url, n = [], UNIPROT_SEARCH, 0
    while url and n < limit:
        r = _get(session, url, params=params)
        text = r.text.strip()
        if text:
            df = pd.read_csv(StringIO(text), sep="\t", dtype=str)
            if len(df):
                df.columns = columns[: len(df.columns)]
                frames.append(df)
                n += len(df)
        url = r.links.get("next", {}).get("url")
        params = None  # the next link already carries the query
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).head(limit)


# ---------------------------------------------------------------- training data
def _base(taxon: int, min_len: int, max_len: int) -> str:
    return f"(reviewed:true) AND (taxonomy_id:{taxon}) AND (fragment:false) AND (length:[{min_len} TO {max_len}])"


def class_queries(scheme: str, taxon: int, min_len: int, max_len: int) -> Dict[str, str]:
    base = _base(taxon, min_len, max_len)
    if scheme == "ec":
        q = {name: f"{base} AND (ec:{digit}.*)" for digit, name in EC_CLASS_NAMES.items()}
        q[NON_ENZYME] = f"{base} NOT (ec:*)"
        return q
    if scheme == "virulence":
        return {
            VIRULENCE: f"{base} AND (keyword:KW-0843)",
            NON_VIRULENCE: f"{base} NOT (keyword:KW-0843)",
        }
    raise ValueError(f"Unknown label scheme '{scheme}'. Use one of: {', '.join(LABELERS)}")


def fetch_training_data(
    scheme: str = "ec",
    per_class: int = 300,
    taxon: int = BACTERIA_TAXON,
    min_len: int = 50,
    max_len: int = 1500,
    session: Optional[requests.Session] = None,
    verbose: bool = True,
    matched_negatives: bool = True,
) -> pd.DataFrame:
    """Build a class balanced labeled dataset: columns uniprot_id, sequence, label, protein_name, ...

    For the virulence scheme, negatives are drawn evenly from well studied pathogens
    (``PATHOGEN_TAXA``) instead of the first N UniProt entries, which are dominated by
    E. coli K-12 and B. subtilis and would let the model shortcut on organism.
    """
    session = session or requests.Session()
    labeler = LABELERS[scheme]
    frames = []
    queries = class_queries(scheme, taxon, min_len, max_len)
    if scheme == "virulence" and matched_negatives:
        base = f"(reviewed:true) AND (fragment:false) AND (length:[{min_len} TO {max_len}])"
        share = max(1, per_class // len(PATHOGEN_TAXA))
        parts = [search_uniprot(f"{base} AND (taxonomy_id:{t}) NOT (keyword:KW-0843)",
                                limit=share, session=session) for t in PATHOGEN_TAXA]
        neg = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        neg["label"] = neg.apply(labeler, axis=1)
        neg = neg[neg["label"] == NON_VIRULENCE]
        if verbose:
            print(f"  {NON_VIRULENCE:<15} {len(neg):>5} proteins (from {len(PATHOGEN_TAXA)} pathogen taxa)")
        frames.append(neg)
        queries = {VIRULENCE: queries[VIRULENCE]}
    for cls, query in queries.items():
        # Ask for extra rows because some are dropped by the client side label check.
        raw = search_uniprot(query, limit=int(per_class * 1.3) + 10, session=session)
        if raw.empty:
            if verbose:
                print(f"  {cls:<15} 0 entries returned")
            continue
        raw["label"] = raw.apply(labeler, axis=1)
        kept = raw[raw["label"] == cls].head(per_class)
        if verbose:
            print(f"  {cls:<15} {len(kept):>5} proteins")
        frames.append(kept)
    if not frames:
        raise RuntimeError("UniProt returned no data. Check your network connection and query.")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("uniprot_id")
    df = df.drop_duplicates("sequence").reset_index(drop=True)
    return df[["uniprot_id", "sequence", "label", "protein_name", "organism", "ec", "keywords"]]


def fetch_uncharacterized(
    taxon: int = PA14_TAXON, limit: int = 200, session: Optional[requests.Session] = None
) -> pd.DataFrame:
    """Proteins of an organism that UniProt lists as uncharacterized: the real use case for ProtiQ."""
    query = f'(taxonomy_id:{taxon}) AND (protein_name:"uncharacterized protein")'
    fields = ["accession", "sequence", "protein_name", "gene_names", "organism_name"]
    cols = ["uniprot_id", "sequence", "protein_name", "gene_names", "organism"]
    return search_uniprot(query, limit=limit, fields=fields, columns=cols, session=session)
