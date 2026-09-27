"""Optional: NCBI remote BLAST (the web service used in the original notebook).

Useful for exploring a single protein against all of NCBI. Too slow for training or
batch runs (minutes per query), which is why the pipeline uses DIAMOND or k-mer search.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import requests

from .sequences import extract_uniprot_id

BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"


def submit_blastp(sequence: str, database: str = "swissprot", session: Optional[requests.Session] = None) -> str:
    """Submit a job and return its request ID (RID). ``swissprot`` hits carry UniProt IDs."""
    session = session or requests.Session()
    r = session.post(BLAST_URL, data={"CMD": "Put", "PROGRAM": "blastp", "DATABASE": database,
                                      "QUERY": sequence, "FORMAT_TYPE": "XML"}, timeout=60)
    r.raise_for_status()
    m = re.search(r"RID = (\S+)", r.text)
    if not m:
        raise RuntimeError("NCBI did not return a request ID. The service may be busy; try again later.")
    return m.group(1)


def wait_for_blast(rid: str, poll_seconds: int = 15, max_wait: int = 900,
                   session: Optional[requests.Session] = None, verbose: bool = True) -> str:
    session = session or requests.Session()
    waited = 0
    while waited <= max_wait:
        s = session.get(BLAST_URL, params={"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"}, timeout=60).text
        if "Status=READY" in s:
            if "ThereAreHits=no" in s:
                return ""
            r = session.get(BLAST_URL, params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"}, timeout=120)
            r.raise_for_status()
            return r.text
        if "Status=FAILED" in s or "Status=UNKNOWN" in s:
            raise RuntimeError(f"BLAST job {rid} failed or expired.")
        if verbose:
            print(f"  waiting for BLAST ({waited}s)...")
        time.sleep(poll_seconds)
        waited += poll_seconds
    raise TimeoutError(f"BLAST job {rid} not ready after {max_wait}s.")


def parse_blast_xml(xml_text: str, max_evalue: float = 1e-3) -> List[Tuple[str, float, float, str]]:
    """Return [(uniprot_or_accession, evalue, percent_identity, description)] sorted by E-value."""
    if not xml_text.strip():
        return []
    root = ET.fromstring(xml_text)
    out = []
    for hit in root.iter("Hit"):
        acc = (hit.findtext("Hit_accession") or "").split(".")[0]
        desc = hit.findtext("Hit_def") or ""
        hsp = hit.find(".//Hsp")
        if hsp is None:
            continue
        ev = float(hsp.findtext("Hsp_evalue"))
        ident = int(hsp.findtext("Hsp_identity") or 0)
        length = int(hsp.findtext("Hsp_align-len") or 1)
        if ev < max_evalue:
            out.append((extract_uniprot_id(acc) or acc, ev, 100.0 * ident / length, desc))
    return sorted(out, key=lambda t: t[1])


def run_blastp(sequence: str, database: str = "swissprot", verbose: bool = True):
    rid = submit_blastp(sequence, database)
    if verbose:
        print(f"Submitted BLAST job {rid}")
    return parse_blast_xml(wait_for_blast(rid, verbose=verbose))
