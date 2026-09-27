"""Offline tests: no network needed. Run with `pytest`."""

import math
import os

import numpy as np
import pandas as pd
import pytest

from protiq import uniprot
from protiq.blast_remote import parse_blast_xml
from protiq.features import FeatureBuilder, neighbor_votes, physchem_features
from protiq.homology import DiamondBackend, Hit, KmerBackend, diamond_available, parse_blast_tab
from protiq.model import TrainConfig, ablation, k_sweep, predict, train_model
from protiq.sequences import (InvalidSequenceError, extract_uniprot_id, read_fasta, validate_sequence,
                              write_fasta, ProteinRecord)
from protiq.structure import (append_feature_cache, fetch_alphafold_pdb, load_feature_cache, parse_foldseek_m8,
                              structure_features)
from protiq.synthetic import make_synthetic_dataset


# ------------------------------------------------------------------ sequences
def test_validate_sequence_cleans_input():
    assert validate_sequence("mkt ayi\nAKQ 12 rqisf*") == "MKTAYIAKQRQISF"


def test_validate_sequence_rejects_dna_like_garbage():
    with pytest.raises(InvalidSequenceError):
        validate_sequence("MKTAYIAKQ-RQISFVKSHF!")


def test_uniprot_ids_beyond_p_and_q():
    # The original notebook only accepted IDs starting with P or Q.
    assert extract_uniprot_id("sp|A0A0C5B5G6|MOTSC_HUMAN") == "A0A0C5B5G6"
    assert extract_uniprot_id("tr|Q02M20|Q02M20_PSEAB") == "Q02M20"
    assert extract_uniprot_id("O05512") == "O05512"


def test_fasta_roundtrip(tmp_path):
    p = tmp_path / "x.fasta"
    write_fasta([ProteinRecord("P12345", "MKT" * 30), ProteinRecord("Q9I579", "MAV" * 25)], str(p))
    recs = read_fasta(str(p))
    assert [r.id for r in recs] == ["P12345", "Q9I579"]
    assert recs[0].sequence == "MKT" * 30


# ------------------------------------------------------------------ UniProt
def test_ec_labels():
    assert uniprot.ec_label("3.2.1.78") == "Hydrolase"
    assert uniprot.ec_label("2.7.1.15; 2.7.1.229") == "Transferase"
    assert uniprot.ec_label("3.2.1.17; 4.2.2.n1") is None  # mixed classes are dropped
    assert uniprot.ec_label("") == "Non-enzyme"
    assert uniprot.ec_label(float("nan")) == "Non-enzyme"


def test_virulence_labels():
    assert uniprot.virulence_label("Secreted;Toxin;Virulence") == "Virulence"
    assert uniprot.virulence_label("Reference proteome") == "Non-virulence"


class _Resp:
    def __init__(self, text, next_url=None, status=200):
        self.text, self.status_code, self.ok = text, status, status == 200
        self.links = {"next": {"url": next_url}} if next_url else {}

    def raise_for_status(self):
        pass


class _Session:
    """Fake UniProt: two pages of results, like the real API's Link header pagination."""

    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.pages[len(self.calls) - 1]


HEADER = "Entry\tSequence\tEC number\tKeywords\tProtein names\tOrganism\n"


def test_search_uniprot_follows_pagination():
    s = _Session([
        _Resp(HEADER + "P1\tMKTAYIAKQR\t3.1.1.1\tHydrolase\tEsterase\tE. coli\n", next_url="https://next"),
        _Resp(HEADER + "P2\tMAVLLLAAKQ\t3.4.21.1\tProtease\tProtease\tE. coli\n"),
    ])
    df = uniprot.search_uniprot("q", limit=10, session=s)
    assert df["uniprot_id"].tolist() == ["P1", "P2"]
    assert s.calls[1][1] is None  # next link already carries the query


def test_fetch_training_data_labels_client_side():
    rows = HEADER + "P1\tMKTAYIAKQRQISFVKSHFS\t3.1.1.1\tX\tA\tB\nP2\tMKTAYIAKQRQISFVKSHFT\t2.7.1.1\tX\tA\tB\n"
    s = _Session([_Resp(rows)] * 16)
    df = uniprot.fetch_training_data("ec", per_class=5, session=s, verbose=False)
    # Rows are only kept under the class their EC number actually belongs to.
    assert set(df["label"]) == {"Hydrolase", "Transferase"}
    assert df.set_index("uniprot_id").loc["P1", "label"] == "Hydrolase"


# ------------------------------------------------------------------ search backends
def test_kmer_backend_finds_homolog_and_excludes_self():
    seqs = {"A": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "B": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVA",
            "C": "WWWWPPPPGGGGHHHHCCCCWWWWPPPPGGGG"}
    kb = KmerBackend().fit(list(seqs), list(seqs.values()))
    hits = kb.search(["A"], [seqs["A"]], top_n=2, exclude_self=True)
    assert hits["A"][0].target == "B"


def test_parse_blast_tab(tmp_path):
    p = tmp_path / "h.tsv"
    p.write_text("q1\tt1\t80.0\t1e-50\t200\nq1\tt2\t40.0\t1e-5\t60\nq1\tt1\t70.0\t1e-10\t90\n")
    hits = parse_blast_tab(str(p))
    assert [h.target for h in hits["q1"]] == ["t1", "t2"]  # duplicate HSP collapsed
    assert hits["q1"][0].score == pytest.approx(0.8)


@pytest.mark.skipif(not diamond_available(), reason="DIAMOND not installed")
def test_diamond_backend(tmp_path):
    df = make_synthetic_dataset(n_classes=2, families_per_class=2, members_per_family=4,
                                mutation_range=(0.1, 0.2))
    b = DiamondBackend(workdir=str(tmp_path)).fit(df.uniprot_id, df.sequence)
    q = df.iloc[0]
    hits = b.search([q.uniprot_id], [q.sequence], top_n=3, exclude_self=True)[q.uniprot_id]
    assert hits and hits[0].target != q.uniprot_id
    assert df.set_index("uniprot_id").loc[hits[0].target, "label"] == q.label


# ------------------------------------------------------------------ structure
PDB = """HEADER    TEST
ATOM      1  N   MET A   1      11.104   6.134  -6.504  1.00 95.00           N
ATOM      2  CA  MET A   1      11.639   6.071  -5.147  1.00 95.00           C
ATOM      3  CA  LYS A   2      13.000   7.000  -4.000  1.00 40.00           C
ATOM      4  CA  THR A   3      14.500   8.000  -3.000  1.00 70.00           C
END
"""


def test_structure_features_from_plddt(tmp_path):
    p = tmp_path / "AF-X.pdb"
    p.write_text(PDB)
    f = structure_features(str(p))
    assert f["af_plddt_mean"] == pytest.approx((95 + 40 + 70) / 3)
    assert f["af_plddt_high_frac"] == pytest.approx(1 / 3)
    assert f["af_plddt_low_frac"] == pytest.approx(1 / 3)
    assert math.isnan(structure_features(None)["af_plddt_mean"])


def test_alphafold_fetch_uses_api_url(tmp_path):
    class S:
        def __init__(self):
            self.urls = []

        def get(self, url, timeout=None):
            self.urls.append(url)
            if "/api/prediction/" in url:
                r = _Resp("")
                r.json = lambda: [{"pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-Q02M20-F1-model_v6.pdb"}]
                return r
            return _Resp(PDB)

    s = S()
    path = fetch_alphafold_pdb("Q02M20", str(tmp_path), session=s)
    assert path and os.path.exists(path)
    assert s.urls[1].endswith("model_v6.pdb")  # current version, not the hardcoded v4


def test_feature_cache_roundtrip(tmp_path):
    append_feature_cache({"P1": {"af_plddt_mean": 80.0, "af_plddt_high_frac": 0.5,
                                 "af_plddt_low_frac": 0.1, "af_radius_gyration": 1.1}}, str(tmp_path))
    assert load_feature_cache(str(tmp_path))["P1"]["af_plddt_mean"] == 80.0


def test_foldseek_parser_reads_evalue_from_correct_column(tmp_path):
    # Default 12 column m8: E-value is column 11, bitscore column 12.
    p = tmp_path / "fs.m8"
    p.write_text(
        "AF-Q1-F1-model_v4.pdb\tAF-P9-F1-model_v4.pdb\t0.45\t200\t0\t0\t1\t200\t1\t200\t1.2e-20\t350\n"
        "AF-Q1-F1-model_v4.pdb\tAF-P8-F1-model_v4.pdb\t0.20\t150\t0\t0\t1\t150\t1\t150\t0.5\t40\n"
    )
    hits = parse_foldseek_m8(str(p), evalue_threshold=1e-3)
    assert list(hits) == ["Q1"]
    assert [h.target for h in hits["Q1"]] == ["P9"]  # the bad hit (E=0.5) is filtered out
    assert hits["Q1"][0].evalue == pytest.approx(1.2e-20)


def test_foldseek_parser_protiq_format(tmp_path):
    p = tmp_path / "fs.m8"
    p.write_text("AF-Q1.pdb\tAF-P9.pdb\t0.45\t0.81\t1e-15\t300\n")
    h = parse_foldseek_m8(str(p))["Q1"][0]
    assert h.target == "P9" and h.score == pytest.approx(0.81) and h.identity == pytest.approx(45)


# ------------------------------------------------------------------ features
def test_neighbor_votes_are_score_weighted():
    hits = [Hit("q", "a", 90, 1e-50, 300, 0.9), Hit("q", "b", 30, 1e-5, 50, 0.3)]
    v = neighbor_votes(hits, {"a": "X", "b": "Y"}, ["X", "Y"], "seq")
    assert v["seq_vote_X"] == pytest.approx(0.75)
    assert v["seq_vote_Y"] == pytest.approx(0.25)


def test_physchem_features_complete():
    f = physchem_features("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
    assert len(f) == 30 and all(np.isfinite(list(f.values())))


# ------------------------------------------------------------------ model
@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_dataset(seed=1)


@pytest.fixture(scope="module")
def bundle(synthetic):
    return train_model(synthetic, TrainConfig(backend="kmer"), verbose=False)


def test_training_beats_baselines(bundle):
    m = bundle["metrics"]
    assert m["accuracy"] > m["baseline_majority_accuracy"] + 0.3
    assert 0 <= m["fallback_rate"] <= 1


def test_predict_new_and_known_proteins(bundle, synthetic):
    known = synthetic.iloc[3]
    res = predict(bundle, ["known", "novel"], [known.sequence, "M" + "ACDEFGHIKLMNPQRSTVWY" * 8])
    assert res.loc[0, "prediction"] == known.label and res.loc[0, "method"] == "reference-match"
    assert res.loc[1, "prediction"] in bundle["classes"]
    assert 0 < res.loc[1, "confidence"] <= 1


def test_ablation_and_k_sweep(synthetic):
    cfg = TrainConfig(backend="kmer")
    ab = ablation(synthetic, cfg, verbose=False)
    assert ab.iloc[0]["removed_group"].startswith("(none") and len(ab) == 3
    ks = k_sweep(synthetic, cfg, ks=(1, 7), verbose=False)
    assert len(ks) == 4


def test_lime_explanation(bundle, synthetic, tmp_path):
    from protiq.explain import explain_prediction, format_explanation
    seq = synthetic.iloc[0].sequence[5:]  # a slightly different, unseen variant
    res = explain_prediction(bundle, "q", seq, num_features=5, num_samples=300,
                             html_path=str(tmp_path / "lime.html"))
    assert len(res["feature_weights"]) == 5 and os.path.exists(res["html"])
    assert "Prediction:" in format_explanation(res)


def test_cli_end_to_end(tmp_path, monkeypatch):
    from protiq.cli import main
    monkeypatch.chdir(tmp_path)
    main(["demo-data", "--out", "d.csv"])
    main(["train", "--data", "d.csv", "--out", "m.joblib", "--backend", "kmer"])
    main(["predict", "--model", "m.joblib", "--csv", "d.csv", "--out", "p.csv"])
    assert len(pd.read_csv("p.csv")) == len(pd.read_csv("d.csv"))
    assert os.path.exists("m_metrics.json")


# ------------------------------------------------------------------ remote BLAST parsing
def test_parse_blast_xml():
    xml = """<?xml version="1.0"?><BlastOutput><BlastOutput_iterations><Iteration><Iteration_hits>
    <Hit><Hit_def>Hemoglobin subunit alpha</Hit_def><Hit_accession>P69905</Hit_accession>
      <Hit_hsps><Hsp><Hsp_evalue>1e-80</Hsp_evalue><Hsp_identity>140</Hsp_identity><Hsp_align-len>142</Hsp_align-len></Hsp></Hit_hsps></Hit>
    <Hit><Hit_def>weak</Hit_def><Hit_accession>P99999</Hit_accession>
      <Hit_hsps><Hsp><Hsp_evalue>0.5</Hsp_evalue><Hsp_identity>10</Hsp_identity><Hsp_align-len>100</Hsp_align-len></Hsp></Hit_hsps></Hit>
    </Iteration_hits></Iteration></BlastOutput_iterations></BlastOutput>"""
    hits = parse_blast_xml(xml)
    assert len(hits) == 1 and hits[0][0] == "P69905" and hits[0][2] == pytest.approx(98.59, 0.01)


# ------------------------------------------------------------------ Foldseek end to end
def _helix_pdb(path, n, twist=100.0, rise=1.5):
    """Ideal backbone (N, CA, C) so Foldseek has real geometry to work with."""
    lines, k = [], 1
    for i in range(n):
        for name, r, dphi, dz in (("N", 1.55, -28, -0.84), ("CA", 2.3, 0, 0), ("C", 1.61, 28, 0.7)):
            th = math.radians(twist * i + dphi)
            x, y, z = r * math.cos(th), r * math.sin(th), rise * i + dz
            lines.append(f"ATOM  {k:5d}  {name:<3} ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           {name[0]}")
            k += 1
    open(path, "w").write("\n".join(lines) + "\nEND\n")


@pytest.mark.skipif(not __import__("protiq.structure", fromlist=["x"]).foldseek_available(),
                    reason="Foldseek not installed")
def test_foldseek_backend_runs(tmp_path):
    from protiq.structure import FoldseekBackend
    cache = tmp_path / "structures"
    cache.mkdir()
    for i, n in enumerate((60, 62, 64, 66)):
        _helix_pdb(str(cache / f"AF-R{i}.pdb"), n)
    _helix_pdb(str(tmp_path / "q.pdb"), 63)
    fb = FoldseekBackend(cache_dir=str(cache), workdir=str(tmp_path / "fs"), evalue=10)
    fb.fit([f"R{i}" for i in range(4)])
    hits = fb.search(["Q"], top_n=3, pdb_paths={"Q": str(tmp_path / "q.pdb")})
    assert hits["Q"], "Foldseek returned no structural neighbors"
    assert all(h.target.startswith("R") for h in hits["Q"])
