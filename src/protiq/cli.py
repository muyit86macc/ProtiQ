"""Command line interface: `protiq <command> --help` for details."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

from . import __version__


def _parse_weights(text):
    if not text:
        return {}
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def _config(args):
    from .model import TrainConfig
    return TrainConfig(
        backend=args.backend, structure_search=args.foldseek, alphafold_features=args.alphafold,
        physchem=not args.no_physchem, top_n=args.top_n, k=args.k, metric=args.metric,
        fallback_threshold=args.fallback_threshold, group_weights=_parse_weights(args.weights),
        test_size=args.test_size, seed=args.seed, structure_cache=args.structure_cache,
    )


def _load_data(path):
    df = pd.read_csv(path)
    missing = {"uniprot_id", "sequence", "label"} - set(df.columns)
    if missing:
        sys.exit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return df


def _read_inputs(args):
    from .sequences import read_fasta, validate_sequence
    if args.sequence:
        return [args.id or "query"], [validate_sequence(args.sequence)]
    if args.fasta:
        recs = read_fasta(args.fasta)
        return [r.id for r in recs], [validate_sequence(r.sequence) for r in recs]
    if args.csv:
        df = pd.read_csv(args.csv)
        return df["uniprot_id"].tolist(), [validate_sequence(s) for s in df["sequence"]]
    sys.exit("Provide --sequence, --fasta, or --csv")


# ------------------------------------------------------------------ commands
def cmd_fetch(args):
    from .uniprot import fetch_training_data
    print(f"Downloading {args.scheme} labeled proteins (taxon {args.taxon}, up to {args.per_class} per class)...")
    df = fetch_training_data(scheme=args.scheme, per_class=args.per_class, taxon=args.taxon)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} proteins to {args.out}")


def cmd_fetch_targets(args):
    from .uniprot import fetch_uncharacterized
    df = fetch_uncharacterized(taxon=args.taxon, limit=args.limit)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} uncharacterized proteins (taxon {args.taxon}) to {args.out}")


def cmd_demo_data(args):
    from .synthetic import make_synthetic_dataset
    df = make_synthetic_dataset(seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} SYNTHETIC proteins to {args.out} (for testing only, not real data)")


def _print_metrics(m):
    print(f"\nHeld out test set: {m['n_test']} proteins (trained on {m['n_train']})")
    print(f"  ProtiQ accuracy          {m['accuracy']:.1%}   macro F1 {m['macro_f1']:.3f}")
    print(f"  Top BLAST hit baseline   {m['baseline_top_hit_accuracy']:.1%}")
    print(f"  Majority class baseline  {m['baseline_majority_accuracy']:.1%}")
    print(f"  Logistic regression fallback used for {m['fallback_rate']:.1%} of predictions\n")
    print(m["report"])


def cmd_train(args):
    from .model import save_bundle, train_model
    df = _load_data(args.data)
    cfg = _config(args)
    t0 = time.time()
    bundle = train_model(df, cfg)
    _print_metrics(bundle["metrics"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_bundle(bundle, args.out)
    m = {k: v for k, v in bundle["metrics"].items() if k != "report"}
    with open(os.path.splitext(args.out)[0] + "_metrics.json", "w") as fh:
        json.dump({"config": cfg.__dict__, "classes": bundle["classes"], **m}, fh, indent=2)
    print(f"Model saved to {args.out} ({time.time() - t0:.1f}s total)")


def cmd_predict(args):
    from .model import load_bundle, predict
    bundle = load_bundle(args.model)
    ids, seqs = _read_inputs(args)
    pdbs = {ids[0]: args.pdb} if args.pdb else None
    t0 = time.time()
    res = predict(bundle, ids, seqs, pdb_paths=pdbs)
    show = res[["id", "prediction", "confidence", "method", "top_sequence_neighbors"]]
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(show.to_string(index=False))
    weak = res["evidence"].str.startswith("no homologs").sum()
    print(f"\n{len(res)} proteins in {time.time() - t0:.1f}s ({(time.time() - t0) / len(res):.3f}s per protein)")
    if weak:
        print(f"Note: {weak} of {len(res)} proteins had no sequence or structure neighbors in the reference set.\n"
              "      Their predictions are low evidence. A larger reference set (fetch --per-class 2000+) helps.")
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"Saved predictions to {args.out}")


def cmd_explain(args):
    from .explain import explain_prediction, format_explanation
    from .model import load_bundle
    bundle = load_bundle(args.model)
    ids, seqs = _read_inputs(args)
    pdbs = {ids[0]: args.pdb} if args.pdb else None
    res = explain_prediction(bundle, ids[0], seqs[0], num_features=args.num_features, html_path=args.html,
                             pdb_paths=pdbs)
    print(format_explanation(res))


def cmd_ablate(args):
    from .model import ablation
    out = ablation(_load_data(args.data), _config(args))
    print("\nAblation (each feature group removed in turn):")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.out:
        out.to_csv(args.out, index=False)


def cmd_sweep_k(args):
    from .model import k_sweep
    out = k_sweep(_load_data(args.data), _config(args))
    print("\nAccuracy by k:")
    print(out.pivot(index="k", columns="logreg_fallback", values="accuracy")
          .rename(columns={False: "kNN only", True: "kNN + LR fallback"})
          .to_string(float_format=lambda v: f"{v:.3f}"))
    if args.out:
        out.to_csv(args.out, index=False)


def cmd_blast(args):
    from .blast_remote import run_blastp
    from .sequences import validate_sequence
    hits = run_blastp(validate_sequence(args.sequence), database=args.database)
    if not hits:
        print("No significant hits.")
    for acc, ev, ident, desc in hits[: args.max_hits]:
        print(f"{acc:<12} E={ev:.1e}  {ident:5.1f}% id  {desc[:80]}")


def cmd_structure(args):
    from .structure import fetch_alphafold_pdb, structure_features
    path = fetch_alphafold_pdb(args.uniprot_id, cache_dir=args.cache)
    if not path:
        sys.exit(f"No AlphaFold model found for {args.uniprot_id}")
    print(f"Saved {path}")
    for k, v in structure_features(path).items():
        print(f"  {k:<22} {v:.3f}")


def cmd_structure_batch(args):
    from .structure import alphafold_features_batch
    ids = pd.read_csv(args.csv)["uniprot_id"].dropna().tolist()
    feats = alphafold_features_batch(ids, cache_dir=args.cache, keep_pdb=args.keep_pdb)
    found = sum(1 for i in ids if i in feats and feats[i]["af_plddt_mean"] == feats[i]["af_plddt_mean"])
    print(f"AlphaFold features available for {found}/{len(set(ids))} proteins (cached in {args.cache})")


def cmd_foldseek_parse(args):
    from .structure import parse_foldseek_m8
    hits = parse_foldseek_m8(args.file, evalue_threshold=args.evalue)
    for q, lst in hits.items():
        print(f"{q}:")
        for i, h in enumerate(lst[: args.max_hits], 1):
            print(f"  [{i}] {h.target}  E={h.evalue:.1e}  bits={h.bitscore:.0f}  score={h.score:.3f}")
    if not hits:
        print("No hits below the E-value threshold.")


# ------------------------------------------------------------------ parser
def _add_model_args(p):
    g = p.add_argument_group("pipeline options")
    g.add_argument("--backend", default="auto", choices=["auto", "diamond", "kmer"],
                   help="sequence search: DIAMOND if installed, else k-mer (default auto)")
    g.add_argument("--foldseek", action="store_true", help="add Foldseek structural neighbors (needs foldseek + network)")
    g.add_argument("--alphafold", action="store_true", help="add AlphaFold pLDDT/compactness features (needs network)")
    g.add_argument("--no-physchem", action="store_true", help="drop physicochemical/composition features")
    g.add_argument("--top-n", type=int, default=10, help="neighbors used for voting (default 10)")
    g.add_argument("--k", type=int, default=7, help="k for kNN (default 7)")
    g.add_argument("--metric", default="cosine", help="kNN distance metric (default cosine)")
    g.add_argument("--fallback-threshold", type=float, default=0.6,
                   help="use logistic regression when kNN confidence is below this (0 disables)")
    g.add_argument("--weights", default="", help="group weights, e.g. sequence=1,structure=1.5,physchem=0.5")
    g.add_argument("--structure-cache", default="data/structures", help="where AlphaFold files/features are cached")
    g.add_argument("--test-size", type=float, default=0.2)
    g.add_argument("--seed", type=int, default=42)


def _add_input_args(p):
    p.add_argument("--model", default="models/protiq.joblib")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence", help="amino acid sequence")
    src.add_argument("--fasta", help="FASTA file with one or more proteins")
    src.add_argument("--csv", help="CSV with uniprot_id and sequence columns")
    p.add_argument("--id", help="name/UniProt ID for --sequence (lets AlphaFold features be fetched)")
    p.add_argument("--pdb", help="your own structure file for the (first) query protein")


def build_parser():
    p = argparse.ArgumentParser(prog="protiq", description="ProtiQ protein function prediction")
    p.add_argument("--version", action="version", version=f"protiq {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("fetch", help="download labeled training proteins from UniProt")
    s.add_argument("--scheme", default="ec", choices=["ec", "virulence"])
    s.add_argument("--per-class", type=int, default=300)
    s.add_argument("--taxon", type=int, default=2, help="NCBI taxonomy ID (default 2 = all bacteria)")
    s.add_argument("--out", default="data/training_ec.csv")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("fetch-targets", help="download uncharacterized proteins of an organism")
    s.add_argument("--taxon", type=int, default=208963, help="default 208963 = P. aeruginosa PA14")
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--out", default="data/targets_pa14.csv")
    s.set_defaults(func=cmd_fetch_targets)

    s = sub.add_parser("demo-data", help="write a small synthetic dataset for offline testing")
    s.add_argument("--out", default="data/synthetic_demo.csv")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_demo_data)

    s = sub.add_parser("train", help="train and evaluate a model")
    s.add_argument("--data", required=True)
    s.add_argument("--out", default="models/protiq.joblib")
    _add_model_args(s)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("predict", help="predict function for one or many proteins")
    _add_input_args(s)
    s.add_argument("--out", help="save predictions CSV")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("explain", help="LIME explanation for one protein")
    _add_input_args(s)
    s.add_argument("--num-features", type=int, default=10)
    s.add_argument("--html", help="save the interactive LIME chart as HTML")
    s.set_defaults(func=cmd_explain)

    s = sub.add_parser("ablate", help="ablation test: remove each feature group in turn")
    s.add_argument("--data", required=True)
    s.add_argument("--out")
    _add_model_args(s)
    s.set_defaults(func=cmd_ablate)

    s = sub.add_parser("sweep-k", help="accuracy for different k values")
    s.add_argument("--data", required=True)
    s.add_argument("--out")
    _add_model_args(s)
    s.set_defaults(func=cmd_sweep_k)

    s = sub.add_parser("blast", help="NCBI remote BLAST for one sequence (slow, exploratory)")
    s.add_argument("--sequence", required=True)
    s.add_argument("--database", default="swissprot")
    s.add_argument("--max-hits", type=int, default=15)
    s.set_defaults(func=cmd_blast)

    s = sub.add_parser("structure", help="download an AlphaFold model and print its features")
    s.add_argument("--uniprot-id", required=True)
    s.add_argument("--cache", default="data/structures")
    s.set_defaults(func=cmd_structure)

    s = sub.add_parser("structure-batch", help="precompute AlphaFold features for every protein in a CSV")
    s.add_argument("--csv", required=True)
    s.add_argument("--cache", default="data/structures")
    s.add_argument("--keep-pdb", action="store_true", help="keep PDB files (needed for --foldseek)")
    s.set_defaults(func=cmd_structure_batch)

    s = sub.add_parser("foldseek-parse", help="print hits from a Foldseek .m8 file")
    s.add_argument("file")
    s.add_argument("--evalue", type=float, default=1e-3)
    s.add_argument("--max-hits", type=int, default=20)
    s.set_defaults(func=cmd_foldseek_parse)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
