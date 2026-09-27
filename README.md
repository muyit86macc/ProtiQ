# ProtiQ

**Automated protein function prediction from sequence and structure similarity.**

ProtiQ takes a protein sequence, finds related proteins by sequence (BLAST-style search) and by 3D structure (Foldseek on AlphaFold models), and predicts the protein's function with a k-nearest neighbors classifier. Every prediction comes with a LIME explanation that shows which features and which neighboring proteins drove the call.

The goal is to speed up the characterization of uncharacterized bacterial proteins, especially virulence-associated ones, which are candidate targets for anti-virulence therapies that disarm pathogens without driving antibiotic resistance.

Built by **Muyi Tang** and **Xenia Huang**. This repository is the open-source version of our Youth Science Canada project, [*ProtiQ: Revolutionizing Protein Function Discovery using Multi-pronged Computational Analysis*](https://partner.projectboard.world/ysc/project/protiq-revolutionizing-protein-function-discovery-using-multi-pronged-computational-analysis).

---

## How it works

```
protein sequence
      |
      +--> sequence search (DIAMOND blastp, or built-in k-mer search) ---+
      |                                                                   |
      +--> AlphaFold model --> Foldseek structural search ----------------+--> features --> kNN (k=7, cosine)
      |                   \--> pLDDT confidence, compactness -------------+        |          |
      |                                                                   |        |   low confidence?
      +--> physicochemical profile (charge, hydrophobicity, composition) -+        |          v
                                                                                   |   logistic regression
                                                                                   v
                                                             prediction + confidence + LIME explanation
```

1. **Similarity search.** The query is searched against a reference set of proteins with known function. DIAMOND is a BLAST-compatible aligner that is about 100 times faster than BLAST. When DIAMOND isn't installed, ProtiQ falls back to a built-in k-mer search so it runs anywhere.
2. **Homology transfer as features.** For each function class, ProtiQ computes a score-weighted vote from the closest sequence neighbors, and separately from the closest structural neighbors. This is the "most common annotation among the top similar proteins" idea, made smooth enough for kNN to learn from. Top-hit identity, E-value, and TM-score are included too.
3. **Structure.** AlphaFold DB models are downloaded automatically. Foldseek compares them to the reference proteins' models, and the per-residue confidence (pLDDT) is summarized as features.
4. **Classification.** kNN with k = 7, cosine distance, and distance weighting, on normalized features. Sequence and structure feature groups can be weighted against each other. When the neighbors disagree (kNN confidence below 0.6), a logistic regression fallback makes the call.
5. **Explanation.** LIME fits a local linear model around each prediction and reports which features pushed toward or away from it. The nearest annotated proteins are listed next to it.
6. **Honest output.** Every prediction states its evidence: how many sequence and structure neighbors backed it. A protein with no homologs gets flagged as low evidence rather than a confident-looking guess.

## Install

```bash
git clone https://github.com/<your-username>/ProtiQ.git
cd ProtiQ
pip install -e .
```

Optional, but recommended for real use:

```bash
conda install -c bioconda diamond foldseek      # or: apt install diamond-aligner
```

Everything also runs on Google Colab: open `notebooks/ProtiQ_Colab.ipynb`.

## Quick start

```bash
# 1. Download labeled bacterial proteins from UniProt (Swiss-Prot, reviewed)
protiq fetch --scheme virulence --per-class 3000 --out data/training_virulence.csv
protiq fetch --scheme ec --per-class 500 --out data/training_ec.csv

# 2. Train and evaluate (held-out test set, compared against two baselines)
protiq train --data data/training_virulence.csv --backend diamond --out models/virulence.joblib

# 3. Get the uncharacterized proteins of P. aeruginosa PA14 and predict them in one batch
protiq fetch-targets --taxon 208963 --out data/targets_pa14.csv
protiq predict --model models/virulence.joblib --csv data/targets_pa14.csv --out results/pa14_predictions.csv

# 4. Explain one prediction
protiq explain --model models/virulence.joblib --sequence MSEFFDRTG... --html lime.html
```

No internet? `protiq demo-data` writes a small synthetic dataset so you can try every command offline.

### All commands

| Command | What it does |
|---|---|
| `fetch` | Labeled training proteins from UniProt. `--scheme ec` gives enzyme class (EC 1 to 7, or non-enzyme). `--scheme virulence` gives virulence vs. non-virulence, with negatives drawn from the same pathogens. |
| `fetch-targets` | Uncharacterized proteins for any organism (default: *P. aeruginosa* PA14). |
| `train` | Train, evaluate on a held-out split, and save the model and metrics. |
| `predict` | Batch prediction from `--sequence`, `--fasta`, or `--csv`, with evidence and timing. |
| `explain` | LIME explanation plus nearest annotated proteins. `--html` saves the interactive chart. |
| `ablate` | Accuracy with each feature group removed in turn (sequence, structure, alphafold, physchem). |
| `sweep-k` | Accuracy for k = 1 to 20, with and without the logistic regression fallback. |
| `structure` / `structure-batch` | Download AlphaFold models and compute or cache structure features. |
| `blast` | NCBI remote BLAST for one sequence (slow; for exploration). |
| `foldseek-parse` | Read a Foldseek `.m8` results file. |

Useful training flags: `--backend diamond|kmer`, `--foldseek`, `--alphafold`, `--k 7`, `--metric cosine`, `--fallback-threshold 0.6`, `--weights sequence=1,structure=1.5`.

## Results

### Science fair results (2025)

On our original dataset, ProtiQ reached a peak accuracy of 88.9% with k = 5 to 7, and a stable 88.3% after ablation-guided optimization. Removing AlphaFold-derived structure information noticeably lowered accuracy. Most errors came from ambiguous annotations, especially within TIM-barrel enzyme families.

### This codebase on public data (September 2026)

The pipeline was rebuilt as a package and benchmarked on fresh UniProt Swiss-Prot data. Full details are in [`results/2026-09-27-benchmarks.md`](results/2026-09-27-benchmarks.md).

| Task | Test proteins | ProtiQ | Top BLAST hit | Majority class |
|---|---|---|---|---|
| Virulence vs. non-virulence (5,142 proteins, organism-matched negatives) | 1,029 | **90.0%** | 87.8% | 59.9% |
| Enzyme class, 8 classes (1,847 proteins) | 370 | **68.9%** | 65.1% | 14.1% |

- **k-sweep:** accuracy plateaus from k = 7 upward (90.0% to 90.4%). k = 1 overfits (87.0%). This matches the original finding.
- **Batch speed:** 326 PA14 proteins were predicted in 2.1 seconds with DIAMOND, compared with roughly an hour per protein for manual cross-analysis.
- **Example:** PA14 protein A0A0H2ZCC1 (listed as uncharacterized) is 99.5% identical to Tse4 (Q9I069), a type VI secretion toxin from *P. aeruginosa* PAO1. ProtiQ predicts Virulence, and the LIME explanation points to the sequence neighbor votes.
- **The hard part:** only 13 of 326 uncharacterized PA14 proteins have any sequence homolog among 5,000 reviewed reference proteins. Sequence alone cannot characterize the rest. This is the gap that structure search is meant to close, and it's the main focus of future work.

## Limitations

- Accuracy on a random train/test split is optimistic, because close homologs can land on both sides. A homology-clustered split (for example, CD-HIT or MMseqs2 at 30% identity) would be stricter.
- The virulence label comes from UniProt's Virulence keyword. That annotation is incomplete: many true virulence factors aren't tagged.
- AlphaFold confidence and compactness features added under 1% on these benchmarks. Foldseek structural-neighbor features are implemented and tested, but they haven't been benchmarked yet on a full reference set of AlphaFold models.
- Predictions for proteins with no homologs are low evidence and are labeled that way.

## Roadmap

- Benchmark the Foldseek structural arm against AlphaFold models of all reference proteins, and repeat the AlphaFold ablation with it.
- Use a homology-clustered train/test split.
- Scale the reference set to all of bacterial Swiss-Prot. DIAMOND handles this easily.
- Add protein language model embeddings as another feature group.
- Experimental follow-up on top PA14 candidates (knockouts, infection models), as outlined in our project.

## Repository layout

```
src/protiq/
  sequences.py      input validation, FASTA I/O, UniProt ID parsing
  uniprot.py        training data and target download (EC and virulence label schemes)
  homology.py       sequence search: DIAMOND and k-mer backends
  structure.py      AlphaFold download, pLDDT features, Foldseek search and parsing
  features.py       feature groups and neighbor voting
  model.py          kNN + logistic regression fallback, training, ablation, k-sweep, prediction
  explain.py        LIME explanations
  blast_remote.py   NCBI remote BLAST (exploratory)
  synthetic.py      synthetic data for offline tests
  cli.py            the `protiq` command
tests/              offline test suite (pytest)
notebooks/          Colab notebook
archive/            the original 2025 Colab notebooks
results/            benchmark write-ups
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests run offline. The DIAMOND and Foldseek tests run automatically when those tools are installed.

## Acknowledgments

Thank you to Dr. Fiona Brinkman and Dr. Amy Lee for mentorship, and to Mila Tkatchouk for guiding us through her pathogen-associated gene workflow.

## References

- van Kempen, M. et al. (2023). Fast and accurate protein structure search with Foldseek. *Nature Biotechnology*.
- Buchfink, B., Reuter, K., & Drost, H.-G. (2021). Sensitive protein alignments at tree-of-life scale using DIAMOND. *Nature Methods*.
- Varadi, M. et al. AlphaFold Protein Structure Database. https://alphafold.ebi.ac.uk
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD*.
- Lau, W. Y. V., Taylor, P. K., Brinkman, F. S. L., & Lee, A. H. Y. (2023). Pathogen-associated gene discovery workflows for novel antivirulence therapeutic development. *EBioMedicine*, 88, 104429.
- The UniProt Consortium. UniProt: the Universal Protein Knowledgebase. https://www.uniprot.org

## License

MIT. See [LICENSE](LICENSE).
